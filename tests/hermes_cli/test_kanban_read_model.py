"""Tests for the `hermes kanban read-model` command — H0.2a + H0.2b3a.

H0.2a covers the CLI grammar (`--kanban-root` and `--board` at the kanban
level, the `read-model` subcommand) and a dedicated early dispatch in
`kanban_command()` that runs before the delegated-mutation check, `--board`
override resolution, and the auto `kb.init_db()`.

H0.2b3a replaces the rejected live-SQLite read model with an owner-published
JSON artifact read by :mod:`hermes_cli.kanban_read_model`. The earlier
`_validated_read_model_db_path` design — which derived and `lstat`-ed
`<root>/kanban.db` — was rejected outright: H0 must never construct, stat,
open, or import its way to the live database, its WAL, or its SHM. Those
tests are gone with it; what remains here is the CLI surface:

  * explicit `--kanban-root`, `--board` and `--artifact` are all required
    (rc=2, a usage error, with no ambient HERMES_*/current-board fallback);
  * every post-usage failure is rc=1 with exactly
    `kanban read-model: unavailable` and empty stdout;
  * a successful read prints only allowlisted, sanitized fields.

Artifact-level behaviour (the openat/O_NOFOLLOW walk, the fstat guards, the
strict JSON contract, binding and freshness) is covered in
``test_kanban_read_model_artifact.py``.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_read_model as krm


def _parse_kanban_args(cli_args):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(cli_args)


def _forbid_generic_kanban_helpers(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "read-model must dispatch before any generic init/board-"
            "resolution helper is called"
        )

    monkeypatch.setattr(kc.kb, "init_db", _boom)
    monkeypatch.setattr(kc.kb, "board_exists", _boom)
    monkeypatch.setattr(kc.kb, "scoped_current_board", _boom)


def test_read_model_command_parses_every_explicit_input():
    """Target grammar: explicit `--kanban-root` and `--board`, then
    `read-model --artifact <path> --max-age-seconds <n> --limit <n> --json`.
    None of these may be resolved from the environment or the current board."""
    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", "/approved/canonical/root",
            "--board", "acme",
            "read-model",
            "--artifact", "/published/read-model.json",
            "--max-age-seconds", "900",
            "--limit", "50",
            "--json",
        ]
    )

    assert args.kanban_action == "read-model"
    assert args.kanban_root == "/approved/canonical/root"
    assert args.board == "acme"
    assert args.artifact == "/published/read-model.json"
    assert args.max_age_seconds == 900
    assert args.limit == 50
    assert args.json is True


def test_read_model_max_age_defaults_to_the_reader_constant():
    """`--max-age-seconds` is optional and falls back to one documented
    constant — not to an environment variable or a profile setting."""
    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", "/approved/canonical/root",
            "--board", "acme",
            "read-model",
            "--artifact", "/published/read-model.json",
        ]
    )

    assert args.max_age_seconds == krm.DEFAULT_MAX_AGE_SECONDS


@pytest.mark.parametrize(
    "cli_args",
    [
        ["kanban", "--board", "acme", "read-model", "--artifact", "/p/a.json"],
        ["kanban", "--kanban-root", "/root", "read-model", "--artifact", "/p/a.json"],
        ["kanban", "--kanban-root", "/root", "--board", "acme", "read-model"],
        ["kanban", "read-model"],
    ],
    ids=["missing-root", "missing-board", "missing-artifact", "missing-everything"],
)
def test_read_model_requires_root_board_and_artifact(cli_args, monkeypatch, capsys):
    """Omitting any of the three explicit inputs is a usage error (rc=2),
    rejected by the early read-model dispatcher itself — before any DB, root,
    or board-resolution helper runs, and with no ambient fallback."""
    args = _parse_kanban_args(cli_args)

    _forbid_generic_kanban_helpers(monkeypatch)

    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 2
    assert (
        captured.err.strip()
        == "kanban read-model: explicit --kanban-root, --board and --artifact are required"
    )
    assert captured.out == ""


def _valid_args(**overrides):
    cli_args = [
        "kanban",
        "--kanban-root", overrides.get("kanban_root", "/approved/canonical/root"),
        "--board", overrides.get("board", "acme"),
        "read-model",
        "--artifact", overrides.get("artifact", "/published/read-model.json"),
    ]
    return _parse_kanban_args(cli_args)


_PAYLOAD = {
    "schema_version": krm.SCHEMA_VERSION,
    "generated_at": 1_700_000_000,
    "board": "acme",
    "truncated": False,
    "tasks": [],
}


def test_read_model_dispatches_before_generic_init_and_board_resolution(monkeypatch, capsys):
    """`read-model` must be handled by a dedicated early dispatch in
    `kanban_command()`, reachable before the delegated-mutation check,
    `--board` override resolution, and the auto `kb.init_db()` call."""
    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", lambda **kwargs: dict(_PAYLOAD))

    assert kc.kanban_command(_valid_args()) == 0
    assert capsys.readouterr().err == ""


def test_read_model_passes_exactly_the_explicit_cli_inputs_to_the_reader(monkeypatch):
    """The reader is called with the explicit CLI inputs and nothing else —
    no ambient root, board, artifact, or freshness window."""
    calls = []

    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(
        krm, "read_read_model", lambda **kwargs: calls.append(kwargs) or dict(_PAYLOAD)
    )

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", "/approved/canonical/root",
            "--board", "acme",
            "read-model",
            "--artifact", "/published/read-model.json",
            "--max-age-seconds", "900",
            "--limit", "7",
        ]
    )
    kc.kanban_command(args)

    assert calls == [
        {
            "kanban_root": "/approved/canonical/root",
            "board": "acme",
            "artifact": "/published/read-model.json",
            "max_age_seconds": 900,
            "limit": 7,
        }
    ]


@pytest.mark.parametrize(
    "case, exception",
    [
        ("declared-unavailable", krm.ReadModelUnavailable),
        ("unexpected-runtime-error", RuntimeError),
        ("unexpected-os-error", OSError),
        ("unexpected-type-error", TypeError),
        ("unexpected-value-error", ValueError),
    ],
)
def test_read_model_reports_one_generic_unavailable_for_every_reader_failure(
    case, exception, monkeypatch, capsys
):
    """Every post-usage failure — declared or not — is rc=1 with exactly
    `kanban read-model: unavailable` and empty stdout. The raw exception text
    may embed the artifact path, the root, or the reason a probe succeeded, so
    none of it may reach the terminal: a caller must not be able to use
    read-model as a filesystem oracle."""
    sentinel = "/sentinel/elFO0P-9f3a1c/published/read-model.json"

    def _boom(**kwargs):
        raise exception(f"failed reading {sentinel}")

    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", _boom)

    rc = kc.kanban_command(_valid_args())

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.strip() == "kanban read-model: unavailable"
    assert captured.out == ""
    assert "sentinel" not in captured.err + captured.out
    assert "elFO0P" not in captured.err + captured.out


def test_read_model_reports_unavailable_for_a_missing_root_without_leaking_it(
    tmp_path, monkeypatch, capsys
):
    """With the real reader (no mock) and a missing explicit root: the same
    generic result, no path leakage, and not a single filesystem write."""
    sentinel = "missing-root-9f3a1c"
    root = tmp_path / sentinel
    assert not root.exists()

    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    _forbid_generic_kanban_helpers(monkeypatch)
    rc = kc.kanban_command(
        _valid_args(kanban_root=str(root), artifact=str(tmp_path / "published" / "a.json"))
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.strip() == "kanban read-model: unavailable"
    assert captured.out == ""
    assert sentinel not in captured.err + captured.out
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert not root.exists()


_TASK = {
    "id": "t-0001",
    "title": "Ship the read model",
    "status": "ready",
    "assignee": "thomas",
    "priority": 3,
    "created_at": 1_699_999_100,
    "started_at": None,
    "completed_at": None,
}


def test_read_model_json_output_is_exactly_the_allowlisted_payload(monkeypatch, capsys):
    """`--json` prints one JSON document built from the allowlist — never the
    artifact's own bytes, which are attacker-influenced and may contain
    anything the publisher was tricked into writing."""
    payload = dict(_PAYLOAD, tasks=[dict(_TASK)])

    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", lambda **kwargs: payload)

    args = _parse_kanban_args(
        [
            "kanban",
            "--kanban-root", "/approved/canonical/root",
            "--board", "acme",
            "read-model",
            "--artifact", "/published/read-model.json",
            "--json",
        ]
    )
    rc = kc.kanban_command(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out) == payload


def test_read_model_text_output_lists_only_allowlisted_task_fields(monkeypatch, capsys):
    """Without `--json`, one header line plus one line per task."""
    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(
        krm, "read_read_model", lambda **kwargs: dict(_PAYLOAD, tasks=[dict(_TASK)])
    )

    rc = kc.kanban_command(_valid_args())

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert rc == 0
    assert len(lines) == 2
    assert "acme" in lines[0]
    assert "t-0001" in lines[1] and "ready" in lines[1] and "Ship the read model" in lines[1]
    assert "{" not in captured.out


@pytest.mark.parametrize("use_json", [True, False], ids=["json", "text"])
@pytest.mark.parametrize(
    "case, payload",
    [
        ("unknown-top-level-key", dict(_PAYLOAD, session_id="s-1", tasks=[])),
        ("unknown-task-key", dict(_PAYLOAD, tasks=[dict(_TASK, body="private worklog")])),
        ("missing-task-key", dict(_PAYLOAD, tasks=[{"id": "t-0001"}])),
    ],
)
def test_read_model_refuses_to_print_a_payload_outside_the_allowlist(
    case, payload, use_json, monkeypatch, capsys
):
    """The CLI re-checks the allowlist before printing instead of trusting the
    reader. If a payload ever carried a field outside it — `body`,
    `session_id`, a workspace path — the command fails closed rather than
    printing it, and stdout stays completely empty."""
    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", lambda **kwargs: payload)

    cli_args = [
        "kanban",
        "--kanban-root", "/approved/canonical/root",
        "--board", "acme",
        "read-model",
        "--artifact", "/published/read-model.json",
    ]
    if use_json:
        cli_args.append("--json")

    rc = kc.kanban_command(_parse_kanban_args(cli_args))

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err.strip() == "kanban read-model: unavailable"
    assert "session_id" not in captured.err
    assert "worklog" not in captured.err


@pytest.mark.parametrize("use_json", [True, False], ids=["json", "text"])
def test_read_model_refuses_to_print_output_over_the_serialized_byte_cap(
    use_json, monkeypatch, capsys
):
    """The complete serialized UTF-8 payload is capped before anything is
    written. A caller reading this command's stdout has its own bounded
    buffer, and a payload that overruns it gets cut mid-document — which is
    worse than no answer, because a truncated JSON object is not detectably
    truncated. Overflow is the same generic failure as any other, with stdout
    left completely empty rather than partially written."""
    oversized = dict(_PAYLOAD, tasks=[dict(_TASK, title="t" * (512 * 1024))])

    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", lambda **kwargs: oversized)

    cli_args = [
        "kanban",
        "--kanban-root", "/approved/canonical/root",
        "--board", "acme",
        "read-model",
        "--artifact", "/published/read-model.json",
    ]
    if use_json:
        cli_args.append("--json")

    rc = kc.kanban_command(_parse_kanban_args(cli_args))

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err.strip() == "kanban read-model: unavailable"


class _WriteRecorder:
    """A stdout stand-in that records every individual write call."""

    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)
        return len(text)

    def flush(self):
        pass


def test_read_model_writes_successful_output_in_exactly_one_stdout_write(monkeypatch):
    """A successful read reaches stdout as ONE write of the whole validated
    payload. The consumer reads this stream incrementally with a byte cap, so
    a payload that arrives in several writes can be observed — and cut — part
    way through; and `print` alone emits the body and the newline separately.
    Validating the complete payload and then writing it once means the
    consumer sees either all of it or none of it."""
    recorder = _WriteRecorder()
    _forbid_generic_kanban_helpers(monkeypatch)
    monkeypatch.setattr(krm, "read_read_model", lambda **kwargs: dict(_PAYLOAD, tasks=[dict(_TASK)]))
    monkeypatch.setattr(kc.sys, "stdout", recorder)

    rc = kc.kanban_command(_valid_args())

    assert rc == 0
    assert len(recorder.writes) == 1
    assert recorder.writes[0].endswith("\n")
    assert "t-0001" in recorder.writes[0]
