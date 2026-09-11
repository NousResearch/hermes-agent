"""Tests for hermes_cli/_scan_venv_blockers.py.

Tests call the real production functions (``main``, ``_redact_sensitive_cmdline``).
The detector is patched directly so no real process table interaction occurs.
"""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.redact as redact_module
from hermes_cli._scan_venv_blockers import (
    _classify_local_preview_args,
    _classify_pool_backend_args,
    _is_pausable_gateway,
    _probe_fail_json,
    _redact_sensitive_cmdline,
    _terminate_safe_preview,
    main,
)


# ---------------------------------------------------------------------------
# main() — stdout, stderr, exit code (with patched detector)
# ---------------------------------------------------------------------------


def _psutil_fake() -> dict:
    """Return a sys.modules dict entry that makes psutil appear available."""
    return {"psutil": types.SimpleNamespace(Process=lambda *a: MagicMock())}






# ---------------------------------------------------------------------------
# _redact_sensitive_cmdline
# ---------------------------------------------------------------------------


def test_redact_long_flag_value_space_separated() -> None:
    """--token SECRET must preserve --token and emit --token <redacted>."""
    raw = "python.exe -m hermes_cli.main serve --token ghp_abc123 --host 10.0.0.1"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m hermes_cli.main serve --token <redacted>"
    assert "ghp_abc123" not in result




def test_redact_sensitive_text_failure_returns_fully_redacted() -> None:
    """When agent.redact.redact_sensitive_text raises, the entire result
    must equal '<redacted>' so PID and name still provide diagnostics."""
    with patch.object(
        redact_module,
        "redact_sensitive_text",
        side_effect=RuntimeError("no redactor"),
    ):
        result = _redact_sensitive_cmdline("python.exe --token abc123")

    assert result == "<redacted>"


def test_redact_session_key() -> None:
    """--session-key <identifier> must redact the value and everything after."""
    raw = "python.exe -m tui_gateway.slash_worker --session-key 20260712-abcdef --model test"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m tui_gateway.slash_worker --session-key <redacted>"


def test_redact_normal_host_port_profile_remain() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 10.0.0.1 --port 9119 --profile work"
    result = _redact_sensitive_cmdline(raw)
    assert "10.0.0.1" in result
    assert "9119" in result
    assert "work" in result


def test_redact_no_sensitive_flags_is_noop() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 127.0.0.1"
    assert _redact_sensitive_cmdline(raw) == raw


def test_redact_empty_string() -> None:
    assert _redact_sensitive_cmdline("") == ""


def test_redact_short_flags_not_redacted() -> None:
    """Short flags -t (toolset), -p (profile), -k are NOT redacted."""
    raw = "python.exe -m hermes_cli.main serve -t web -p default -k somearg"
    result = _redact_sensitive_cmdline(raw)
    assert result == raw  # short flags pass through unchanged


def test_classify_local_preview_args_preserves_full_directory_label_and_port() -> None:
    args = [
        r"C:\Hermes\venv\Scripts\python.exe",
        "-m",
        "http.server",
        "8766",
        "--bind",
        "0.0.0.0",
        "--directory",
        r"C:\Projects\Example Preview",
    ]

    assert _classify_local_preview_args(args) == {
        "kind": "local-preview",
        "safeToStop": True,
        "label": "Example Preview",
        "port": 8766,
    }


def test_classify_local_preview_args_rejects_arbitrary_python_process() -> None:
    assert _classify_local_preview_args(["python.exe", "important-script.py"]) == {}


def test_classify_pool_backend_args_accepts_profile_scoped_serve() -> None:
    assert _classify_pool_backend_args(
        [r"C:\Hermes\venv\Scripts\python.exe", "-m", "hermes_cli.main", "--profile", "writer", "serve"]
    ) == {"kind": "pool-backend", "safeToStop": True, "label": "serve"}


def test_classify_pool_backend_args_rejects_gateway_and_arbitrary_python() -> None:
    assert _classify_pool_backend_args(
        [r"C:\Hermes\venv\Scripts\python.exe", "-m", "hermes_cli.main", "gateway", "run"]
    ) == {}
    assert _classify_pool_backend_args(["python.exe", "important-script.py"]) == {}


def test_classify_local_preview_args_rejects_module_flags_passed_to_a_script() -> None:
    assert _classify_local_preview_args(
        ["python.exe", "important-script.py", "-m", "http.server", "8765"]
    ) == {}


def test_terminate_safe_preview_revalidates_identity_and_exact_argv() -> None:
    class FakeProcess:
        def __init__(self, pid: int, *, created: float, args: list[str]) -> None:
            self.pid = pid
            self._created = created
            self._args = args
            self.terminated = False
            self.killed = False
            self._children: list[FakeProcess] = []

        def create_time(self) -> float:
            return self._created

        def cmdline(self) -> list[str]:
            return self._args

        def children(self, *, recursive: bool) -> list[FakeProcess]:
            assert recursive is True
            return self._children

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    child = FakeProcess(200, created=10.0, args=["child.exe"])
    parent = FakeProcess(100, created=1722798000.25, args=["python.exe", "-m", "http.server", "8766"])
    parent._children = [child]
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: parent if pid == 100 else child,
        wait_procs=lambda processes, timeout: (processes, []),
    )

    stopped, error = _terminate_safe_preview(100, 1722798000.25, psutil_module=fake_psutil)

    assert stopped is True
    assert error is None
    assert parent.terminated is True
    assert child.terminated is True


def test_terminate_safe_preview_accepts_leftover_serve_backend() -> None:
    class FakeProcess:
        def __init__(self, pid: int, *, created: float, args: list[str]) -> None:
            self.pid = pid
            self._created = created
            self._args = args
            self.terminated = False
            self._children: list[FakeProcess] = []

        def create_time(self) -> float:
            return self._created

        def cmdline(self) -> list[str]:
            return self._args

        def children(self, *, recursive: bool) -> list[FakeProcess]:
            return self._children

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            pass

    parent = FakeProcess(
        100,
        created=1722798000.25,
        args=["python.exe", "-m", "hermes_cli.main", "--profile", "writer", "serve"],
    )
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid: parent,
        wait_procs=lambda processes, timeout: (processes, []),
    )

    stopped, error = _terminate_safe_preview(100, 1722798000.25, psutil_module=fake_psutil)

    assert stopped is True
    assert error is None
    assert parent.terminated is True


def test_terminate_safe_preview_refuses_reused_pid() -> None:
    process = MagicMock()
    process.create_time.return_value = 1722798999.0
    fake_psutil = types.SimpleNamespace(Process=lambda _pid: process)

    stopped, error = _terminate_safe_preview(100, 1722798000.25, psutil_module=fake_psutil)

    assert stopped is False
    assert error == "process identity changed"
    process.cmdline.assert_not_called()
    process.terminate.assert_not_called()


# ---------------------------------------------------------------------------
# _is_pausable_gateway — the gateway exemption
#
# `hermes-setup` always invokes `hermes update --yes --gateway`, whose
# `_pause_windows_gateways_for_update()` stops running gateways itself. The
# Desktop preflight must therefore not report gateway launcher/worker chains
# as blockers — doing so aborts the handoff before the component that can
# handle them ever runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmdline",
    [
        # venv-side launcher, exactly as the scheduled task spawns it
        r"C:\Users\u\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
        " -m hermes_cli.main gateway run --replace",
        # uv-side worker re-running the same argv (quoted exe, double space)
        r'"C:\Users\u\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"'
        "  -m hermes_cli.main gateway run --replace",
        # profile-scoped gateway
        "python.exe -m hermes_cli.main --profile work gateway run",
        # a profile literally NAMED "gateway" — the profile value must not
        # shadow the subcommand token (the hand-rolled matcher regressed this)
        "python.exe -m hermes_cli.main --profile gateway gateway run",
        "python.exe -m hermes_cli.main -p gateway gateway run",
        # bare `gateway` defaults to `run` (mirrors the canonical matcher)
        "python.exe -m hermes_cli.main gateway",
        # case variations survive
        "PYTHON.EXE -m hermes_cli.main GATEWAY RUN",
    ],
)
def test_is_pausable_gateway_accepts_gateway_run_chains(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is True


@pytest.mark.parametrize(
    "cmdline",
    [
        # desktop backend: no pause machinery downstream, must keep blocking
        "python.exe -m hermes_cli.main serve --host 127.0.0.1 --port 8756",
        # other gateway subcommands are not running gateways
        "python.exe -m hermes_cli.main gateway stop",
        "python.exe -m hermes_cli.main gateway status",
        "python.exe -m hermes_cli.main gateway install",
        # operator REPL / stray script
        "python.exe",
        "python.exe myscript.py gateway run",  # not a hermes_cli.main invocation
        "",
    ],
)
def test_is_pausable_gateway_rejects_everything_else(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is False


def _run_main_with_detector(monkeypatch, capsys, matches):
    """Run main() with the process detector patched to return *matches*."""
    for name, mod in _psutil_fake().items():
        monkeypatch.setitem(sys.modules, name, mod)
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(update_cmd, "_detect_venv_python_processes", lambda: matches)
    with pytest.raises(SystemExit) as excinfo:
        main()
    out = capsys.readouterr().out
    return excinfo.value.code, json.loads(out)


def test_probe_fail_json_is_unambiguous_failure() -> None:
    """A failed probe must not look like a clear scan (#83149).

    Humans and naive callers used to read ``blocked: false`` as "no holders"
    when psutil was missing after a gutted venv. The document must mark
    ``probe_failed`` and keep ``ok`` false.
    """
    data = json.loads(_probe_fail_json("psutil is not available: No module named 'psutil'"))
    assert data["ok"] is False
    assert data["probe_failed"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert "psutil" in data["error"]


def test_main_psutil_missing_is_probe_failure_not_clear(monkeypatch, capsys):
    """Missing psutil exits non-zero with probe_failed JSON — never a clear scan."""
    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            raise ImportError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main()
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    data = json.loads(captured.out)
    assert data["ok"] is False
    assert data["probe_failed"] is True
    assert "psutil" in captured.err.lower()


def test_main_exempts_gateway_chain_but_keeps_other_holders(monkeypatch, capsys):
    """A gateway launcher/worker pair alone must scan clear; a non-gateway
    holder alongside it must still block (and be the only reported PID)."""
    gateway_launcher = (
        12,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main gateway run --replace",
    )
    gateway_worker = (
        34,
        "python.exe",
        r'"C:\u\uv\python\python.exe"  -m hermes_cli.main gateway run --replace',
    )
    stray_repl = (56, "python.exe", r"C:\x\venv\Scripts\python.exe")

    # Gateway chain only → clear
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker]
    )
    assert code == 0
    assert data["ok"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 2

    # Gateway chain + stray REPL → blocked, reporting only the REPL
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker, stray_repl]
    )
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [56]
    assert data["pausable_gateways"] == 2


def test_main_desktop_serve_backend_still_blocks(monkeypatch, capsys):
    """A `serve` backend with NO ledger identity must keep blocking —
    the updater's stop/relaunch rungs only own ledger-verified backends."""
    import hermes_cli.process_identity as pi

    monkeypatch.setattr(pi, "ledger_entries", lambda **kw: [])
    serve = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [serve])
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["pausable_gateways"] == 0
    assert data["deferred_backends"] == 0


# ---------------------------------------------------------------------------
# _is_updater_owned_backend — the serve/dashboard deferral (#98336)
# ---------------------------------------------------------------------------


_SERVE_CMD = r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1"


def _patch_ledger(monkeypatch, entries, dead):
    import hermes_cli.process_identity as pi

    monkeypatch.setattr(pi, "ledger_entries", lambda **kw: entries)
    monkeypatch.setattr(pi, "spawner_is_dead", lambda entry: dead)


def test_updater_owned_backend_dead_spawner_is_deferred(monkeypatch, capsys):
    """A ledger-verified serve whose spawner is provably dead is exactly the
    orphan `_ledger_reapable_backend_pids` reaps — the scan must defer it
    instead of dead-ending the hand-off (#98336)."""
    _patch_ledger(
        monkeypatch,
        [{"pid": 78, "purpose": "serve", "spawner_pid": 4242, "port": 9119}],
        dead=True,
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert code == 0
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["deferred_backends"] == 1
    # Sanitized decision evidence (#98350): structured ledger identity only —
    # the deferral must explain itself without echoing the command line.
    assert data["deferred_backend_evidence"] == [
        {"pid": 78, "purpose": "serve", "port": 9119}
    ]
    assert "--host" not in json.dumps(data["deferred_backend_evidence"])


def test_updater_owned_backend_unrecorded_spawner_is_deferred(monkeypatch, capsys):
    """spawner unrecorded (None from spawner_is_dead) but ledger-registered:
    the manual-serve rung stops and relaunches it — defer."""
    _patch_ledger(monkeypatch, [{"pid": 78, "purpose": "serve"}], dead=None)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is False
    assert data["deferred_backends"] == 1


def test_updater_owned_backend_live_foreign_spawner_still_blocks(monkeypatch, capsys):
    """A serve whose recorded spawner is ALIVE and not in this scan's
    ancestry (another supervisor would respawn it) must keep blocking."""
    from hermes_cli import _scan_venv_blockers as scan_mod

    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=False
    )
    monkeypatch.setattr(scan_mod, "_spawner_is_this_handoff_desktop", lambda entry: False)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_handoff_desktop_spawner_is_deferred(monkeypatch, capsys):
    """A serve owned by THIS hand-off's Desktop (the scan's own ancestor) is
    deferred: the Desktop exits before the updater runs, turning it into the
    dead-spawner orphan the ledger rung reaps (#98336 field case)."""
    from hermes_cli import _scan_venv_blockers as scan_mod

    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=False
    )
    monkeypatch.setattr(scan_mod, "_spawner_is_this_handoff_desktop", lambda entry: True)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is False
    assert data["deferred_backends"] == 1


def test_updater_owned_backend_purpose_mismatch_blocks(monkeypatch, capsys):
    """Argv says serve but the ledger entry says a different purpose — the
    identity is inconsistent; fail closed and keep blocking."""
    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "mcp-helper", "spawner_pid": 4242}], dead=True
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_non_backend_argv_never_deferred(monkeypatch, capsys):
    """A ledger entry cannot exempt a process whose argv is not a
    serve/dashboard subcommand (#90778 token rules: `kanban --preserve-cache`
    must not read as serve)."""
    _patch_ledger(
        monkeypatch, [{"pid": 78, "purpose": "serve", "spawner_pid": 4242}], dead=True
    )
    kanban = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main kanban --preserve-cache",
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [kanban])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_updater_owned_backend_ledger_failure_blocks(monkeypatch, capsys):
    """Ledger unreadable → unprovable → fail closed (pre-exemption behavior)."""
    import hermes_cli.process_identity as pi

    def _boom(**kw):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(pi, "ledger_entries", _boom)
    code, data = _run_main_with_detector(monkeypatch, capsys, [(78, "python.exe", _SERVE_CMD)])
    assert data["blocked"] is True
    assert data["deferred_backends"] == 0


def test_main_gateway_with_long_managed_runtime_path_is_exempt(monkeypatch, capsys):
    """Regression: the detector must hand the FULL cmdline to the exemption.

    Gateways launched via the managed-runtime interpreter carry a >120-char
    exe path (`.hermes-runtime\python\generation-...\cpython-3.11-...`).
    The old `cmdline_raw[:120]` truncation in the detector cut the cmdline
    before `-m hermes_cli.main gateway run`, so the exemption never matched
    and every Desktop update aborted with 'Update didn't finish'.
    Here the detector returns full cmdlines (post-fix contract); the scan
    must exempt the gateway and truncate only the *displayed* cmdline.
    """
    long_exe = (
        r'"C:\Users\u\AppData\Local\hermes\hermes-agent\.hermes-runtime\python'
        r"\generation-1785095035-66720-be29ea9c\cpython-3.11-windows-x86_64-none"
        r'\python.exe"'
    )
    assert len(long_exe) > 120  # the truncation point was inside the exe path
    gateway = (91, "python.exe", long_exe + "  -m hermes_cli.main gateway run --replace")
    code, data = _run_main_with_detector(monkeypatch, capsys, [gateway])
    assert code == 0
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 1

    # A long-path NON-gateway holder still blocks, with cmdline truncated for display.
    stray = (92, "python.exe", long_exe + "  -m some_other_module --serve-forever")
    code, data = _run_main_with_detector(monkeypatch, capsys, [gateway, stray])
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [92]
    assert len(data["processes"][0]["cmdline"]) <= 120
