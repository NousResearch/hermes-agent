"""Runtime-hardening guards for ``hermes_cli.container_boot`` (P0-1 / P0-2).

Platform-independent unit tests for the two reconcile guards:

- P0-1: ``main()`` refuses a full reconcile outside boot — with the PID
  anti-reuse identity check (``_pid_is_hermes_gateway``) distinguishing a live
  Hermes gateway (MATCH) from a stale/reused pid (STALE) and an unverifiable
  one (UNKNOWN).
- P0-2: ``_register_service`` refuses to rmtree an existing slot unless
  ``slot_supervision_state`` proves DOWN (fail-closed on LIVE and UNKNOWN).

All host interaction (``os.kill``, ``/proc/<pid>/cmdline``, ``s6-svstat``) is
monkeypatched, so these run identically on every host. The Linux-gated
integration-level behaviour of the same code lives in
``test_container_boot.py``.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import container_boot
import hermes_cli.service_manager as sm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_stale_pid_files(hermes_home: Path) -> dict[Path, bytes]:
    """Drop the runtime files a previous boot left on the persistent volume."""
    files = {
        hermes_home / "gateway.pid": b"999999\n",
        hermes_home / "processes.json": b"{}\n",
    }
    for path, data in files.items():
        path.write_bytes(data)
    return files


def _make_named_profile(hermes_home: Path, name: str) -> None:
    """Minimal named profile (SOUL.md marker) so reconcile sees it."""
    profile = hermes_home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "SOUL.md").write_text("# test profile\n", encoding="utf-8")


def _hermetic_env(
    monkeypatch: pytest.MonkeyPatch, hermes_home: Path, scandir: Path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("S6_PROFILE_GATEWAY_SCANDIR", str(scandir))
    # /proc scanning for main-wrapper.sh must never leak host processes into tests.
    monkeypatch.setattr(container_boot, "_read_container_argv", lambda: ())


def _fake_s6_run(returncode: int, stdout: str):
    def run(cmd: str, *args: str, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return run


# ---------------------------------------------------------------------------
# P0-1: full reconcile is boot-only (main() guard)
# ---------------------------------------------------------------------------


def test_main_refuses_full_reconcile_when_slots_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: any existing gateway-* slot means a supervision runtime is live (or
    was); main() must refuse and run zero cleanup/registration side effects."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    (scandir / "gateway-default").mkdir(parents=True)
    _seed_stale_pid_files(hermes_home)

    calls: list[str] = []
    monkeypatch.setattr(
        container_boot, "_register_service", lambda *a, **kw: calls.append("register")
    )
    monkeypatch.setattr(
        container_boot, "_cleanup_stale_runtime_files", lambda *a, **kw: calls.append("cleanup")
    )
    _hermetic_env(monkeypatch, hermes_home, scandir)

    rc = container_boot.main()

    assert rc != 0
    assert calls == [], f"side effects ran during refusal: {calls}"
    # The slot directory must be untouched (in particular: not rmtree'd).
    assert (scandir / "gateway-default").is_dir()


def test_main_boot_context_with_empty_scandir_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: empty service root and no confirmed live gateway → fresh boot proceeds
    and registers every slot."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    _hermetic_env(monkeypatch, hermes_home, scandir)

    rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()
    assert (scandir / "gateway-coder").is_dir()


def test_main_refusal_precedes_every_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3: on refusal the volume-persisted runtime files must be byte-identical —
    the guard fires before _cleanup_stale_runtime_files would unlink them."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    (scandir / "gateway-default").mkdir(parents=True)
    seeded = _seed_stale_pid_files(hermes_home)
    _hermetic_env(monkeypatch, hermes_home, scandir)

    rc = container_boot.main()

    assert rc != 0
    for path, data in seeded.items():
        assert path.read_bytes() == data, f"{path} was modified by the refused run"


def test_main_boot_proceeds_when_recorded_pid_is_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T9: a stale gateway.pid pointing at a non-existent process (the normal
    fresh-boot case across container recreation) must not block boot."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    dead_pid = 4_000_000 + os.getpid()
    (hermes_home / "gateway.pid").write_text(f"{dead_pid}\n")
    _hermetic_env(monkeypatch, hermes_home, scandir)
    monkeypatch.setattr(
        container_boot.os, "kill",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()
    # boot proceeded far enough to clean the stale runtime file
    assert not (hermes_home / "gateway.pid").exists()


def test_main_boot_proceeds_when_pid_reused_by_unrelated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T10: the recorded pid exists but its argv is an unrelated binary —
    PID reuse must not be mistaken for a live gateway."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    reused_pid = os.getpid()  # a definitely-alive pid
    (hermes_home / "gateway.pid").write_text(f"{reused_pid}\n")
    _hermetic_env(monkeypatch, hermes_home, scandir)
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        container_boot, "_cmdline_argv", lambda cmdline: ("/bin/sleep", "999")
    )

    rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()


def test_main_refuses_when_recorded_pid_is_a_real_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T11: a pid whose argv is a real gateway (root and named shapes) is a live
    gateway — main() refuses and runs zero side effects."""
    for shape, cmdline in (
        ("root", ("/opt/hermes/.venv/bin/hermes", "gateway", "run", "--replace")),
        ("named", ("/opt/hermes/.venv/bin/hermes", "-p", "coder", "gateway", "run", "--replace")),
    ):
        hermes_home = tmp_path / f"hermes-home-{shape}"
        hermes_home.mkdir()
        scandir = tmp_path / f"run-service-{shape}"
        _seed_stale_pid_files(hermes_home)
        (hermes_home / "gateway.pid").write_text(f"{os.getpid()}\n")
        _hermetic_env(monkeypatch, hermes_home, scandir)
        monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(
            container_boot, "_cmdline_argv", lambda cmdline_path, _c=cmdline: _c
        )

        calls: list[str] = []
        monkeypatch.setattr(
            container_boot, "_register_service", lambda *a, **kw: calls.append("register")
        )
        monkeypatch.setattr(
            container_boot,
            "_cleanup_stale_runtime_files",
            lambda *a, **kw: calls.append("cleanup"),
        )

        rc = container_boot.main()

        assert rc != 0, f"cmdline {cmdline} must be recognised as a live gateway"
        assert calls == []
        assert not (scandir / "gateway-default").exists()


def test_main_boot_proceeds_when_gateway_identity_unverifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T12: cmdline unreadable → UNKNOWN → warning, boot proceeds; UNKNOWN must
    never be escalated into a refusal."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text(f"{os.getpid()}\n")
    _hermetic_env(monkeypatch, hermes_home, scandir)
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)

    def unreadable(cmdline: Path) -> tuple[str, ...]:
        raise PermissionError(13, "proc cmdline hidden")

    monkeypatch.setattr(container_boot, "_cmdline_argv", unreadable)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.container_boot"):
        rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()
    assert any("identity cannot be verified" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# P0-1: _pid_is_hermes_gateway matcher (identity derived from _render_run_script)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ("/opt/hermes/.venv/bin/hermes", "gateway", "run", "--replace"),
        ("/opt/hermes/.venv/bin/hermes", "-p", "coder", "gateway", "run", "--replace"),
        ("/opt/hermes/.venv/bin/hermes", "gateway", "run"),  # stray manual gateway: protected too
    ],
)
def test_pid_matcher_matches_real_gateway_argv(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """T13: the argv shapes _render_run_script actually execs (plus the stray
    no---replace variant) all classify as MATCH; --replace is not required."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH"


@pytest.mark.parametrize(
    "argv",
    [
        ("/usr/bin/python", "-m", "someapp", "gateway", "run"),  # not hermes argv[0]
        ("/opt/hermes/.venv/bin/hermes", "gateway", "start"),  # dispatcher, not the gateway
        ("/opt/hermes/.venv/bin/hermes", "run"),  # missing the gateway subcommand
    ],
)
def test_pid_matcher_rejects_non_gateway_argv(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """T13: only `hermes gateway run ...` is a live gateway."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE"


def test_pid_matcher_unknown_on_unreadable_cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T12: unreadable /proc/<pid>/cmdline is UNKNOWN — never MATCH, never
    silently STALE."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)

    def unreadable(cmdline: Path) -> tuple[str, ...]:
        raise PermissionError(13, "hidden")

    monkeypatch.setattr(container_boot, "_cmdline_argv", unreadable)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "UNKNOWN"


# ---------------------------------------------------------------------------
# P0-2: pre-deletion supervision guard (LIVE / DOWN / UNKNOWN)
# ---------------------------------------------------------------------------


def test_register_service_refuses_live_supervised_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: an existing slot reported `up` by s6-svstat is LIVE — rmtree must be
    refused and the slot left byte-identical."""
    import hermes_cli.service_manager as sm
    from hermes_cli import container_boot

    monkeypatch.setattr(
        sm, "_s6_run", _fake_s6_run(0, "up (pid 1234 pgid 1234) 5 seconds\n")
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    existing = scandir / "gateway-coder"
    existing.mkdir(parents=True)
    (existing / "run").write_text("#!/command/with-contenv sh\n", encoding="utf-8")
    before = sorted(p.name for p in existing.rglob("*"))
    _make_named_profile(hermes_home, "coder")

    with pytest.raises(RuntimeError, match="LIVE supervision"):
        container_boot._register_service(scandir, "coder", start=False)

    assert existing.is_dir()
    assert sorted(p.name for p in existing.rglob("*")) == before


def test_register_service_rebuilds_down_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5a: `down` verdict → the existing rebuild contract holds."""
    import hermes_cli.service_manager as sm
    from hermes_cli import container_boot

    monkeypatch.setattr(sm, "_s6_run", _fake_s6_run(0, "down (5 seconds)\n"))
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    existing = scandir / "gateway-coder"
    existing.mkdir(parents=True)
    (existing / "stale").write_text("x", encoding="utf-8")
    _make_named_profile(hermes_home, "coder")

    container_boot._register_service(scandir, "coder", start=False)

    assert (scandir / "gateway-coder" / "run").exists()
    assert not (scandir / "gateway-coder" / "stale").exists()


@pytest.mark.parametrize(
    "exc_or_rc, stdout",
    [
        (FileNotFoundError(2, "s6-svstat"), ""),  # binary missing
        (OSError(5, "io error"), ""),  # subprocess OSError
        (subprocess.TimeoutExpired(cmd="s6-svstat", timeout=5), ""),  # hang
        (0, ""),  # empty output
        (1, "up (pid 1234 pgid 1234) 5 seconds\n"),  # rc!=0 → not a reliable up
        (0, "flagged\n"),  # unknown format
    ],
)
def test_register_service_fail_closed_on_unknown_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_or_rc: Exception | int,
    stdout: str,
) -> None:
    """T5b: every unverifiable s6-svstat outcome is UNKNOWN → fail closed —
    rmtree refused, slot untouched. Production code never fails open."""
    import hermes_cli.service_manager as sm
    from hermes_cli import container_boot

    if isinstance(exc_or_rc, Exception):
        def broken_run(cmd, *args, **kwargs):
            raise exc_or_rc

        monkeypatch.setattr(sm, "_s6_run", broken_run)
    else:
        monkeypatch.setattr(sm, "_s6_run", _fake_s6_run(exc_or_rc, stdout))

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    existing = scandir / "gateway-coder"
    existing.mkdir(parents=True)
    (existing / "run").write_text("#!/command/with-contenv sh\n", encoding="utf-8")
    before = sorted(p.name for p in existing.rglob("*"))
    _make_named_profile(hermes_home, "coder")

    with pytest.raises(RuntimeError, match="supervision"):
        container_boot._register_service(scandir, "coder", start=False)

    assert existing.is_dir()
    assert sorted(p.name for p in existing.rglob("*")) == before


def test_slot_supervision_state_prefix_only_no_pid_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-2 helper contract: the verdict comes from the FIRST token, by exact equality —
    the s6 `up (pid N pgid N) ...` render is never pid-parsed (that regex broke before),
    `upstream`-style renders must not read as up, nor `downstream`-style as down, and an
    unknown or empty format is UNKNOWN (fail-closed)."""
    for stdout, expected in (
        ("up (pid 32869 pgid 32869) 67020 seconds\n", "LIVE"),
        ("down (5 seconds)\n", "DOWN"),
        ("down", "DOWN"),
        ("upstream (pid 1)\n", "UNKNOWN"),
        ("downstream format v2\n", "UNKNOWN"),
        ("flagged\n", "UNKNOWN"),
        ("", "UNKNOWN"),
        ("\n", "UNKNOWN"),
    ):
        monkeypatch.setattr(sm, "_s6_run", _fake_s6_run(0, stdout))
        assert sm.slot_supervision_state(tmp_path, "gateway-default") == expected, stdout


# ---------------------------------------------------------------------------
# P1-A: the persisted PID record reaches the identity check intact
# ---------------------------------------------------------------------------


def _production_pid_record(python_argv: list[str] | None = None) -> dict:
    """A record in the exact schema gateway.status.write_pid_file persists."""
    return {
        "pid": os.getpid(),
        "kind": "hermes-gateway",
        "argv": python_argv if python_argv is not None else ["hermes", "gateway", "run"],
        "start_time": 12345678,
        "hermes_home": "/opt/data",
    }


@pytest.mark.parametrize(
    "record,label",
    [
        ({"pid": True, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "bool pid"),
        ({"pid": 123.9, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "float pid"),
        ({"pid": "123", "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "string pid"),
        ({"pid": 1e999, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "1e999 inf pid"),
        ({"pid": 0, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "pid 0"),
        ({"pid": -42, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "negative pid"),
        ({"pid": 10**40, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"]}, "oversized pid"),
        ({"pid": "not-a-number", "kind": "hermes-gateway"}, "non-numeric pid"),
        ({}, "record without a pid"),
    ],
)
def test_invalid_pid_records_are_skipped_without_probing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict,
    label: str,
) -> None:
    """pid 0 / negative / beyond the OS pid range (a JSON arbitrary-precision integer that
    would OverflowError inside os.kill) / non-numeric / bool / float / numeric string: an
    invalid record is "no evidence" — the identity check is never reached, boot is never
    crashed (P1-1 + P2). Strict JSON-integer typing, no int() coercion, no os.kill."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")
    _hermetic_env(monkeypatch, hermes_home, scandir)

    probed: list[int] = []
    monkeypatch.setattr(
        container_boot,
        "_pid_is_hermes_gateway",
        lambda pid: probed.append(pid) or "MATCH",
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    rc = container_boot.main()

    assert rc == 0, label
    assert probed == [], f"{label}: identity check must not run on an invalid record"
    assert kill_calls == [], f"{label}: os.kill must never see an out-of-range pid"
    assert (scandir / "gateway-default").is_dir()


@pytest.mark.parametrize(
    "raw,label",
    [
        ('{"pid": 1e999}', "1e999 parses to float inf"),
        ('{"pid": ' + "9" * 5000 + "}", "5000-digit integer (json digit limit)"),
        ('{"pid": 123.9}', "float pid literal"),
        ('{"pid": true}', "bool pid literal"),
        ('{"pid": "123"}', "string pid literal"),
    ],
)
def test_hostile_pid_records_cannot_crash_the_real_boot_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    label: str,
) -> None:
    """The two crash paths Codex proved on the REAL main() — ``{"pid": 1e999}`` (float inf
    → int() → OverflowError) and a beyond-the-digit-limit JSON integer (plain ValueError
    out of json.loads, not JSONDecodeError) — plus the strict-typing literals: every one
    collapses to invalid/stale record, the boot proceeds, identity is never consulted."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text(raw, encoding="utf-8")
    _hermetic_env(monkeypatch, hermes_home, scandir)

    probed: list[int] = []
    monkeypatch.setattr(
        container_boot,
        "_pid_is_hermes_gateway",
        lambda pid: probed.append(pid) or "MATCH",
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    rc = container_boot.main()  # must not raise

    assert rc == 0, label
    assert probed == [], label
    assert kill_calls == [], label
    assert (scandir / "gateway-default").is_dir(), label


def test_legacy_bare_integer_pid_file_still_reaches_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy bare-integer pid files stay compatible — on the production reader's own
    evidence (``_read_json_file(bare_pid_ok=True)`` accepts them as ``{"pid": N}``): the
    pid still reaches the identity check, as ``{"pid": N}`` with no argv to cross-check."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text("424242\n", encoding="utf-8")
    _hermetic_env(monkeypatch, hermes_home, scandir)

    seen: list[tuple[int, dict | None]] = []
    monkeypatch.setattr(
        container_boot,
        "_pid_is_hermes_gateway",
        lambda pid: seen.append((pid, container_boot._gateway_pid_record(
            hermes_home / "gateway.pid")[1])) or "STALE",
    )

    rc = container_boot.main()

    assert rc == 0
    assert seen == [(424242, {"pid": 424242})]


def test_production_pid_record_reaches_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE P1-A joint proof: a REAL ``gateway.status.write_pid_file()`` record (root and
    named profile) flows through the guard's reader into the identity check — the spy is
    called with the writer's real pid and the real JSON schema — and a MATCH verdict there
    refuses the reconcile with zero side effects. Under the pre-fix ``int(file.read_text())``
    reader this test fails: the JSON record never reaches the identity check at all
    (Codex probe: identity_calls=0, rc=0, slot created)."""
    from gateway import status as gateway_status

    for shape in ("root", "named"):
        hermes_home = tmp_path / f"hermes-home-{shape}"
        (hermes_home / "profiles" / "coder").mkdir(parents=True)
        (hermes_home / "profiles" / "coder" / "SOUL.md").write_text(
            "# test profile\n", encoding="utf-8"
        )
        scandir = tmp_path / f"run-service-{shape}"
        scandir.mkdir()
        monkeypatch.setenv(
            "HERMES_HOME",
            str(hermes_home / "profiles" / "coder" if shape == "named" else hermes_home),
        )
        gateway_status.write_pid_file()  # the REAL production writer, O_CREAT|O_EXCL JSON record
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # write_pid_file persists exactly one runtime file per home — gateway.pid at the
        # root for the root shape, inside profiles/coder/ for the named shape.
        pid_file = (
            hermes_home / "profiles" / "coder" / "gateway.pid"
            if shape == "named" else hermes_home / "gateway.pid"
        )
        seeded = {pid_file: pid_file.read_bytes()}
        _hermetic_env(monkeypatch, hermes_home, scandir)
        seen: list[int] = []
        monkeypatch.setattr(
            container_boot,
            "_pid_is_hermes_gateway",
            lambda pid: seen.append(pid) or "MATCH",
        )
        calls: list[str] = []
        monkeypatch.setattr(
            container_boot, "_register_service", lambda *a, **kw: calls.append("register")
        )
        monkeypatch.setattr(
            container_boot,
            "_cleanup_stale_runtime_files",
            lambda *a, **kw: calls.append("cleanup"),
        )

        rc = container_boot.main()

        assert rc != 0, f"{shape}: the real production record must refuse the reconcile"
        assert calls == []
        assert seen == [os.getpid()], f"{shape}: exactly the recorded pid reaches identity"
        # The REAL production schema (not a bare int) flowed through the guard's reader:
        probed_pid, probed_record = container_boot._gateway_pid_record(pid_file)
        assert probed_pid == os.getpid()
        assert probed_record is not None
        assert probed_record["pid"] == os.getpid()
        assert probed_record["kind"] == "hermes-gateway"
        assert isinstance(probed_record["argv"], list)
        assert "start_time" in probed_record and "hermes_home" in probed_record
        for path, data in seeded.items():
            assert path.read_bytes() == data, f"{shape}: {path} modified by the refused run"


def test_production_pid_record_with_real_matcher_still_boots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end with NO stubs on the identity path: a real production record naming THIS
    (pytest) process must not refuse the boot. On Linux the real /proc cmdline classifies
    pytest as STALE; on macOS (no /proc) as UNKNOWN + warning — both proceed. This test
    fails loudly if the matcher ever MATCHes the test process itself."""
    from gateway import status as gateway_status

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    gateway_status.write_pid_file()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _hermetic_env(monkeypatch, hermes_home, scandir)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.container_boot"):
        rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()
    assert (scandir / "gateway-coder").is_dir()
    if sys.platform != "linux":
        # macOS has no /proc/<pid>/cmdline: the honest verdict here is UNKNOWN + warning.
        assert any("identity cannot be verified" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# P1-B: the matcher recognises the launchers this repository actually ships
# ---------------------------------------------------------------------------


def _payload_launcher_cmdline(
    extra_args: tuple[str, ...],
    python: str = "/opt/hermes/tools/python/bin/python3",
) -> tuple[str, ...]:
    """The /proc argv of a gateway started by the REAL minted payload launcher.

    Renders ``scripts.build.launchers.posix_launcher`` for the pyproject ``hermes`` entry
    and shlex-splits its ``exec`` line — the exact derivation Codex used to prove the old
    matcher stale-classified every real gateway. Anti-drift: if the renderer's bootstrap
    ever changes shape, these command lines change with it and the matcher tests fail."""
    from scripts.build.launchers import posix_launcher

    script = posix_launcher(
        "hermes", "hermes_cli.main:main",
        python=python, repo=".", site=".", target="x86_64-unknown-linux-gnu",
    )
    exec_tokens = shlex.split(next(line for line in script.splitlines() if line.startswith("exec ")))
    c_index = exec_tokens.index("-c")
    # ["$PYTHON", "-P", "-c", "<bootstrap source>"] + the Hermes CLI argv
    return (python, *exec_tokens[2 : c_index + 2], *extra_args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ("gateway", "run", "--replace"),  # root slot (S6ServiceManager._render_run_script)
        ("-p", "coder", "gateway", "run", "--replace"),  # named slot
    ],
)
def test_pid_matcher_matches_real_payload_launcher(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: tuple[str, ...],
) -> None:
    """The payload launcher shapes the image actually execs (derived from the real
    renderer) are MATCH — the exact command lines Codex proved the pre-fix matcher
    mis-classified as STALE."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    cmdline = _payload_launcher_cmdline(extra_args)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: cmdline)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH"


def test_pid_matcher_rejects_arbitrary_python_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`python -P -c <anything else> gateway run` is NOT trusted as a gateway: only the
    exact Hermes payload bootstrap is (a stray `gateway run` tail is the inline program's
    data, #107002)."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    for source in (
        "import requests; requests.post('https://example.invalid')",
        "import os; os.system('id')",
        "import os, site, sys; sys.argv[0]='hermes'; something_else_entirely()",
    ):
        cmdline = ("/usr/bin/python3", "-P", "-c", source, "gateway", "run", "--replace")
        monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path, _c=cmdline: _c)
        assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE", source


@pytest.mark.parametrize(
    "argv",
    [
        # Codex second-review misses: every one must be recognised.
        ("/opt/venv/bin/python", "/opt/venv/bin/hermes", "gateway", "run", "--replace"),
        ("/opt/venv/bin/python", "/opt/venv/bin/hermes", "-p", "coder", "gateway", "run"),
        # profile flag AFTER the subcommand (argparse order-free top-level flags)
        ("/opt/hermes/.venv/bin/hermes", "gateway", "-p", "coder", "run"),
        ("/opt/hermes/.venv/bin/hermes", "gateway", "--profile=coder", "run"),
        # other value-taking top-level flags (real parser introspection)
        ("/opt/hermes/.venv/bin/hermes", "--model", "abc", "gateway", "run"),
        ("/opt/hermes/.venv/bin/hermes", "--reasoning", "high", "gateway", "run"),
        # interpreter flags before -m
        ("/usr/bin/python", "-P", "-m", "hermes_cli.main", "gateway", "run", "--replace"),
        ("/usr/bin/python", "-X", "utf8", "-m", "hermes_cli.main", "gateway", "run"),
    ],
)
def test_pid_matcher_matches_second_review_misses(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """The real entry/command shapes Codex proved the first-round matcher mis-classified:
    interpreter + console-script path, profile flags after the subcommand and
    ``--profile=`` anywhere, other value-taking top-level flags, and interpreter flags
    before ``-m`` — all are MATCH."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH", argv


@pytest.mark.parametrize(
    "argv",
    [
        ("/opt/hermes/gateway/run.py",),
        ("/opt/hermes/venv/lib/hermes/gateway/run.py",),
        ("/usr/local/bin/hermes-gateway",),
    ],
)
def test_pid_matcher_matches_gateway_dedicated_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """The gateway-dedicated entrypoints the production identity matcher waves through
    (gateway/run.py — shipped in-repo — and the hermes-gateway binary) ARE the runtime."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH", argv


@pytest.mark.parametrize(
    "source,label",
    [
        (
            "import sys, runpy; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)",
            "tampered runpy.run_module bootstrap",
        ),
        (
            "import os, sys, runpy; os.environ['X']='1'; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)",
            "store-launcher-shaped bootstrap",
        ),
        (
            "import sys, runpy; sys.path.insert(0, '/opt/site'); runpy.run_path('/x/hermes_cli/main.py', run_name='__main__')",
            "venv_sync-shaped bootstrap",
        ),
        (
            "import base64; exec(base64.b64decode('aW1wb3J0IGhlcm1lc19ib290c3RyYXA='))",
            "base64-encoded bootstrap",
        ),
        (
            "import os, site, sys; sys.argv[0]='hermes'; site.addsitedir(os.environ['HERMES_SITE']); from hermes_cli.main import main; sys.exit(main()) # extra",
            "official bootstrap with appended code",
        ),
    ],
)
def test_pid_matcher_rejects_fallback_and_tampered_bootstraps(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    label: str,
) -> None:
    """P2 tightening: "argv extractable" ≠ "Hermes identity". The inline_bootstrap_argv
    fallback family (runpy/store-launcher/venv_sync/base64) — which Codex proved accepts a
    tampered source — is no longer accepted at all; only the EXACT current renderer
    bootstrap is. An appended comment on the official bootstrap also refuses."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    argv = ("/usr/bin/python3", "-P", "-c", source, "gateway", "run", "--replace")
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE", label


@pytest.mark.parametrize(
    "argv",
    [
        ("hermes", "-p", "gateway", "run"),  # -p eats "gateway": command word is `run`
        ("hermes", "chat", "gateway", "run"),  # the pair belongs to chat's argparse
        ("hermes", "gateway", "start", "gateway", "run"),  # dispatcher, then its argument
        ("hermes", "gateway", "start"),
        ("hermes", "gateway", "status"),
    ],
)
def test_pid_matcher_rejects_command_structure_lookalikes(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """Command STRUCTURE decides, never a positional token search: the three lookalike
    shapes from the independent review (plus other non-run gateway subcommands) are all
    STALE despite containing `gateway run` / starting with `gateway`."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE", argv


@pytest.mark.parametrize(
    "argv",
    [
        ("/opt/hermes/.venv/bin/python", "-m", "hermes_cli.main", "gateway", "run", "--replace"),
        ("/opt/hermes/.venv/bin/python", "-m", "hermes_cli.main", "-p", "coder", "gateway", "run"),
        ("/opt/hermes/.venv/bin/hermes_cli/main.py", "gateway", "run"),
    ],
)
def test_pid_matcher_matches_module_and_script_path_forms(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    """`python -m hermes_cli.main` and the `hermes_cli/main.py` script-path form are real
    Hermes entry shapes (the production matcher accepts both spellings) and are MATCH."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH", argv


def test_pid_matcher_rejects_non_hermes_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    argv = ("/usr/bin/python", "-m", "hermes_cli.chat", "gateway", "run")
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: argv)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE"


# ---------------------------------------------------------------------------
# P1-B auxiliary cross-validation + P2 kill/cmdline error sealing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record,label",
    [
        (
            {"pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "chat"]},
            "stale non-gateway argv",
        ),
        ({"pid": os.getpid(), "argv": ["hermes", "gateway", "run"]}, "missing kind"),
        ({"pid": os.getpid(), "kind": "hermes-gateway"}, "no argv"),
        ({"pid": os.getpid(), "kind": "something-else", "argv": None, "start_time": 1}, "corrupt fields"),
    ],
)
def test_stale_or_corrupt_record_cannot_demote_a_confirmed_live_gateway(
    monkeypatch: pytest.MonkeyPatch,
    record: dict,
    label: str,
) -> None:
    """P2: the live /proc cmdline is the SOLE identity authority. When it already proves
    `hermes gateway run`, a stale/corrupted record (old argv, missing kind, junk fields)
    must NOT demote the verdict — the record only ever supplies the pid."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    cmdline = ("/opt/hermes/.venv/bin/hermes", "gateway", "run", "--replace")
    monkeypatch.setattr(container_boot, "_cmdline_argv", lambda cmdline_path: cmdline)
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "MATCH", label


def test_record_claiming_gateway_cannot_upgrade_a_non_gateway_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 (mirror side): a record whose argv/kind claims a gateway must never upgrade a
    live NON-gateway process — the record alone is not identity evidence."""
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        container_boot, "_cmdline_argv", lambda cmdline_path: ("/usr/bin/vim", "notes.txt")
    )
    claiming = {
        "pid": os.getpid(), "kind": "hermes-gateway",
        "argv": ["hermes", "gateway", "run", "--replace"], "start_time": 1,
        "hermes_home": "/opt/data",
    }
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE"


def test_pid_reuse_by_non_gateway_process_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale PID reuse settles on the live process identity: a real production record
    naming a pid that now runs an unrelated binary is STALE — the fresh boot proceeds."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text(
        json.dumps({
            "pid": os.getpid(), "kind": "hermes-gateway",
            "argv": ["hermes", "gateway", "run", "--replace"],
            "start_time": 1, "hermes_home": str(hermes_home),
        }),
        encoding="utf-8",
    )
    _hermetic_env(monkeypatch, hermes_home, scandir)
    monkeypatch.setattr(container_boot.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        container_boot, "_cmdline_argv", lambda cmdline_path: ("/bin/sleep", "999")
    )

    rc = container_boot.main()

    assert rc == 0
    assert (scandir / "gateway-default").is_dir()


@pytest.mark.parametrize(
    "kill_exc,label",
    [
        (OverflowError("Python int too large to convert to C int"), "OverflowError"),
        (OSError(5, "EIO"), "other OSError"),
        (ValueError("bad sig"), "ValueError"),
    ],
)
def test_pid_kill_errors_sealed_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kill_exc: Exception,
    label: str,
) -> None:
    """Every unexpected os.kill failure is UNKNOWN → warning + boot proceeds; a malformed
    persisted record must never crash main() (P2; Codex reproduced the OverflowError
    crash with a 10**30 pid)."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_named_profile(hermes_home, "coder")
    (hermes_home / "gateway.pid").write_text(str(os.getpid()), encoding="utf-8")
    _hermetic_env(monkeypatch, hermes_home, scandir)

    def exploding_kill(pid: int, sig: int) -> None:
        raise kill_exc

    monkeypatch.setattr(container_boot.os, "kill", exploding_kill)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.container_boot"):
        rc = container_boot.main()

    assert rc == 0, label
    assert (scandir / "gateway-default").is_dir(), label
    assert any("identity cannot be verified" in r.message for r in caplog.records), label


def test_pid_kill_permission_error_still_reads_cmdline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EPERM on os.kill is not UNKNOWN by itself: the /proc argv read still decides
    (here: an unrelated binary → STALE)."""
    monkeypatch.setattr(
        container_boot.os, "kill",
        lambda pid, sig: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")),
    )
    monkeypatch.setattr(
        container_boot, "_cmdline_argv", lambda cmdline_path: ("/usr/bin/vim", "notes.txt")
    )
    assert container_boot._pid_is_hermes_gateway(os.getpid()) == "STALE"


def test_pid_zero_and_negative_are_rejected_before_the_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pid 0 (process group) and negatives (process groups) are not single-process pids:
    the reader rejects them before any os.kill probe."""
    assert container_boot._gateway_pid_record(Path("/nonexistent/gateway.pid")) == (None, None)

    probe_targets: list[int] = []

    def spying_kill(pid: int, sig: int) -> None:
        probe_targets.append(pid)

    monkeypatch.setattr(container_boot.os, "kill", spying_kill)
    assert container_boot._pid_is_hermes_gateway(0) == "STALE"
    assert container_boot._pid_is_hermes_gateway(-1) == "STALE"
    assert probe_targets == []
