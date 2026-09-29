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

import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import container_boot
import hermes_cli.service_manager as sm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_stale_pid_files(hermes_home: Path) -> dict[str, bytes]:
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
    """P0-2 helper contract: verdicts come from the output prefix only; the s6
    `up (pid N pgid N) ...` render is never pid-parsed (that regex broke before)."""
    monkeypatch.setattr(
        sm, "_s6_run", _fake_s6_run(0, "up (pid 32869 pgid 32869) 67020 seconds\n")
    )
    assert sm.slot_supervision_state(tmp_path, "gateway-default") == "LIVE"
    monkeypatch.setattr(sm, "_s6_run", _fake_s6_run(0, "down (5 seconds)\n"))
    assert sm.slot_supervision_state(tmp_path, "gateway-default") == "DOWN"
