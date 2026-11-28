"""plist Standard*Path must never sit on a volume xpcproxy cannot open pre-exec.

2026-11-13, HERMES_HOME on an external NVMe APFS volume: xpcproxy opens the plist stdio
files BEFORE exec'ing the job, that pre-exec open of an external-volume path is
TCC-denied (EPERM), posix_spawn never runs the job, and xpcproxy exits 78 on every
respawn — masquerading as the gateway's own EX_CONFIG 78 (FDA grants to the wrapped
executables do not help: the denied open happens before they exist). The plist handles
only capture pre-redirection output; the real logs keep appending under
HERMES_HOME/logs through the JXA ``>>`` redirections, which are post-exec opens by the
job's own processes and stay put.
"""

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import _launchers, gateway_launchd


def _patch_store_probe(monkeypatch):
    """generation reads the PM store via <tree>/../manifest.json; a checkout that lives
    inside the real HERMES_HOME makes that guarded home I/O on this host, and the store is
    not part of this contract."""
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root, **kw: None)


class TestLaunchdStdioPaths:
    def test_stdio_moves_off_external_volume(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gateway_launchd, "_paths_share_one_volume", lambda a, b: False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        _patch_store_probe(monkeypatch)

        plist = gateway_cli.generate_launchd_plist()
        boot_logs = tmp_path / "Library" / "Logs"
        label = gateway_launchd.get_launchd_label()

        assert f"<string>{boot_logs / f'{label}.stdio.log'}</string>" in plist
        assert f"<string>{boot_logs / f'{label}.stdio.err'}</string>" in plist
        # The JXA redirections are post-exec opens by the job's own processes: they stay put.
        hermes_logs = gateway_cli.get_hermes_home() / "logs"
        assert str(hermes_logs / "gateway.log") in plist
        assert str(hermes_logs / "gateway.error.log") in plist

    def test_stdio_keeps_log_dir_paths_on_the_home_volume(self, monkeypatch):
        """Default installs (HERMES_HOME on the boot volume) keep the historical stdio
        paths — the redirect is only for homes xpcproxy cannot open pre-exec."""
        monkeypatch.setattr(gateway_launchd, "_paths_share_one_volume", lambda a, b: True)
        _patch_store_probe(monkeypatch)

        plist = gateway_cli.generate_launchd_plist()
        hermes_logs = gateway_cli.get_hermes_home() / "logs"

        assert f"<string>{hermes_logs / 'gateway.log'}</string>" in plist
        assert f"<string>{hermes_logs / 'gateway.error.log'}</string>" in plist
        assert ".stdio." not in plist

    def test_volume_probe_counts_unstattable_paths_as_same_volume(self, tmp_path):
        """A volume that cannot be probed must not reroute stdio: the caller keeps its
        original paths rather than guessing."""
        missing = tmp_path / "missing"
        assert gateway_launchd._paths_share_one_volume(missing, tmp_path / "other-missing")
        assert gateway_launchd._paths_share_one_volume(tmp_path, missing)


@pytest.mark.parametrize(
    ("probe_result", "expected"),
    [(True, "hermes_home"), (False, "boot_volume")],
)
def test_stdio_pair_follows_the_volume_probe(monkeypatch, tmp_path, probe_result, expected):
    """The pair routes solely on the volume probe: no probe, no reroute."""
    monkeypatch.setattr(gateway_launchd, "_paths_share_one_volume", lambda a, b: probe_result)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stdout_log, stderr_log = tmp_path / "logs" / "gateway.log", tmp_path / "logs" / "gateway.error.log"
    (tmp_path / "logs").mkdir()

    out, err = gateway_launchd._launchd_stdio_log_paths(stdout_log, stderr_log, "ai.hermes.gateway")

    if expected == "hermes_home":
        assert (out, err) == (stdout_log, stderr_log)
    else:
        boot = tmp_path / "Library" / "Logs"
        assert (out, err) == (boot / "ai.hermes.gateway.stdio.log", boot / "ai.hermes.gateway.stdio.err")


def test_reload_helper_stdio_parks_off_external_volume(monkeypatch, tmp_path):
    """launchctl submit's -o/-e are launchd pre-exec opens: on an external-volume HERMES_HOME they
    are TCC-denied and the one-shot reload job dies with xpcproxy's 78 without running a line, so a
    deferred plist reload never happens (2026-11-15). They must park on the boot volume; the
    script's own >> appends keep the canonical HERMES_HOME log."""
    monkeypatch.setattr(gateway_launchd, "_paths_share_one_volume", lambda a, b: False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    captured = {}
    monkeypatch.setattr(
        gateway_launchd.subprocess, "Popen",
        lambda argv, **kw: captured.setdefault("argv", argv),
    )

    spawned = gateway_launchd._spawn_deferred_launchd_reload(
        domain="gui/501", label="ai.hermes.gateway", target="gui/501/ai.hermes.gateway",
        plist_path=tmp_path / "ai.hermes.gateway.plist", gateway_pid=4242,
    )

    assert spawned
    argv = captured["argv"]
    boot = tmp_path / "Library" / "Logs"
    assert argv[argv.index("-o") + 1] == str(boot / "ai.hermes.gateway.reload.stdio.log")
    assert argv[argv.index("-e") + 1] == str(boot / "ai.hermes.gateway.reload.stdio.err")
    # The script appends its progress to the canonical HERMES_HOME log (post-exec opens).
    assert str(gateway_cli.get_hermes_home() / "logs" / "launchd-reload.log") in argv[-1]
