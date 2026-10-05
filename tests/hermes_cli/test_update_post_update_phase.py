"""The post-update maintenance tail is announced around the best-effort loop (#126855).

``_print_post_update_notices_and_self_heals()`` runs after ``✓ Update complete!`` and can
keep the process busy for minutes (pinned cua-driver refresh, default PM tool installs).
The completion summary must not be the last thing the operator reads before a long silent
stretch: the tail announces itself before the step loop and prints a terminal marker after.
"""
from __future__ import annotations

import pytest

from hermes_cli import update_cmd_maint as maint

HEADER = "→ Finishing post-update maintenance (notices, self-heals; this can take a while)..."
MARKER = "✓ Post-update maintenance finished."
WARNING_MARKER = "⚠ Post-update maintenance finished with warnings."

ALL_STEPS = {
    "fts_notice", "fhs_guard", "curator_first", "curator_recent", "expose_cli",
    "windows_bin", "cua_refresh", "default_tools", "checkpoint_notice", "relay_migrate",
}


@pytest.fixture
def tail(monkeypatch):
    """Run the REAL tail loop with every step stubbed; ``run()`` returns the step log."""
    import hermes_cli._install_repair as repair
    import hermes_cli._launchers as launchers
    import hermes_cli.update_cmd as update_cmd

    class _MainRef:
        PROJECT_ROOT = "/unused"

    ran = []
    # Module-global steps resolve in maint's namespace at call time.
    for name, tag in (
        ("_print_fts_optimize_available_notice", "fts_notice"),
        ("_ensure_fhs_path_guard", "fhs_guard"),
        ("_refresh_cua_driver_after_update", "cua_refresh"),
        ("_install_default_tools_after_update", "default_tools"),
        ("_print_checkpoint_footprint_notice", "checkpoint_notice"),
        ("_migrate_relay_exporter_env", "relay_migrate"),
    ):
        monkeypatch.setattr(maint, name, lambda tag=tag: ran.append(tag))
    # Steps the tail imports lazily: patch at their source modules.
    monkeypatch.setattr(update_cmd, "_print_curator_first_run_notice", lambda: ran.append("curator_first"))
    monkeypatch.setattr(update_cmd, "_print_curator_recent_run_notice", lambda: ran.append("curator_recent"))
    monkeypatch.setattr(update_cmd, "_m", lambda: _MainRef)
    monkeypatch.setattr(launchers, "expose_cli", lambda root: ran.append("expose_cli"))
    monkeypatch.setattr(repair, "migrate_windows_bin_path", lambda root, **kw: ran.append("windows_bin"))

    def run(*, phase="post-update", **overrides):
        for name, step in overrides.items():
            monkeypatch.setattr(maint, name, step)
        maint._print_post_update_notices_and_self_heals(phase=phase)
        return ran

    return run


def test_tail_announces_maintenance_phase_around_steps(tail, capsys):
    ran = tail()
    out = capsys.readouterr().out
    assert HEADER in out, "the maintenance tail must announce itself after the summary"
    assert MARKER in out, "the maintenance tail must end with an explicit terminal marker"
    assert out.index(HEADER) < out.index(MARKER)
    # Every best-effort step ran inside the announced window.
    assert set(ran) == ALL_STEPS


def test_tail_marker_prints_when_a_step_raises(tail, capsys):
    def broken():
        raise RuntimeError("cua-driver refresh exploded")

    tail(_refresh_cua_driver_after_update=broken)
    out = capsys.readouterr().out
    assert HEADER in out
    assert WARNING_MARKER in out, "best-effort failures must not produce a success marker"
    assert MARKER not in out


def test_tail_accepts_install_phase(tail, capsys):
    tail(phase="install")
    out = capsys.readouterr().out
    assert "→ Finishing install maintenance" in out
    assert "✓ Install maintenance finished." in out
