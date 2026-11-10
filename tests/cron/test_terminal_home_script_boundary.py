"""Cron's actual script process sees destination HOME, identity and scratch."""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def unpinned_temp_environment(monkeypatch):
    # The canonical runner may own TMPDIR without a Hermes marker. These HOME
    # cases exercise Hermes-default scratch, not the explicit-user-temp override.
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.delenv(key, raising=False)



@pytest.mark.platforms("posix")
@pytest.mark.parametrize("reverse", [False, True], ids=["default-to-secondary", "secondary-to-default"])
@pytest.mark.parametrize("owner_mode", ["real", "profile", None])
def test_cron_script_owner_home(tmp_path, monkeypatch, owner_mode, reverse):
    from cron.scheduler_script import _run_job_script
    from cron.scheduler import _fire_secret_scope
    import hermes_constants as hc
    from tools.terminal_scope import install_and_reset_profile_terminal_scope, get_terminal_scope

    real = tmp_path / "real"
    root = tmp_path / ".hermes"
    secondary = root / "profiles" / "secondary"
    real.mkdir()
    (root / "home").mkdir(parents=True)
    (secondary / "home").mkdir(parents=True)
    launch, owner = (secondary, root) if reverse else (root, secondary)
    (owner / "scripts").mkdir()
    (owner / "config.yaml").write_text("terminal:\n  backend: local\n" + (f"  home_mode: {owner_mode}\n" if owner_mode else ""))
    (launch / ".env").write_text("LAUNCH_PRIVATE=launch-only\n")
    (owner / ".env").write_text("TARGET_MARKER=target-value\n")
    script = owner / "scripts" / "inspect.sh"
    script.write_text('printf "%s\\n%s\\n%s\\n%s\\n%s\\n[%s]\\n" "$HOME" "$HERMES_HOME" "$HERMES_REAL_HOME" "$TMPDIR" "$TARGET_MARKER" "$LAUNCH_PRIVATE"\n')
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", str(launch))
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("LAUNCH_PRIVATE", "launch-only")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "real" if owner_mode == "profile" else "profile")
    before = dict(os.environ)
    previous_scope = get_terminal_scope()
    home_token = hc.set_hermes_home_override(str(owner))
    try:
        # The in-process fire installs these scopes before invoking script execution.
        with _fire_secret_scope(), install_and_reset_profile_terminal_scope(owner):
            ok, output = _run_job_script(str(script))
            assert ok, output
            expected_profile = owner_mode == "profile" or (owner_mode is None and hc.is_container())
            assert output.strip().splitlines() == [str(owner / "home" if expected_profile else real), str(owner),
                str(real), str(owner / "cache" / "scratch"), "target-value", "[]"]
        assert get_terminal_scope() is previous_scope
    finally:
        hc.reset_hermes_home_override(home_token)
    assert dict(os.environ) == before
