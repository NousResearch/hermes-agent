"""`hermes update` must self-heal the ``hermes-acp`` launcher.

ACP hosts (Zed, JetBrains, Buzz Desktop) resolve the agent by the
``hermes-acp`` command name on the login-shell PATH. Fresh installs get the
launcher from ``scripts/install.sh``; existing installs get it from
``_ensure_acp_launcher()`` during ``hermes update``.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main import _ensure_acp_launcher, _ensure_hermes_agent_launcher


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    return bin_dir








def test_does_not_follow_symlink_into_venv(fake_home, tmp_path):
    """#21454 failure mode: never write through a symlinked hermes-acp."""
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    console_script = tmp_path / "venv" / "bin" / "hermes-acp"
    console_script.parent.mkdir(parents=True)
    marker = "#!/usr/bin/env python\n# real console script\n"
    console_script.write_text(marker, encoding="utf-8")
    (fake_home / "hermes-acp").symlink_to(console_script)

    _ensure_acp_launcher()

    assert console_script.read_text(encoding="utf-8") == marker
    assert (fake_home / "hermes-acp").is_symlink()






def test_unwritable_bin_dir_is_skipped(fake_home):
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    fake_home.chmod(0o555)
    try:
        _ensure_acp_launcher()  # must not raise
        assert not (fake_home / "hermes-acp").exists()
    finally:
        fake_home.chmod(0o755)


def test_update_self_heals_hermes_agent_launcher(fake_home):
    """Existing installs should gain the declared `hermes-agent` command."""
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")

    _ensure_hermes_agent_launcher()

    launcher = fake_home / "hermes-agent"
    assert launcher.is_file()
    assert launcher.stat().st_mode & stat.S_IXUSR, "launcher must be executable"
    text = launcher.read_text(encoding="utf-8")
    assert "unset PYTHONPATH" in text
    assert "unset PYTHONHOME" in text
    assert "run_agent.py" in text


def test_hermes_agent_repair_does_not_follow_symlink_into_venv(fake_home, tmp_path):
    """Never overwrite a symlinked venv/bin/hermes-agent console script."""
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    console_script = tmp_path / "venv" / "bin" / "hermes-agent"
    console_script.parent.mkdir(parents=True)
    marker = "#!/usr/bin/env python\n# real hermes-agent console script\n"
    console_script.write_text(marker, encoding="utf-8")
    (fake_home / "hermes-agent").symlink_to(console_script)

    _ensure_hermes_agent_launcher()

    assert console_script.read_text(encoding="utf-8") == marker
    assert (fake_home / "hermes-agent").is_symlink()
