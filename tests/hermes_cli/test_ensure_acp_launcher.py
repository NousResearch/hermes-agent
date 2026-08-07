"""`hermes update` must self-heal the ``hermes-acp`` launcher.

ACP hosts (Zed, JetBrains, Buzz Desktop) resolve the agent by the
``hermes-acp`` command name on the login-shell PATH. Fresh installs get the
launcher from ``scripts/install.sh``; existing installs get it from
``_ensure_acp_launcher()`` during ``hermes update``.
"""

import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main import _ensure_acp_launcher

# All tests in this module exercise the POSIX launcher convention
# (.local/bin on $PATH via login shell). Windows installs the launcher
# via a different code path (scripts/install.ps1 + PATH registration),
# and os.geteuid() doesn't exist on Windows. Skip the whole module.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only: .local/bin launcher convention + os.geteuid; Windows uses a different install path",
)

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
