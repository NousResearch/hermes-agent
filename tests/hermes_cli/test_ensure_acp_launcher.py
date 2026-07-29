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

from hermes_cli.main import _ensure_acp_launcher


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    return bin_dir


def test_writes_launcher_next_to_hermes(fake_home):
    hermes = fake_home / "hermes"
    hermes.write_text("#!/usr/bin/env bash\nexec true\n", encoding="utf-8")
    hermes.chmod(0o755)

    _ensure_acp_launcher()

    acp = fake_home / "hermes-acp"
    assert acp.is_file()
    assert acp.stat().st_mode & stat.S_IXUSR
    text = acp.read_text(encoding="utf-8")
    # Delegates to the sibling `hermes` launcher with the acp subcommand.
    assert f'exec "{hermes}" acp "$@"' in text


def test_noop_without_hermes_launcher(fake_home):
    _ensure_acp_launcher()
    assert not (fake_home / "hermes-acp").exists()


def test_does_not_overwrite_existing_command(fake_home):
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    existing = fake_home / "hermes-acp"
    marker = "#!/usr/bin/env python\n# real console script\n"
    existing.write_text(marker, encoding="utf-8")

    _ensure_acp_launcher()

    assert existing.read_text(encoding="utf-8") == marker


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


def test_skips_broken_symlink(fake_home, tmp_path):
    (fake_home / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    dangling = fake_home / "hermes-acp"
    dangling.symlink_to(tmp_path / "gone")

    _ensure_acp_launcher()

    assert dangling.is_symlink()
    assert not dangling.exists()


def test_symlinked_hermes_counts_as_present(fake_home, tmp_path):
    """FHS-style installs symlink ~/.local/bin/hermes — still eligible."""
    real = tmp_path / "real-hermes"
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    (fake_home / "hermes").symlink_to(real)

    _ensure_acp_launcher()

    assert (fake_home / "hermes-acp").is_file()


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
