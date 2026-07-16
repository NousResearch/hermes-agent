"""Tests for ``hermes adopt`` (hop 3, Python-side adoption command).

See ``docs/plans/updater-rework/03-phase2-compat-and-adoption.md`` task 2.5.

These tests exercise behavior via the function call — they do NOT read
source code (AGENTS.md §"Never read source code in tests").
"""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.subcommands.adopt import cmd_adopt
from hermes_cli.subcommands import adopt as adopt_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> SimpleNamespace:
    """Build a minimal args namespace for cmd_adopt."""
    defaults = {
        "source": None,
        "yes": False,
        "yes_dirty": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_download(dest, *, source_base=None):
    """Stand-in for _download_updater that writes a dummy binary."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"dummy")
    dest.chmod(0o755)
    return dest


class _ExecvIntercept(Exception):
    """Raised by the fake os.execv to simulate process replacement.

    ``os.execv`` never returns in real life — it replaces the process image.
    Tests monkeypatch it to capture the call, but must also prevent fallthrough
    past the execv site.  Raising this sentinel achieves both.
    """


def _patch_execv(monkeypatch, execv_calls):
    """Monkeypatch os.execv to record calls and raise _ExecvIntercept."""
    def fake_execv(path, argv):
        execv_calls.append((path, argv))
        raise _ExecvIntercept()
    monkeypatch.setattr(os, "execv", fake_execv)


# ---------------------------------------------------------------------------
# (a) Refuses for docker / nix / brew / pip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,expected_cmd_fragment", [
    ("docker", "docker pull"),
    ("nixos", "nixos-rebuild"),
    ("homebrew", "brew upgrade"),
    ("pip", "pip install --upgrade"),
])
def test_cmd_adopt_refuses_blocked_methods(method, expected_cmd_fragment, capsys):
    """``cmd_adopt`` exits(1) for docker/nixos/homebrew/pip with recommended cmd."""
    with patch("hermes_cli.config.detect_install_method", return_value=method):
        with pytest.raises(SystemExit) as excinfo:
            cmd_adopt(_make_args())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert expected_cmd_fragment in err
    assert method in err.lower() or method.replace("os", "") in err.lower()


# ---------------------------------------------------------------------------
# (b) Requires --yes-dirty for dirty trees
# ---------------------------------------------------------------------------

def test_cmd_adopt_requires_yes_dirty_for_dirty_tree(capsys):
    """A dirty (non-pristine) legacy install requires --yes-dirty to proceed."""
    fake_legacy = SimpleNamespace(pristine=False, reasons=["dirty working tree"])

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=fake_legacy,
         ):
        with pytest.raises(SystemExit) as excinfo:
            cmd_adopt(_make_args())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "--yes-dirty" in err


def test_cmd_adopt_proceeds_dirty_tree_with_yes_dirty(monkeypatch, tmp_path):
    """With --yes-dirty, adoption proceeds past the dirty guard to download."""
    fake_legacy = SimpleNamespace(pristine=False, reasons=["dirty working tree"])

    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=fake_legacy,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args(yes_dirty=True))

    assert len(execv_calls) == 1
    _path, argv = execv_calls[0]
    assert "adopt" in argv
    assert "--from-checkout" in argv
    assert str(tmp_path) in argv


# ---------------------------------------------------------------------------
# (c) Downloads the updater
# ---------------------------------------------------------------------------

def test_cmd_adopt_downloads_updater(monkeypatch, tmp_path):
    """cmd_adopt downloads the updater to $HERMES_HOME/bin/."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)

    # Monkeypatch _platform_suffix to get a known name.
    monkeypatch.setattr(adopt_mod, "_platform_suffix", lambda: "test-platform")

    # Create a fake release server directory with the updater binary.
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    fake_binary = release_dir / "hermes-updater-test-platform"
    fake_binary.write_bytes(b"fake updater binary content")

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=None,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args(source=f"file://{release_dir}"))

    # The updater should have been downloaded to $HERMES_HOME/bin/
    expected_name = "hermes-updater.exe" if os.name == "nt" else "hermes-updater"
    downloaded = bin_dir / expected_name
    assert downloaded.exists()
    assert downloaded.read_bytes() == b"fake updater binary content"


# ---------------------------------------------------------------------------
# (d) Calls os.execv with the right args
# ---------------------------------------------------------------------------

def test_cmd_adopt_calls_execv(monkeypatch, tmp_path):
    """cmd_adopt calls os.execv with ['hermes-updater', 'adopt', '--from-checkout', PROJECT_ROOT]."""
    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=None,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args())

    assert len(execv_calls) == 1
    updater_path, argv = execv_calls[0]

    # The path should point to $HERMES_HOME/bin/hermes-updater
    assert "bin" in str(updater_path)
    assert "hermes-updater" in str(updater_path)

    # The argv must start with "hermes-updater" and contain the adopt subcommand.
    assert argv[0] == "hermes-updater"
    assert "adopt" in argv
    assert "--from-checkout" in argv
    # PROJECT_ROOT should be in the argv
    assert str(tmp_path) in argv


def test_cmd_adopt_never_returns(monkeypatch, tmp_path):
    """If os.execv somehow returns, cmd_adopt raises RuntimeError (never falls through)."""
    # Make execv return None instead of replacing the process.
    monkeypatch.setattr(os, "execv", lambda path, argv: None)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=None,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(RuntimeError, match="os.execv returned"):
            cmd_adopt(_make_args())


# ---------------------------------------------------------------------------
# (e) --source is forwarded to the updater argv
# ---------------------------------------------------------------------------

def test_cmd_adopt_forwards_source_flag(monkeypatch, tmp_path):
    """The --source URL is forwarded to the updater's --source flag."""
    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    test_source = "file:///tmp/fake-release"

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=None,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args(source=test_source))

    assert len(execv_calls) == 1
    _path, argv = execv_calls[0]

    # --source <url> should appear in the updater argv
    assert "--source" in argv
    source_idx = argv.index("--source")
    assert source_idx + 1 < len(argv)
    assert argv[source_idx + 1] == test_source


# ---------------------------------------------------------------------------
# (f) --yes and --yes-dirty accepted
# ---------------------------------------------------------------------------

def test_cmd_adopt_accepts_yes_flag(monkeypatch, tmp_path):
    """--yes is accepted and doesn't prevent adoption."""
    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=None,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args(yes=True))

    assert len(execv_calls) == 1


# ---------------------------------------------------------------------------
# Pristine legacy install — proceeds without --yes-dirty
# ---------------------------------------------------------------------------

def test_cmd_adopt_pristine_git_proceeds_without_yes_dirty(monkeypatch, tmp_path):
    """A pristine git checkout proceeds to adoption without needing --yes-dirty."""
    fake_legacy = SimpleNamespace(pristine=True, reasons=[])

    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             return_value=fake_legacy,
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args())  # no --yes-dirty

    assert len(execv_calls) == 1


# ---------------------------------------------------------------------------
# Detector crash-proof: proceeds even if detect_legacy_install raises
# ---------------------------------------------------------------------------

def test_cmd_adopt_crash_proof_detector(monkeypatch, tmp_path):
    """If detect_legacy_install raises, adoption still proceeds (crash-proof)."""
    execv_calls = []
    _patch_execv(monkeypatch, execv_calls)
    monkeypatch.setattr(adopt_mod, "_download_updater", _fake_download)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch(
             "hermes_cli.adoption.detect_legacy_install",
             side_effect=RuntimeError("detector boom"),
         ), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("hermes_cli.main.PROJECT_ROOT", tmp_path):
        with pytest.raises(_ExecvIntercept):
            cmd_adopt(_make_args())  # should not raise from the detector

    assert len(execv_calls) == 1
