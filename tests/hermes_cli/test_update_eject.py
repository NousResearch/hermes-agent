"""Tests for ``hermes update --eject`` (hermes_cli/update_cmd.py::cmd_update_eject).

An embedded desktop install runs the agent out of the sealed app bundle:
a gitless tree whose build stamp says ``distribution: desktop-app``. The
eject is a full handoff to Hermes Setup. The tests fake only the two hard
process boundaries — the download and the installer launch — and run
everything else for real.
"""

import json

import pytest

import hermes_cli.update_cmd as update_cmd
from hermes_cli.update_cmd import cmd_update_eject

COMMIT = "ab" * 20


def _write_build_info(root, **overrides):
    info = {"commit": COMMIT, "tag": "v0.1.0", "distribution": "desktop-app"}
    info.update(overrides)
    (root / ".hermes_build_info.json").write_text(json.dumps(info))


@pytest.fixture
def bundle_repo(tmp_path, monkeypatch):
    """The payload repo of an embedded bundle: build info, no .git."""
    repo = tmp_path / "bundle" / "repo"
    repo.mkdir(parents=True)
    _write_build_info(repo)
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    return repo


class _Args:
    def __init__(self, channel=None):
        self.eject = True
        self.channel = channel


@pytest.fixture
def fake_setup(monkeypatch):
    """Fake the download + launch boundary; record what eject asked for."""
    calls = {}

    def fake_download(url, dest):
        calls["url"] = url
        dest.write_bytes(b"fake-installer")
        return True

    def fake_launch(setup_path, scratch, commit):
        calls["setup_path"] = setup_path
        calls["commit"] = commit
        return True

    monkeypatch.setattr(update_cmd, "_download_hermes_setup", fake_download)
    monkeypatch.setattr(update_cmd, "_launch_hermes_setup", fake_launch)
    monkeypatch.setattr(update_cmd.sys, "platform", "darwin")
    return calls


class TestEjectEmbedded:
    def test_eject_downloads_setup_and_pins_the_bundle_commit(
        self, bundle_repo, fake_setup, capsys
    ):
        rc = cmd_update_eject(_Args())
        out = capsys.readouterr().out

        assert rc == 0
        # The pin is the bundle's own commit — never the tag, never HEAD.
        assert fake_setup["commit"] == COMMIT
        assert "Hermes-Setup.dmg" in fake_setup["url"]
        assert "hermes-assets.nousresearch.com" in fake_setup["url"]
        # The handoff instructs the user to close the app: Setup replaces it.
        assert "full handoff" in out
        assert "Close the Hermes desktop app" in out

    def test_eject_windows_uses_the_exe(self, bundle_repo, fake_setup, monkeypatch):
        monkeypatch.setattr(update_cmd.sys, "platform", "win32")
        assert cmd_update_eject(_Args()) == 0
        assert fake_setup["url"].endswith("Hermes-Setup.exe")

    def test_eject_refuses_unsupported_platforms(self, bundle_repo, monkeypatch, capsys):
        monkeypatch.setattr(update_cmd.sys, "platform", "linux")
        assert cmd_update_eject(_Args()) == 1
        assert "install.sh" in capsys.readouterr().out

    def test_eject_refuses_without_a_valid_commit(self, bundle_repo, fake_setup, capsys):
        _write_build_info(bundle_repo, commit="not-a-sha")
        assert cmd_update_eject(_Args()) == 1
        assert "commit" in capsys.readouterr().out
        assert "commit" not in fake_setup  # never launched

    def test_eject_skips_when_a_source_checkout_already_exists(
        self, bundle_repo, fake_setup, tmp_path, monkeypatch, capsys
    ):
        home = tmp_path / "hermes-home"
        target = home / "hermes-agent"
        (target / ".git").mkdir(parents=True)
        monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)

        assert cmd_update_eject(_Args()) == 0
        out = capsys.readouterr().out
        assert "already exists" in out
        assert "url" not in fake_setup  # no download

    def test_failed_download_aborts_cleanly(self, bundle_repo, fake_setup, monkeypatch, capsys):
        monkeypatch.setattr(
            update_cmd, "_download_hermes_setup", lambda url, dest: False
        )
        assert cmd_update_eject(_Args()) == 1
        assert "unchanged" in capsys.readouterr().out


class TestEjectOtherSealedTrees:
    def test_docker_tree_gets_the_docker_message_not_an_eject(
        self, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "docker-tree"
        repo.mkdir()
        _write_build_info(repo, distribution="docker")
        import hermes_cli.main as hermes_main

        monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

        assert cmd_update_eject(_Args()) == 1
        assert "docker pull" in capsys.readouterr().out


class TestEjectGitCheckout:
    def test_git_checkout_with_channel_switches_channel_only(
        self, tmp_path, monkeypatch, capsys
    ):
        repo = tmp_path / "src-checkout"
        (repo / ".git").mkdir(parents=True)
        import hermes_cli.main as hermes_main

        monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
        written = {}
        import hermes_cli.config as config_mod

        monkeypatch.setattr(
            config_mod, "set_config_value", lambda key, value, **kw: written.update({key: value})
        )

        assert cmd_update_eject(_Args(channel="stable")) == 0
        assert written == {"update.channel": "stable"}
        assert "git-managed" in capsys.readouterr().out

    def test_git_checkout_without_channel_is_a_noop(self, tmp_path, monkeypatch, capsys):
        repo = tmp_path / "src-checkout"
        (repo / ".git").mkdir(parents=True)
        import hermes_cli.main as hermes_main

        monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

        assert cmd_update_eject(_Args()) == 0
        assert "Nothing to eject" in capsys.readouterr().out
