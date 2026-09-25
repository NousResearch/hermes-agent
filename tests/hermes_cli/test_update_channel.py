"""Per-install update-channel records (hermes_cli/update_channel.py).

Channel is config, keyed by the install id (sha16 of the canonical
install-root path — inline helper, to be deduped with
boot_bootstrap._install_key at assembly), never home-global. Mechanism
comes from the stamp; external installs have no channel at all.
"""
import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli.update_channel import (
    CHANNEL_MAIN,
    CHANNEL_CANARY,
    CHANNEL_STABLE,
    default_channel,
    install_id,
    resolve_update_channel,
    set_install_channel,
    stale_channel_records,
)


def _stamp(root: Path, mechanism: str, tag: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stamp = {"schemaVersion": 2, "updateMechanism": mechanism}
    if tag is not None:
        stamp["tag"] = tag
    (root / "install-stamp.json").write_text(json.dumps(stamp))


def _config_for(root: Path, channel: str) -> dict:
    return {
        "update": {
            "installs": {
                install_id(root): {"path": str(root), "channel": channel}
            }
        }
    }


def test_dynamic_channel_parser_and_per_install_round_trip(tmp_path, monkeypatch):
    import argparse
    import hermes_yaml as yaml
    from hermes_cli.subcommands.update import build_update_parser

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=lambda args: None)
    root = tmp_path / "source"
    root.mkdir()
    name = "new-preview-493"
    args = parser.parse_args(["update", "--set-channel", name])
    set_install_channel(args.set_channel, root)
    saved = yaml.safe_load((tmp_path / "home/config.yaml").read_text())
    assert resolve_update_channel(saved, root) == name
    assert parser.parse_args(["update", "--channel", name]).channel == name
    assert resolve_update_channel(saved, tmp_path / "other") == "main"
    for invalid in (" New-preview", "PREVIEW", "preview/other", "preview--other", ""):
        with pytest.raises(ValueError):
            set_install_channel(invalid, root)
        with pytest.raises(ValueError):
            resolve_update_channel(_config_for(root, invalid), root)
        with pytest.raises(SystemExit):
            parser.parse_args(["update", "--channel", invalid])


class TestInstallId:
    def test_path_derived_and_stable(self, tmp_path):
        """The id hashes the canonical PATH — same path, same id, no matter
        what the tree contains (survives electron-updater artifact swaps)."""
        root = tmp_path / "install"
        root.mkdir()
        before = install_id(root)
        _stamp(root, "electron-updater")  # contents change...
        assert install_id(root) == before  # ...id does not

    def test_two_installs_two_ids(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert install_id(a) != install_id(b)

    def test_matches_the_state_folder_key_derivation(self, tmp_path):
        """sha256(canonical path)[:16] — the boot_bootstrap._install_key
        contract, recomputed independently so the inline helper cannot
        drift before the assembly-time dedupe."""
        root = tmp_path / "install"
        root.mkdir()
        expected = hashlib.sha256(
            str(root.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        assert install_id(root) == expected


class TestResolve:

    def test_channel_bundle_keeps_baked_subscription_in_its_cli(self, tmp_path):
        (tmp_path / 'install-stamp.json').write_text(json.dumps({
            'payload': 'bundled', 'source': 'channel-build',
            'updateMechanism': 'electron-updater', 'tag': None,
            'channelBuild': {'channel': 'r2-preview'},
        }), encoding='utf-8')
        assert default_channel(tmp_path) == 'r2-preview'
        assert resolve_update_channel(_config_for(tmp_path, 'stable'), tmp_path) == 'r2-preview'

    def test_multi_install_isolation(self, tmp_path):
        """Two installs, one config: each resolves its own record and a
        missing record falls to the mechanism default — never the sibling's."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        _stamp(a, "self")
        _stamp(b, "self")
        config = _config_for(a, "stable")
        assert resolve_update_channel(config, a) == CHANNEL_STABLE
        assert resolve_update_channel(config, b) == CHANNEL_MAIN

    @pytest.mark.parametrize('mechanism,tag,payload,record,expected', [
        ('self', None, None, None, 'main'),
        ('self', 'v1.2.3-canary.20260818', None, None, 'main'),
        (None, None, None, None, 'main'),
        ('self', None, None, 'canary', 'canary'),
        ('self', None, None, 'yolo', 'yolo'),
        ('electron-updater', 'v1.2.3', None, None, 'stable'),
        ('electron-updater', 'v1.2.3+canary.20260819T171926Z', None, 'stable', 'canary'),
        ('app-installer', 'v1.2.3+canary.20260819T171926Z', None, None, 'canary'),
        ('microsoft-store', 'v1.2.3+canary.20260819T171926Z', None, None, 'canary'),
        ('electron-updater', 'v1.2.3-canary.20260818', None, None, 'stable'),
        ('electron-updater', None, None, 'canary', 'stable'),
        ('external', 'v1.2.3', 'bundled', 'main', 'stable'),
        ('external', 'v1.2.3', 'light', 'main', 'stable'),
        ('external', 'v1.2.4+canary.20260911T125822Z', 'bundled', 'main', 'canary'),
        ('external', 'v1.2.4+canary.20260911T125822Z', 'light', 'main', 'canary'),
    ])
    def test_channel_policy(self, tmp_path, mechanism, tag, payload, record, expected):
        if mechanism is not None:
            stamp = {'updateMechanism': mechanism, 'tag': tag}
            if payload:
                stamp['payload'] = payload
            (tmp_path / 'install-stamp.json').write_text(json.dumps(stamp), encoding='utf-8')
        config = _config_for(tmp_path, record) if record else {}
        assert resolve_update_channel(config, tmp_path) == expected
        if record is None or mechanism != 'self':
            assert default_channel(tmp_path) == expected


class TestSetChannel:
    def _home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    @pytest.mark.parametrize("channel", ["stable", "canary", "main"])
    def test_set_resolve_round_trip(self, tmp_path, monkeypatch, channel):
        import hermes_yaml as yaml

        home = self._home(tmp_path, monkeypatch)
        root = tmp_path / "install"
        _stamp(root, "self")

        sha16 = set_install_channel(channel, root)
        assert sha16 == install_id(root)

        written = yaml.safe_load((home / "config.yaml").read_text())
        record = written["update"]["installs"][sha16]
        assert record["channel"] == channel
        assert record["path"] == str(root)
        assert resolve_update_channel(written, root) == channel



    @pytest.mark.parametrize("mechanism", ["external", "electron-updater", "app-installer", "microsoft-store"])
    def test_os_owned_mechanism_refuses_channel_writes(self, tmp_path, monkeypatch, mechanism):
        self._home(tmp_path, monkeypatch)
        root = tmp_path / "os-owned-tree"
        _stamp(root, mechanism)
        with pytest.raises(ValueError, match="owned by"):
            set_install_channel("stable", root)

    @pytest.mark.parametrize("channel", ["main", "stable", "canary"])
    def test_commit_build_channel_refusal_keeps_existing_config(self, tmp_path, monkeypatch, channel):
        home = self._home(tmp_path, monkeypatch)
        config = home / "config.yaml"
        config.write_text("# preserve\nupdate: {}\n")
        before = config.read_bytes()
        root = tmp_path / "commit-build"
        root.mkdir()
        (root / "install-stamp.json").write_text(json.dumps({
            "source": "commit-build", "updateMechanism": "external",
        }))
        message = "This build doesn't get updates. Ask the developer who gave it to you for a new build."
        with pytest.raises(ValueError) as error:
            set_install_channel(channel, root)
        assert str(error.value) == message
        assert config.read_bytes() == before

    def test_bad_channel_refuses(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        root = tmp_path / "install"
        _stamp(root, "self")
        with pytest.raises(ValueError, match="Invalid channel name"):
            set_install_channel("../beta", root)


class TestSetChannelCLI:
    """cmd_update --set-channel: the switch texts (design record)."""

    @pytest.fixture(autouse=True)
    def _fail_fast_updater_sentinel(self, monkeypatch, tmp_path):
        """Isolation harness (C03 revalidation).

        If the informational flags were ever unhandled in preflight,
        ``cmd_update`` would enter the real updater (lock, git, backups,
        process pause) — that must fail HERE, at a ``_cmd_update_impl``
        sentinel, instead of mutating anything. Subprocess and network
        are denied outright, and the install root is pinned to ``tmp_path``
        (a real root, never this checkout) so stamp reads cannot touch the
        worktree either.
        """
        import socket
        import subprocess

        def _sentinel(*_a, **_kw):
            raise AssertionError(
                "fail-fast sentinel: _cmd_update_impl entered — channel "
                "flags were not handled by the update preflight"
            )

        def _denied(what):
            def _deny(*_a, **_kw):
                raise AssertionError(
                    f"fail-fast guard: {what} attempted from a channel-flag test"
                )

            return _deny

        monkeypatch.setattr(
            "hermes_cli.update_cmd._cmd_update_impl", _sentinel
        )
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            monkeypatch.setattr(subprocess, name, _denied(f"subprocess.{name}"))
        monkeypatch.setattr(
            socket, "create_connection", _denied("socket.create_connection")
        )
        monkeypatch.setattr(socket.socket, "connect", _denied("socket.connect"))
        # Real root, real filesystem — but a temp one, never the checkout.
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))

    def _home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_malformed_update_section_refused_not_normalized(self, tmp_path, monkeypatch):
        """A scalar/malformed ``update`` (or ``update.installs``) is refused —
        the dotted writer would silently turn it into a mapping and destroy
        the user's value. The file is left byte-identical."""
        for bad in ("update: not-a-mapping\n", "update:\n  installs: 7\n"):
            home = self._home(tmp_path, monkeypatch)
            cfg = home / "config.yaml"
            cfg.write_text("# user header\n" + bad)
            root = tmp_path / "install"
            _stamp(root, "self")
            with pytest.raises(ValueError, match="not a mapping"):
                set_install_channel("stable", root)
            assert cfg.read_text() == "# user header\n" + bad
            # cleanup between loop iterations
            import shutil

            shutil.rmtree(home)

    def test_metadata_commands_precede_managed_refusal_and_write_once(
        self, tmp_path, monkeypatch, capsys
    ):
        import hermes_yaml as yaml
        import utils
        from hermes_cli import config, main

        home = self._home(tmp_path, monkeypatch)
        monkeypatch.setenv("HERMES_MANAGED", "nix")
        assert config.is_managed()
        root, other = tmp_path, tmp_path / "sibling"
        _stamp(root, "self")
        initial = {"model": {"provider": "fixture"}, **_config_for(other, "stable")}
        cfg = home / "config.yaml"
        cfg.write_text("# retain user comment\n" + yaml.safe_dump(initial), encoding="utf-8")
        before = cfg.read_bytes()

        with pytest.raises(SystemExit) as exc:
            main.cmd_update(self._args(install_id=True))
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == install_id(root)
        assert cfg.read_bytes() == before

        writes = []
        original = utils.atomic_roundtrip_yaml_update

        def write(path, key, value):
            writes.append((path, key))
            return original(path, key, value)

        monkeypatch.setattr(utils, "atomic_roundtrip_yaml_update", write)
        with pytest.raises(SystemExit) as exc:
            main.cmd_update(self._args(set_channel="canary"))
        assert exc.value.code == 0
        assert writes == [(cfg, f"update.installs.{install_id(root)}")]
        expected = initial
        expected["update"]["installs"][install_id(root)] = {"path": str(root), "channel": "canary"}
        assert yaml.safe_load(cfg.read_text(encoding="utf-8")) == expected
        assert cfg.read_text(encoding="utf-8").startswith("# retain user comment\n")
        output = capsys.readouterr().out
        assert f"Update channel for {install_id(root)}: canary" in output
        assert "forward-incompatible" in output
        assert not (home / "logs" / "update_receipts").exists()

    def test_metadata_refusals_leave_config_unchanged(self, tmp_path, monkeypatch, capsys):
        from hermes_cli import main

        home = self._home(tmp_path, monkeypatch)
        monkeypatch.setenv("HERMES_MANAGED", "nix")
        cfg = home / "config.yaml"
        before = b"# retain user comment\nmodel:\n  provider: fixture\n"
        cfg.write_bytes(before)
        for mechanism, channel, message in (
            ("self", "../bogus", "Invalid channel name"),
            ("external", "stable", "owned by"),
            ("app-installer", "canary", "owned by"),
        ):
            _stamp(tmp_path, mechanism)
            with pytest.raises(SystemExit) as exc:
                main.cmd_update(self._args(set_channel=channel))
            assert exc.value.code == 2
            assert message in capsys.readouterr().out
            assert cfg.read_bytes() == before
        assert not (home / "logs" / "update_receipts").exists()

    def _args(self, **kw):
        from types import SimpleNamespace

        base = dict(check=False, gateway=False, branch=None, channel=None,
                    set_channel=None, install_id=False, plan=False)
        base.update(kw)
        return SimpleNamespace(**base)

    @pytest.fixture(autouse=True)
    def _no_real_updater(self, monkeypatch, tmp_path):
        """Metadata commands must not enter the code-update pipeline."""
        from hermes_cli import main, update_cmd

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        def unexpected_update(*args, **kwargs):
            pytest.fail("metadata command entered the real updater")
        monkeypatch.setattr(update_cmd, "_cmd_update_impl", unexpected_update)

    def test_stable_switch_warns_about_state_without_requiring_a_newer_release(self, tmp_path, monkeypatch, capsys):
        from hermes_cli.main import cmd_update

        self._home(tmp_path, monkeypatch)
        _stamp(tmp_path, "self", "v0.28.0+canary.20260818T000000Z")
        with pytest.raises(SystemExit) as exc:
            cmd_update(self._args(set_channel="stable"))
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Back up your data" in out
        assert "older stable release" in out
        assert "Wait" not in out



@pytest.mark.parametrize('kind,expected', [('missing', 'missing'), ('replaced', 'replaced'),
                                             ('unclaimed', 'unclaimed'), ('healthy', None)])
def test_doctor_channel_records_are_read_only(tmp_path, monkeypatch, kind, expected):
    from copy import deepcopy
    home, root = tmp_path / '.hermes', tmp_path / 'install'
    monkeypatch.setenv('HERMES_HOME', str(home))
    if kind != 'missing':
        root.mkdir()
    key = '0' * 16 if kind == 'replaced' else install_id(root)
    config = {'update': {'installs': {key: {'path': str(root), 'channel': 'main'}}}}
    if kind == 'healthy':
        state = home / 'installs' / key
        state.mkdir(parents=True)
        (state / 'install.json').write_text(json.dumps({'root': str(root)}), encoding='utf-8')
    before = deepcopy(config)
    assert [(key, reason) for key, _, reason in stale_channel_records(config)] == (
        [(key, expected)] if expected else [])
    assert config == before
