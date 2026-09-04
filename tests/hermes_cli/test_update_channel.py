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
    def test_per_install_record_wins(self, tmp_path):
        root = tmp_path / "install"
        _stamp(root, "self")
        config = _config_for(root, "stable")
        assert resolve_update_channel(config, root) == CHANNEL_STABLE

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

    def test_defaults_by_mechanism(self, tmp_path):
        source = tmp_path / "src"
        bundle = tmp_path / "bundle"
        _stamp(source, "self")
        _stamp(bundle, "electron-updater", tag="v0.27.0")
        assert default_channel(source) == CHANNEL_MAIN
        assert default_channel(bundle) == CHANNEL_STABLE
        assert resolve_update_channel({}, source) == CHANNEL_MAIN
        assert resolve_update_channel({}, bundle) == CHANNEL_STABLE

    def test_canary_artifact_defaults_to_its_own_feed(self, tmp_path):
        """A canary bundle with no per-install record tracks canary.

        The artifact publishes to canary.yml (product-identity.cjs keys the
        feed on this same tag). Defaulting it to stable made the updater ask
        for canary.yml under the newest STABLE release, which 404s and
        leaves a fresh canary install unable to update at all.
        """
        root = tmp_path / "canary-bundle"
        _stamp(root, "electron-updater", tag="v0.28.0-canary.20260819171926")
        assert default_channel(root) == CHANNEL_CANARY
        assert resolve_update_channel({}, root) == CHANNEL_CANARY

    def test_legacy_date_only_canary_tag_is_a_canary(self, tmp_path):
        root = tmp_path / "legacy-canary"
        _stamp(root, "electron-updater", tag="v0.28.0-canary.20260818")
        assert default_channel(root) == CHANNEL_CANARY

    def test_explicit_record_still_overrides_the_artifact_default(self, tmp_path):
        """The tag only supplies the DEFAULT: an opt-out must still work."""
        root = tmp_path / "canary-bundle"
        _stamp(root, "electron-updater", tag="v0.28.0-canary.20260819171926")
        assert resolve_update_channel(_config_for(root, "stable"), root) == CHANNEL_STABLE

    def test_a_canary_tag_on_a_source_install_is_not_a_canary_channel(self, tmp_path):
        """Only electron-updater bundles have release feeds to track."""
        root = tmp_path / "src"
        _stamp(root, "self", tag="v0.28.0-canary.20260819171926")
        assert default_channel(root) == CHANNEL_MAIN

    def test_stampless_tree_defaults_to_main(self, tmp_path):
        root = tmp_path / "bare"
        root.mkdir()
        assert resolve_update_channel({}, root) == CHANNEL_MAIN

    def test_canary_normalizes_to_main_for_source(self, tmp_path):
        root = tmp_path / "src"
        _stamp(root, "self")
        config = _config_for(root, "canary")
        assert resolve_update_channel(config, root) == CHANNEL_MAIN

    def test_canary_stays_for_electron_updater(self, tmp_path):
        root = tmp_path / "bundle"
        _stamp(root, "electron-updater")
        config = _config_for(root, "canary")
        assert resolve_update_channel(config, root) == CHANNEL_CANARY

    def test_garbage_record_falls_to_default(self, tmp_path):
        root = tmp_path / "src"
        _stamp(root, "self")
        config = _config_for(root, "yolo")
        assert resolve_update_channel(config, root) == CHANNEL_MAIN


class TestSetChannel:
    def _home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_set_resolve_round_trip(self, tmp_path, monkeypatch):
        import yaml

        home = self._home(tmp_path, monkeypatch)
        root = tmp_path / "install"
        _stamp(root, "self")

        sha16 = set_install_channel("stable", root)
        assert sha16 == install_id(root)

        written = yaml.safe_load((home / "config.yaml").read_text())
        record = written["update"]["installs"][sha16]
        assert record["channel"] == "stable"
        assert record["path"] == str(root)
        assert resolve_update_channel(written, root) == CHANNEL_STABLE

    def test_preserves_other_config_and_other_installs(self, tmp_path, monkeypatch):
        import yaml

        home = self._home(tmp_path, monkeypatch)
        other = tmp_path / "other"
        other.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"provider": "nous"},
                    "update": {
                        "installs": {
                            install_id(other): {"path": str(other), "channel": "canary"}
                        }
                    },
                }
            )
        )
        root = tmp_path / "install"
        _stamp(root, "self")
        set_install_channel("stable", root)

        written = yaml.safe_load((home / "config.yaml").read_text())
        assert written["model"] == {"provider": "nous"}
        assert written["update"]["installs"][install_id(other)]["channel"] == "canary"
        assert written["update"]["installs"][install_id(root)]["channel"] == "stable"

    def test_external_mechanism_refuses(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        root = tmp_path / "nix-tree"
        _stamp(root, "external")
        with pytest.raises(ValueError, match="owned by"):
            set_install_channel("stable", root)

    def test_bad_channel_refuses(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        root = tmp_path / "install"
        _stamp(root, "self")
        with pytest.raises(ValueError, match="unknown channel"):
            set_install_channel("beta", root)


class TestSetChannelCLI:
    """cmd_update --set-channel: the switch texts (design record)."""

    def _args(self, **kw):
        from types import SimpleNamespace

        base = dict(check=False, gateway=False, branch=None, channel=None,
                    set_channel=None, install_id=False, plan=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_stable_switch_from_canary_is_an_honest_wait(self, capsys):
        from unittest.mock import patch

        from hermes_cli.main import cmd_update

        with (
            patch("hermes_cli.config.is_managed", return_value=False),
            patch("hermes_cli.config.detect_install_method", return_value="unknown"),
            patch("hermes_cli.update_channel.set_install_channel", return_value="a" * 16),
            patch(
                "hermes_cli.steward.read_install_stamp",
                return_value={
                    "updateMechanism": "electron-updater",
                    "displayVersion": "0.28.0-canary.20260818",
                },
            ),
        ):
            with pytest.raises(SystemExit) as exc:
                cmd_update(self._args(set_channel="stable"))
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "0.28.0-canary.20260818" in out       # names where you are
        assert "v0.28.0" in out                       # names the wait target
        assert "hermes-agent.nousresearch.com" in out  # the impatient path

    def test_canary_optin_warns_about_forward_incompatible_state(self, capsys):
        from unittest.mock import patch

        from hermes_cli.main import cmd_update

        with (
            patch("hermes_cli.config.is_managed", return_value=False),
            patch("hermes_cli.config.detect_install_method", return_value="unknown"),
            patch("hermes_cli.update_channel.set_install_channel", return_value="a" * 16),
        ):
            with pytest.raises(SystemExit) as exc:
                cmd_update(self._args(set_channel="canary"))
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "forward-incompatible" in out

    def test_install_id_prints_and_exits(self, capsys):
        from unittest.mock import patch

        from hermes_cli.main import cmd_update

        with (
            patch("hermes_cli.config.is_managed", return_value=False),
            patch("hermes_cli.update_channel.install_id", return_value="b" * 16),
        ):
            with pytest.raises(SystemExit) as exc:
                cmd_update(self._args(install_id=True))
        assert exc.value.code == 0
        assert "b" * 16 in capsys.readouterr().out


class TestDoctorStaleness:
    def test_missing_path_flagged(self, tmp_path):
        gone = tmp_path / "gone"
        config = {
            "update": {"installs": {"deadbeefdeadbeef": {"path": str(gone), "channel": "main"}}}
        }
        stale = stale_channel_records(config)
        assert [(sha, reason) for sha, _r, reason in stale] == [
            ("deadbeefdeadbeef", "missing")
        ]

    def test_replaced_install_flagged(self, tmp_path):
        """The recorded path exists but keys to a different sha16 — the
        record is a leftover from a tree that used to live elsewhere."""
        root = tmp_path / "install"
        root.mkdir()
        config = {
            "update": {"installs": {"0" * 16: {"path": str(root), "channel": "main"}}}
        }
        stale = stale_channel_records(config)
        assert [(sha, reason) for sha, _r, reason in stale] == [("0" * 16, "replaced")]

    def test_unclaimed_record_flagged(self, tmp_path, monkeypatch):
        """sha16 matches the path but no installs/<sha16>/install.json exists."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        root = tmp_path / "install"
        root.mkdir()
        config = _config_for(root, "main")
        stale = stale_channel_records(config)
        assert [(sha, reason) for sha, _r, reason in stale] == [
            (install_id(root), "unclaimed")
        ]

    def test_healthy_record_not_flagged(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        root = tmp_path / "install"
        _stamp(root, "self")
        # The live install-state record the sha16 must be claimed by
        # (boot_bootstrap.ensure_install_dir writes this at boot; that
        # module is landing in a parallel lane, so write the marker
        # directly — the layout IS the contract).
        state = home / "installs" / install_id(root)
        state.mkdir(parents=True)
        (state / "install.json").write_text(json.dumps({"root": str(root)}))
        config = _config_for(root, "main")
        assert stale_channel_records(config) == []
