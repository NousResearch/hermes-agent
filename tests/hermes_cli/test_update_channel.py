"""Install-scoped update channel record — contract tests (root decision 1).

The channel is an installation property stored at
``get_default_hermes_root()/update-channel.json``: schema
``{"schema_version": 1, "channel": "stable" | "beta"}``, resolved at call
time, shared by every profile using that install. A missing or invalid record
means stable and is NEVER written by a read. Only explicit selection persists.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import update_channel

pytestmark = pytest.mark.stable_channel_default


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    """Isolated install root; the module must resolve the record path per call."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    monkeypatch.setattr(update_channel, "get_default_hermes_root", lambda: root)
    return root


class TestReadContract:
    def test_missing_record_reads_stable_and_never_writes(self, hermes_root):
        assert update_channel.read_update_channel(hermes_root) == "stable"
        assert update_channel.read_channel_record(hermes_root) is None
        # The read path must not materialize the file (no startup write).
        assert not (hermes_root / "update-channel.json").exists()

    def test_invalid_record_reads_as_stable(self, hermes_root):
        path = hermes_root / "update-channel.json"
        path.write_text(json.dumps({"schema_version": 1, "channel": "rc"}), encoding="utf-8")
        assert update_channel.read_update_channel(hermes_root) == "stable"
        # The invalid record is left untouched — reads never repair/write.
        assert json.loads(path.read_text(encoding="utf-8"))["channel"] == "rc"

    def test_malformed_json_reads_as_stable(self, hermes_root):
        (hermes_root / "update-channel.json").write_text("{not json", encoding="utf-8")
        assert update_channel.read_update_channel(hermes_root) == "stable"

    def test_beta_round_trips(self, hermes_root):
        update_channel.write_channel_record("beta", hermes_root)
        assert update_channel.read_update_channel(hermes_root) == "beta"
        record = update_channel.read_channel_record(hermes_root)
        assert record == {"schema_version": 1, "channel": "beta"}

    def test_stable_round_trips(self, hermes_root):
        update_channel.write_channel_record("stable", hermes_root)
        assert update_channel.read_update_channel(hermes_root) == "stable"


class TestWriteContract:
    def test_write_rejects_unknown_channel(self, hermes_root):
        with pytest.raises(ValueError):
            update_channel.write_channel_record("rc", hermes_root)
        assert not (hermes_root / "update-channel.json").exists()

    def test_write_aliases_main_to_beta(self, hermes_root):
        record = update_channel.write_channel_record("main", hermes_root)
        assert record["channel"] == "beta"
        assert update_channel.read_update_channel(hermes_root) == "beta"

    def test_write_aliases_release_to_stable(self, hermes_root):
        record = update_channel.write_channel_record("release", hermes_root)
        assert record["channel"] == "stable"

    def test_no_torn_temp_file_survives_write(self, hermes_root):
        update_channel.write_channel_record("beta", hermes_root)
        assert not (hermes_root / "update-channel.json.tmp").exists()


class TestSharedRootMultiProfile:
    """Two profiles on one install see ONE record (shared-root, A→B→A)."""

    def test_record_is_shared_across_profile_homes(self, tmp_path, monkeypatch):
        install_root = tmp_path / "hermes-root"
        install_root.mkdir()
        profile_a = install_root / "profiles" / "a"
        profile_b = install_root / "profiles" / "b"
        profile_a.mkdir(parents=True)
        profile_b.mkdir(parents=True)

        seen = []

        def serve(profile_home):
            monkeypatch.setattr(update_channel, "get_default_hermes_root", lambda: install_root)
            monkeypatch.setenv("HERMES_HOME", str(profile_home))
            seen.append(update_channel.read_update_channel())

        serve(profile_a)  # A: no record -> stable
        update_channel.write_channel_record("beta", install_root)  # B: explicit beta
        serve(profile_b)
        serve(profile_a)  # A again: sees B's write — shared root, not per-profile
        assert seen == ["stable", "beta", "beta"]

    def test_independent_roots_do_not_leak(self, tmp_path, monkeypatch):
        root_one = tmp_path / "one"
        root_two = tmp_path / "two"
        root_one.mkdir()
        root_two.mkdir()

        monkeypatch.setattr(update_channel, "get_default_hermes_root", lambda: root_one)
        update_channel.write_channel_record("beta", root_one)

        monkeypatch.setattr(update_channel, "get_default_hermes_root", lambda: root_two)
        assert update_channel.read_update_channel() == "stable"
        assert not (root_two / "update-channel.json").exists()


class TestLegacyConfigMigration:
    """An explicit legacy config channel counts as consent until a record exists."""

    def test_explicit_legacy_main_reads_beta(self, hermes_root, monkeypatch):

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"updates": {"channel": "main"}},
        )
        assert update_channel.read_update_channel(hermes_root) == "beta"
        # Still no record written: migration is read-only.
        assert not (hermes_root / "update-channel.json").exists()

    def test_record_wins_over_legacy_config(self, hermes_root, monkeypatch):

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"updates": {"channel": "main"}},
        )
        update_channel.write_channel_record("stable", hermes_root)
        assert update_channel.read_update_channel(hermes_root) == "stable"

    def test_no_explicit_choice_is_not_migration(self, hermes_root, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"updates": {}},
        )
        assert update_channel.read_update_channel(hermes_root) == "stable"


class TestConfiguredReleaseRequest:
    """The CLI's target resolution honors the record (via update_cmd_release)."""

    def test_missing_record_means_stable_release_target(self, hermes_root, monkeypatch):

        from hermes_cli.update_cmd_release import RELEASE_LATEST, _configured_release_request

        monkeypatch.setattr(
            "hermes_cli.update_channel.get_default_hermes_root", lambda: hermes_root)
        assert _configured_release_request(SimpleNamespace()) == RELEASE_LATEST

    def test_beta_record_means_branch_target(self, hermes_root, monkeypatch):

        from hermes_cli.update_cmd_release import _configured_release_request

        monkeypatch.setattr(
            "hermes_cli.update_channel.get_default_hermes_root", lambda: hermes_root)
        update_channel.write_channel_record("beta", hermes_root)
        assert _configured_release_request(SimpleNamespace()) is None

    def test_explicit_branch_wins_once_over_record(self, hermes_root, monkeypatch):

        from hermes_cli.update_cmd_release import _configured_release_request

        monkeypatch.setattr(
            "hermes_cli.update_channel.get_default_hermes_root", lambda: hermes_root)
        assert _configured_release_request(SimpleNamespace(branch="dev")) is None
