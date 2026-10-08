"""Real-file contracts for installation-wide channels and legacy consolidation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_installation as installation
from hermes_cli.update_channel import install_id, set_install_channel


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    (home / "config.yaml").write_text("# root comment\nmodel: root\n")
    (work / "config.yaml").write_text("# profile comment\nmodel: work\n")
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root, home, work


def _legacy(root, home, channel):
    config = {"model": home.name, "update": {"installs": {
        install_id(root): {"path": str(root), "channel": channel}}}}
    (home / "config.yaml").write_text(json.dumps(config))


@pytest.mark.parametrize("selected", ["root", "named"])
def test_channel_write_is_installation_owned_and_leaves_profile_config_alone(homes, monkeypatch, selected):
    root, home, work = homes
    monkeypatch.setenv("HERMES_HOME", str(home if selected == "root" else work))
    before = (work / "config.yaml").read_bytes()
    set_install_channel("preview", root)
    assert installation.resolve_install_channel(root) == "preview"
    assert installation.resolve_install_channel(root, home=home) == "preview"
    assert installation.resolve_install_channel(root, home=work) == "preview"
    assert (work / "config.yaml").read_bytes() == before
    assert "# root comment" in (home / "config.yaml").read_text()
    assert installation.read_install_channel_record(root)["scope"] == "installation"


def test_consistent_legacy_channel_is_read_only_until_migration(homes):
    root, home, work = homes
    _legacy(root, work, "stable")
    before = {p: p.read_bytes() for p in (home / "config.yaml", work / "config.yaml")}
    assert installation.resolve_install_channel(root, home=home) == "stable"
    assert installation.resolve_install_channel(root, home=work) == "stable"
    assert all(path.read_bytes() == data for path, data in before.items())
    receipt = installation.migrate_install_channel(root)
    assert receipt.changed
    assert installation.read_install_channel_record(root)["scope"] == "installation"
    assert (work / "config.yaml").read_bytes() == before[work / "config.yaml"]
    assert receipt.rollback()["ok"]
    assert installation.resolve_install_channel(root) == "stable"
    assert "scope" not in installation.read_install_channel_record(root)
    assert "# root comment" in (home / "config.yaml").read_text()


def test_matching_legacy_records_consolidate_without_selecting_a_profile(homes):
    root, home, work = homes
    _legacy(root, home, "canary")
    _legacy(root, work, "canary")
    installation.migrate_install_channel(root)
    assert installation.resolve_install_channel(root, home=work) == "canary"


def test_conflicting_legacy_channels_name_both_sources_and_do_not_write(homes):
    root, home, work = homes
    _legacy(root, home, "main")
    _legacy(root, work, "stable")
    before = {p: p.read_bytes() for p in (home / "config.yaml", work / "config.yaml")}
    for reader in (installation.resolve_install_channel, installation.migrate_install_channel):
        with pytest.raises(ValueError, match="Conflicting update channels") as raised:
            reader(root)
        assert str(home / "config.yaml") in str(raised.value)
        assert str(work / "config.yaml") in str(raised.value)
        assert "--set-channel" in str(raised.value)
    assert all(path.read_bytes() == data for path, data in before.items())
    set_install_channel("preview", root)
    assert installation.resolve_install_channel(root, home=work) == "preview"
    assert (work / "config.yaml").read_bytes() == before[work / "config.yaml"]


def test_rollback_does_not_overwrite_a_later_channel_choice(homes):
    root, _, work = homes
    _legacy(root, work, "stable")
    migration = installation.migrate_install_channel(root)
    set_install_channel("canary", root)
    assert migration.rollback()["ok"] is False
    assert installation.resolve_install_channel(root) == "canary"


def test_rollback_preserves_unrelated_concurrent_configuration(homes):
    from hermes_cli.config import atomic_config_write, require_readable_config_before_write

    root, home, _ = homes
    migration = installation.migrate_install_channel(root)
    config = require_readable_config_before_write(home / "config.yaml")
    config["model"] = "new-model"
    atomic_config_write(home / "config.yaml", config)
    assert migration.rollback()["ok"]
    assert require_readable_config_before_write(home / "config.yaml")["model"] == "new-model"


def test_separate_checkouts_have_independent_channels_in_one_home(homes):
    root, home, work = homes
    other = root.parent / "other-checkout"
    other.mkdir()
    set_install_channel("stable", root)
    set_install_channel("canary", other)
    assert installation.resolve_install_channel(root, home=work) == "stable"
    assert installation.resolve_install_channel(other, home=home) == "canary"


def test_borrowed_home_uses_the_committed_installation_owner(homes, monkeypatch):
    from hermes_cli.update_installation_owner import installation_home

    root, home, _ = homes
    root = home / "hermes-agent"
    root.mkdir()
    state = home / "installs" / install_id(root)
    state.mkdir(parents=True)
    (state / "facts.json").write_text("{}")
    set_install_channel("stable", root)
    borrower = home.parent / "scratch-home"
    monkeypatch.setenv("HERMES_HOME", str(borrower))
    assert installation_home(root) == home
    assert installation.resolve_install_channel(root) == "stable"
    set_install_channel("canary", root)
    assert not (borrower / "config.yaml").exists()


def test_deleted_and_markerless_profiles_do_not_choose_an_update_channel(homes):
    from hermes_constants import profile_tombstone_path

    root, home, work = homes
    _legacy(root, work, "canary")
    tombstone = profile_tombstone_path(work)
    tombstone.parent.mkdir(parents=True)
    tombstone.write_text("deleted")
    (home / "profiles" / "stray").mkdir()
    assert installation.resolve_install_channel(root) == "main"


@pytest.mark.parametrize("path_kind", ["root", "profile"])
def test_unreadable_legacy_configuration_does_not_silently_choose_main(homes, path_kind):
    root, home, work = homes
    path = (home if path_kind == "root" else work) / "config.yaml"
    path.write_text("update: [broken")
    with pytest.raises(RuntimeError, match="formatting error"):
        installation.resolve_install_channel(root)


def test_manual_check_and_auto_share_the_installation_channel(homes, monkeypatch):
    from hermes_cli import source_check, source_releases, update_auto_run, update_cmd
    from hermes_cli.update_auto_state import AutoUpdateContext

    root, home, work = homes
    _legacy(root, work, "preview")
    monkeypatch.setenv("HERMES_HOME", str(work))
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    assert update_cmd._source_update_channel(SimpleNamespace(branch=None, channel=None)) == "preview"
    monkeypatch.setattr(update_auto_run, "require_source_install", lambda c: None)
    monkeypatch.setattr(source_releases, "resolve_source_target", lambda *a: SimpleNamespace(branch="release-branch"))
    calls = []
    monkeypatch.setattr(source_check, "check_for_updates", lambda **kw: calls.append(kw) or {
        "supported": True, "updateAvailable": False, "branch": kw["branch"]})
    context = AutoUpdateContext(root, work, home / "logs" / "update_receipts")
    update_auto_run.check_update(context, SimpleNamespace(branch=None, channel=None))
    assert calls[0]["channel"] == "preview"
    assert calls[0]["home"] == home


def test_retired_channel_adoption_updates_installation_root_even_from_named_profile(homes, monkeypatch):
    from hermes_cli.update_channel import adopt_retired_channel

    root, home, work = homes
    set_install_channel("retired", root)
    original = installation.read_install_channel_record(root)
    monkeypatch.setenv("HERMES_HOME", str(work))
    before = (work / "config.yaml").read_bytes()
    assert adopt_retired_channel({"source": str(root), "home": str(work),
                                  "channel_retirement": {"original": original, "destination": "stable"}})
    assert installation.resolve_install_channel(root) == "stable"
    assert (work / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("layout", ["absent", "null-update", "null-installs", "null-record"])
def test_nullable_legacy_config_roundtrips_through_migration_and_rollback(homes, layout):
    from hermes_cli.config import require_readable_config_before_write

    root, home, _ = homes
    key = install_id(root)
    original = {"absent": {}, "null-update": {"update": None},
                "null-installs": {"update": {"installs": None}},
                "null-record": {"update": {"installs": {key: None}}}}[layout]
    (home / "config.yaml").write_text(json.dumps(original))
    assert installation.resolve_install_channel(root) == "main"
    migrated = installation.migrate_install_channel(root)
    assert installation.resolve_install_channel(root) == "main"
    assert migrated.rollback()["ok"]
    assert require_readable_config_before_write(home / "config.yaml") == original


@pytest.mark.parametrize("scope", [[], {}, 2, "profile"])
def test_unknown_scope_cannot_be_treated_as_a_legacy_choice(homes, scope):
    root, home, _ = homes
    record = {"channel": "stable", "scope": scope}
    (home / "config.yaml").write_text(json.dumps({"update": {"installs": {install_id(root): record}}}))
    with pytest.raises(ValueError, match="Unrecognized installation update scope"):
        installation.resolve_install_channel(root)


def test_canonical_channel_no_longer_depends_on_unrelated_profile_config(homes):
    root, _, work = homes
    set_install_channel("stable", root)
    (work / "config.yaml").write_text("update: [broken")
    assert installation.resolve_install_channel(root, home=work) == "stable"
    assert installation.migrate_install_channel(root, home=work).changed is False


@pytest.mark.platforms("posix")
def test_installation_aliases_share_channel_and_scheduler_identity(homes):
    from hermes_cli.update_auto_state import AutoUpdateContext

    root, home, work = homes
    alias = root.parent / "checkout-link"
    alias.symlink_to(root, target_is_directory=True)
    set_install_channel("stable", alias)
    direct = AutoUpdateContext(root, home, home / "logs" / "update_receipts")
    linked = AutoUpdateContext(alias, work, home / "logs" / "update_receipts")
    assert direct == linked
    assert installation.read_install_channel_record(alias)["path"] == str(root)
    assert installation.resolve_install_channel(root) == "stable"


@pytest.mark.platforms("posix")
def test_known_pm_owner_wins_when_active_profile_is_an_external_symlink(homes, monkeypatch):
    from pm.environments import installed_home_root, owning_home_root

    _, home, _ = homes
    root = home / "hermes-agent"
    root.mkdir()
    state = home / "installs" / install_id(root)
    state.mkdir(parents=True)
    (state / "facts.json").write_text("{}")
    set_install_channel("stable", root)
    external = home.parent / "external-profile"
    external.mkdir()
    _legacy(root, external, "canary")
    named = home / "profiles" / "linked"
    named.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(named))
    assert owning_home_root(root) is None
    assert installed_home_root(root) == home
    assert installation.resolve_install_channel(root) == "stable"
    assert installation.resolve_install_channel(root, home=named.resolve()) == "stable"
