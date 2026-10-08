"""Checkouts without PM ownership still have one durable update data-root owner."""

import json
from pathlib import Path

import pytest

from hermes_cli.update_auto_state import AutoUpdateContext
from hermes_cli.update_channel import install_id, set_install_channel
from hermes_cli.update_installation import resolve_install_channel
from hermes_cli.update_installation_owner import ensure_installation_home, installation_home, owner_path


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root, first, second = (tmp_path / name for name in ("checkout", "first-home", "second-home"))
    for path in (root, first, second):
        path.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(first))
    return root, first, second


def test_read_only_resolution_does_not_bind_an_unconfigured_checkout(layout):
    root, first, _ = layout
    assert installation_home(root) == first
    assert resolve_install_channel(root) == "main"
    assert not owner_path(root).exists()
    assert not (root / ".hermes").exists()


def test_explicit_configuration_binds_one_owner_for_unrelated_custom_homes(layout, monkeypatch):
    root, first, second = layout
    set_install_channel("stable", root)
    monkeypatch.setenv("HERMES_HOME", str(second))
    (second / "config.yaml").write_text("# unrelated configuration\nmodel: other\n")
    before = (second / "config.yaml").read_bytes()
    assert installation_home(root) == first
    assert resolve_install_channel(root) == "stable"
    set_install_channel("canary", root)
    assert resolve_install_channel(root, home=first) == "canary"
    assert (second / "config.yaml").read_bytes() == before
    a = AutoUpdateContext(root, first, first / "logs")
    b = AutoUpdateContext(root, second, second / "logs")
    assert a == b
    assert a.home == first
    assert a.status_path == b.status_path


def test_two_configuration_attempts_never_create_two_owners(layout):
    root, first, second = layout
    assert ensure_installation_home(root, home=first) == first
    before = owner_path(root).read_bytes()
    assert ensure_installation_home(root, home=second) == first
    assert owner_path(root).read_bytes() == before


def test_conflicting_pm_and_explicit_owner_evidence_is_not_prioritized(layout):
    root, first, second = layout
    root = second / "hermes-agent"
    root.mkdir()
    ensure_installation_home(root, home=first)
    state = second / "installs" / install_id(root)
    state.mkdir(parents=True)
    (state / "facts.json").write_text("{}")
    with pytest.raises(ValueError, match="Conflicting installation owners") as raised:
        installation_home(root)
    assert str(first) in str(raised.value) and str(second) in str(raised.value)


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"schema": 2}', '{"schema": 1, "dataRoot": "relative"}'])
def test_corrupt_owner_does_not_silently_adopt_the_current_home(layout, payload):
    root, _, second = layout
    path = owner_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(payload)
    with pytest.raises(ValueError):
        ensure_installation_home(root, home=second)
    assert path.read_text() == payload


@pytest.mark.platforms("posix")
def test_owner_record_symlink_is_never_followed_or_replaced(layout):
    root, first, _ = layout
    other = first / "other.json"
    other.write_text(json.dumps({"schema": 1, "installationRoot": str(root), "dataRoot": str(first)}))
    path = owner_path(root)
    path.parent.mkdir(parents=True)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        ensure_installation_home(root)
    assert path.is_symlink()


def test_missing_bound_data_root_does_not_create_a_replacement(layout):
    root, first, second = layout
    ensure_installation_home(root, home=first)
    first.rmdir()
    with pytest.raises(ValueError, match="unavailable"):
        ensure_installation_home(root, home=second)
    assert not first.exists()
