"""The launchd staleness check must see the Hermes-owned PATH prefix.

A pm-managed Node bump rewrites the store dir the generated plist pins, but
``_normalize_launchd_plist_for_comparison`` blanked the whole PATH payload before comparing, so an
installed plist that still pointed at the OLD Node tree kept passing the check: ``gateway status``
reported "Service definition matches the current Hermes install", ``gateway install`` refused without
``--force``, and ``refresh_launchd_plist_if_needed()`` never rewrote the plist — leaving the
long-running gateway (the process that fetches arbitrary URLs) on the previous Node binary (#144735).

The masking still has a reason to exist: the PATH tail is whatever shell ran ``hermes gateway
install``, and that must not flag a plist as stale. So these tests pin BOTH halves: the deterministic
Hermes-owned prefix is compared, the shell tail is not.
"""

from __future__ import annotations

import json

from hermes_cli import gateway as gateway_cli


def _seed_managed_node(home, version: str) -> str:
    """Record a pm-managed Node version the way ``pm install`` writes ``facts.json``.

    ``_append_node_dir_for_service`` resolves the Node dir through ``Facts.env_for`` against the
    target home's store (``{{store}}`` templates), so the record — not the test's PATH — decides what
    the generated definition pins.
    """
    store = home / "tools"
    entry = f"node-{version}-darwin-arm64"
    node_dir = store / entry
    node_dir.mkdir(parents=True)
    facts = {
        "schema": 1,
        "packages": {"node": {"entry": entry, "version": version, "env": {"PATH": [f"{{{{store}}}}/{entry}"]}}},
    }
    (store / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    return str(node_dir)


def _install_plist(monkeypatch, tmp_path, home) -> tuple:
    """Write the plist a ``gateway install`` would leave behind for the current managed Node."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")
    return plist_path, plist_path.read_text(encoding="utf-8")


def test_a_managed_node_bump_makes_the_installed_plist_stale(monkeypatch, tmp_path):
    """The pm-managed Node dir is part of the generated definition, so a bump must flag the plist."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    old_dir = _seed_managed_node(home, "26.7.0")
    _, installed = _install_plist(monkeypatch, tmp_path, home)
    assert old_dir in installed
    assert gateway_cli.launchd_plist_is_current() is True

    new_dir = _seed_managed_node(home, "26.12.1")

    assert gateway_cli.launchd_plist_is_current() is False
    refreshed = gateway_cli.generate_launchd_plist()
    assert new_dir in refreshed
    assert old_dir not in refreshed


def test_a_changed_shell_path_does_not_flag_the_installed_plist(monkeypatch, tmp_path):
    """The PATH tail is the invoking shell's; a different shell must not report a stale service."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_managed_node(home, "26.7.0")
    _install_plist(monkeypatch, tmp_path, home)
    assert gateway_cli.launchd_plist_is_current() is True

    monkeypatch.setenv("PATH", f"{tmp_path / 'somewhere-else' / 'bin'}:/usr/bin:/bin")

    assert gateway_cli.launchd_plist_is_current() is True


def test_an_installed_plist_without_the_managed_node_dir_is_stale(monkeypatch, tmp_path):
    """A definition written before pm recorded Node pins no store dir; that is drift, not a match."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    node_dir = _seed_managed_node(home, "26.7.0")
    plist_path, installed = _install_plist(monkeypatch, tmp_path, home)
    assert node_dir in installed

    # Stand-in for a definition generated before the store Node was recorded (or by another install):
    # the same plist with the managed dir dropped from its PATH.
    plist_path.write_text(installed.replace(f"{node_dir}:", "", 1), encoding="utf-8")

    assert gateway_cli.launchd_plist_is_current() is False
