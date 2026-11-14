"""Profile-ID gates take their grammar from ``hermes_constants.PROFILE_ID_RE`` (state-24, #155595).

Each gate below once held its own copy of ``^[a-z0-9][a-z0-9_-]{0,63}$``. Copies drift silently: the
owner is changed and one gate keeps accepting or refusing the old set. Swapping the owner for a
different grammar shows the gate follows it, so a re-introduced copy fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import hermes_constants

# Accepted by the real grammar, refused by the swapped one; then the reverse.
OLD_ONLY = "work"
NEW_ONLY = "zz.custom"
SWAPPED = re.compile(r"^zz\.[a-z]+$")


def _status_label(name: str):
    from gateway.status import _profile_label_for_home

    return _profile_label_for_home(f"/nonexistent-root/profiles/{name}")


def _lock_owner(name: str):
    from gateway.status import scoped_lock_owner_label

    return scoped_lock_owner_label({"profile": name})


def _session_key_profile(name: str):
    from gateway.run import _parse_session_key

    return (_parse_session_key(f"agent:{name}:telegram:dm:1") or {}).get("profile")


def _cli_profile_flag(name: str):
    from hermes_cli._parser import scan_profile_flag_quietly

    scanned = scan_profile_flag_quietly(["-p", name, "chat"])
    return scanned[0] if scanned else None


def _ssh_spawn_arg(name: str):
    from hermes_cli.gateway_connections_ssh_remote import backend_profile_arg

    return backend_profile_arg(name)


def _saved_remote_profile(name: str):
    from hermes_cli.gateway_connections_normalize import normalize_ssh_config

    return (normalize_ssh_config({"mode": "ssh", "host": "h", "remoteProfile": name}) or {}).get("remoteProfile")


def _service_profile_name(name: str):
    from hermes_cli.gateway import _profile_name_from_home

    root = Path("/nonexistent-root")
    return _profile_name_from_home(root / "profiles" / name, root)


def _status_key(name: str):
    from hermes_cli.web_routers.status import _is_profile_platform_status_key

    return name if _is_profile_platform_status_key(f"{name}:telegram") else None


GATES = {
    "gateway.status._profile_label_for_home": _status_label,
    "gateway.status.scoped_lock_owner_label": _lock_owner,
    "gateway.run._parse_session_key": _session_key_profile,
    "hermes_cli._parser.scan_profile_flag": _cli_profile_flag,
    "gateway_connections_ssh_remote.backend_profile_arg": _ssh_spawn_arg,
    "gateway_connections_normalize.normalize_ssh_config": _saved_remote_profile,
    "hermes_cli.gateway._profile_name_from_home": _service_profile_name,
    "web_routers.status._is_profile_platform_status_key": _status_key,
}


@pytest.mark.parametrize("gate", sorted(GATES))
def test_gate_accepts_exactly_what_the_owner_accepts(gate, monkeypatch):
    check = GATES[gate]
    assert check(OLD_ONLY) == OLD_ONLY
    assert check(NEW_ONLY) is None

    monkeypatch.setattr(hermes_constants, "PROFILE_ID_RE", SWAPPED)

    assert check(NEW_ONLY) == NEW_ONLY
    assert check(OLD_ONLY) is None


def test_session_store_owner_follows_the_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from persistence.sqlite.conversation._store.sessions import _own_profile_name

    root = tmp_path / ".hermes"
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)

    def owner(name: str):
        home = root / "profiles" / name
        home.mkdir(parents=True, exist_ok=True)
        return _own_profile_name(SimpleNamespace(db_path=home / "state.db"))

    assert owner(OLD_ONLY) == OLD_ONLY
    assert owner(NEW_ONLY) is None
    monkeypatch.setattr(hermes_constants, "PROFILE_ID_RE", SWAPPED)
    assert owner(NEW_ONLY) == NEW_ONLY
    assert owner(OLD_ONLY) is None
