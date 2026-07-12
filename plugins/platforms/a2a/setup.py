"""Configuration operations shared by gateway setup and ``hermes a2a``."""

from __future__ import annotations

import threading
import secrets
from contextlib import contextmanager
from typing import Any

from hermes_constants import get_hermes_home

from . import auth
from . import config as a2a_config

_PROCESS_SETUP_LOCK = threading.RLock()


def setup_lock_path():
    return get_hermes_home() / "a2a" / "setup.lock"


@contextmanager
def _setup_transaction():
    """Serialize setup across processes; lock order is setup then credential."""
    auth._secure_store_directory(auth.credentials_path())
    with _PROCESS_SETUP_LOCK, auth._safe_file_lock(setup_lock_path()):
        yield


def _ensure_a2a_platform_config_unlocked(*, public_url: str | None = None) -> None:
    normalized_url = a2a_config.validate_peer_url(public_url) if public_url else None

    def mutate(root: dict[str, Any]) -> None:
        extra = a2a_config.a2a_extra(root)
        platform = root["platforms"]["a2a"]
        platform["enabled"] = True
        extra.setdefault("host", "127.0.0.1")
        extra.setdefault("port", 8645)
        extra.setdefault("principals", {})
        extra.setdefault("peers", {})
        if normalized_url is not None:
            extra["public_url"] = normalized_url
        toolsets = root.setdefault("platform_toolsets", {})
        if not isinstance(toolsets, dict):
            raise ValueError("platform_toolsets config must be a mapping")
        toolsets["a2a"] = []

    a2a_config.update_a2a_config(mutate)


def ensure_a2a_platform_config(*, public_url: str | None = None) -> None:
    with _setup_transaction():
        _ensure_a2a_platform_config_unlocked(public_url=public_url)


def add_principal(name: str, *, profile: str) -> str:
    name = a2a_config.validate_name(name, label="principal name")
    profile = a2a_config.validate_name(profile, label="profile")
    with _setup_transaction():
        _ensure_a2a_platform_config_unlocked()
        if name in a2a_config.load_a2a_settings().principals:
            raise ValueError(f"principal {name} already exists; use credential rotate")
        ref = f"inbound:{name}"
        token = auth.create_inbound_credential(ref)
        try:
            def mutate(root: dict[str, Any]) -> None:
                principals = a2a_config.a2a_extra(root).setdefault("principals", {})
                if not isinstance(principals, dict):
                    raise ValueError("A2A principals config must be a mapping")
                principals[name] = {"credential_ref": ref, "profile": profile}

            a2a_config.update_a2a_config(mutate)
        except Exception:
            auth.delete_credential(ref, direction="inbound")
            raise
        return token


def remove_principal(name: str) -> bool:
    name = a2a_config.validate_name(name, label="principal name")
    with _setup_transaction():
        entry = a2a_config.load_a2a_settings().principals.get(name)
        if entry is None:
            return False
        ref = entry.get("credential_ref")
        snapshot = (
            auth._delete_credential_with_snapshot(ref, direction="inbound") if ref else None
        )
        try:
            def mutate(root: dict[str, Any]) -> None:
                principals = a2a_config.a2a_extra(root).setdefault("principals", {})
                if isinstance(principals, dict):
                    principals.pop(name, None)

            a2a_config.update_a2a_config(mutate)
        except Exception:
            if snapshot is not None:
                auth._restore_credential_snapshot(snapshot, direction="inbound")
            raise
        return True


def add_peer(name: str, *, url: str, token: str) -> None:
    name = a2a_config.validate_name(name, label="peer name")
    url = a2a_config.validate_peer_url(url)
    with _setup_transaction():
        _ensure_a2a_platform_config_unlocked()
        if name in a2a_config.load_a2a_settings().peers:
            raise ValueError(f"peer {name} already exists; remove it before adding")
        ref = f"outbound:{name}"
        generation = secrets.token_urlsafe(24)
        auth.store_outbound_credential(ref, token)
        try:
            def mutate(root: dict[str, Any]) -> None:
                peers = a2a_config.a2a_extra(root).setdefault("peers", {})
                if not isinstance(peers, dict):
                    raise ValueError("A2A peers config must be a mapping")
                peers[name] = {
                    "credential_ref": ref,
                    "url": url,
                    "generation": generation,
                }

            a2a_config.update_a2a_config(mutate)
        except Exception:
            auth.delete_credential(ref, direction="outbound")
            raise


def remove_peer(name: str) -> bool:
    name = a2a_config.validate_name(name, label="peer name")
    with _setup_transaction():
        entry = a2a_config.load_a2a_settings().peers.get(name)
        if entry is None:
            return False
        ref = entry.get("credential_ref")
        from . import client_state

        state_snapshot = client_state._snapshot_peer_state_unlocked(name)
        snapshot = (
            auth._delete_credential_with_snapshot(ref, direction="outbound") if ref else None
        )
        try:
            client_state._clear_peer_state_unlocked(name)

            def mutate(root: dict[str, Any]) -> None:
                peers = a2a_config.a2a_extra(root).setdefault("peers", {})
                if isinstance(peers, dict):
                    peers.pop(name, None)

            a2a_config.update_a2a_config(mutate)
        except Exception:
            try:
                if snapshot is not None:
                    auth._restore_credential_snapshot(snapshot, direction="outbound")
            finally:
                client_state._restore_peer_state_unlocked(name, state_snapshot)
            raise
        return True


def rotate_principal_credential(name: str) -> str:
    name = a2a_config.validate_name(name, label="principal name")
    with _setup_transaction():
        entry = a2a_config.load_a2a_settings().principals.get(name)
        if entry is None or not entry.get("credential_ref"):
            raise KeyError("principal not found")
        return auth.rotate_inbound_credential(entry["credential_ref"])


def gateway_setup() -> None:
    ensure_a2a_platform_config()
    print("A2A platform enabled with zero default tools.")
    print("Add an inbound principal with: hermes a2a principal add NAME --profile PROFILE")
