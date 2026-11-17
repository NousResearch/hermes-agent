"""Wiring into the plugin hook manager: registration from config (idempotent per home), the
gateway's resync with config and consent, and the launch/process opt-ins."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home

from agent.shell_hooks import consent, rewake
from agent.shell_hooks.consent import (
    _approved_pairs, _consented, _read_allowlist, _resolve_effective_accept, allowlist_path, allowlisted_pairs,
)
from agent.shell_hooks.runner import _make_callback
from agent.shell_hooks.spec import ShellHookSpec, _parse_hooks

logger = logging.getLogger(__name__)

# (home, event, matcher, command) wired in this process: matcher in the key (one script may register
# per-tool under one event), home so multiplexed-gateway profiles can register identical triples.
_registered: set[tuple[str, str, Optional[str], str]] = set()

_registered_lock = threading.Lock()

# sync_from_config: read + apply as one step, so an older snapshot never lands after a newer one.
_sync_lock = threading.Lock()

# Consent that lives only in this process (guarded by _sync_lock): (home, event, command) a launch's
# --accept-hooks approved but the allowlist does not hold yet, this process's own --accept-hooks, and
# the skipped hooks already logged.
_launch_consent: set[tuple[str, str, str]] = set()

_skip_logged: set[tuple[str, str, str]] = set()


def _home_key() -> str:
    return str(get_hermes_home().expanduser().resolve())


def _forget_home_registrations(registry: set[tuple], lock: threading.Lock) -> None:
    """Drop the current home's keys only (shared with outbound webhooks): profile A's reload must not drop B."""
    home_key = _home_key()
    with lock:
        registry.difference_update({k for k in registry if k[0] == home_key})


def register_from_config(cfg: Optional[dict[str, Any]], *, accept_hooks: bool = False,
                         interactive: bool = False) -> list[ShellHookSpec]:
    """Register every configured shell hook (idempotent); returns the newly wired specs. Skipped
    entries (unknown, malformed, not allowlisted, already registered) are logged only.

    ``interactive=True`` lets a hook that is not yet allowlisted ask at the terminal; only a
    foreground CLI command passes it. Every other caller (gateway startup and profile loads, cron
    and managed workers, tui_gateway, force-reload re-registration) never prompts: ``isatty()``
    cannot tell a hidden console or ``NUL`` from a person, and a prompt there blocks the event loop
    until the liveness watchdog kills the gateway (#120356, #127822)."""
    specs, _complete = _hook_specs(cfg)
    if not specs:
        return []
    from hermes_cli.plugins import get_plugin_manager  # lazy: avoids import cycle
    manager, home_key = get_plugin_manager(), _home_key()
    with _registered_lock:  # an already-wired hook is never re-asked for (main checked this first)
        specs = [spec for spec in specs if _key(home_key, spec) not in _registered]
    accept, allowed = _resolve_effective_accept(cfg, accept_hooks), allowlisted_pairs()
    # TTY prompt outside the lock; _wire takes the lock and re-checks.
    consented = []
    for spec in specs:
        if _consented(spec, accept, allowed, prompt=interactive):
            consented.append(spec)
        else:
            logger.warning("shell hook for %s (%s) not allowlisted in %s — skipped. Use --accept-hooks / "
                           "HERMES_ACCEPT_HOOKS=1 / hooks_auto_accept: true, or approve it at a CLI prompt "
                           "for this profile.", spec.event, spec.command, allowlist_path())
    with _registered_lock:
        return _wire(manager, home_key, consented)


def sync_from_config(load_cfg: Callable[[], dict[str, Any]]) -> None:
    """Bring this home's registered shell hooks in step with its config (``load_cfg()``) and
    consent, for a long-lived process (the gateway) that must behave as a new process would: hooks
    configured and consented since start register, hooks removed from config or revoked are
    unregistered. Consent is the allowlist, an opt-in in force (``hooks_auto_accept`` /
    ``HERMES_ACCEPT_HOOKS`` / this process's own ``--accept-hooks``), or a launch's ``--accept-hooks``
    whose allowlist write failed (``accept_for_launch``). Never prompts or writes. An empty config
    (a loader that failed open), a malformed ``hooks:`` entry or an unreadable allowlist must never
    disarm a ``fail_closed`` guard, so each adds but never removes. Config and allowlist are read
    under the same lock as the apply, so the last sync to run always applies the newest state."""
    from hermes_cli.plugins import get_plugin_manager  # lazy: avoids import cycle
    with _sync_lock:
        cfg = load_cfg()
        specs, complete = _hook_specs(cfg)
        if specs is None or not cfg:
            return
        allowlist, readable = _read_allowlist()
        allowed = _approved_pairs(allowlist["approvals"])
        home_key = _home_key()
        # A launch approval the allowlist now holds is the allowlist's to keep (or revoke).
        _launch_consent.difference_update({(home_key, *pair) for pair in allowed})
        allowed |= {(event, command) for home, event, command in _launch_consent if home == home_key}
        accept = _resolve_effective_accept(cfg, False)
        wanted = []
        for spec in specs:
            if accept or (spec.event, spec.command) in allowed:
                wanted.append(spec)
            elif (home_key, spec.event, spec.command) not in _skip_logged:  # once per process, not per open
                _skip_logged.add((home_key, spec.event, spec.command))
                logger.warning("shell hook for %s (%s) not allowlisted in %s — skipped. Approve it at a CLI "
                               "prompt for this profile, or use --accept-hooks / HERMES_ACCEPT_HOOKS=1 / "
                               "hooks_auto_accept: true.", spec.event, spec.command, allowlist_path())
        manager = get_plugin_manager()
        with _registered_lock:
            _wire(manager, home_key, wanted)
            if not (readable and complete):
                suspended = (home_key, "", "removal")
                if suspended not in _skip_logged:
                    _skip_logged.add(suspended)
                    logger.warning("shell hooks: %s, so removed or revoked hooks stay registered until it is "
                                   "fixed", "the allowlist is unreadable" if not readable
                                   else "a hooks: entry in config.yaml is malformed")
                return
            # In-memory consent ends with the hook's config entry, so a removed-and-re-added hook asks again.
            configured = {(home_key, spec.event, spec.command) for spec in specs}
            _launch_consent.difference_update({c for c in _launch_consent if c[0] == home_key and c not in configured})
            keep = {_key(home_key, spec) for spec in wanted}
            stale = {k for k in _registered if k[0] == home_key and k not in keep}
            _registered.difference_update(stale)
            for event in {k[1] for k in stale}:
                # Replaced, never edited in place: a concurrent invoke_hook keeps iterating the old list.
                manager._hooks[event] = [cb for cb in manager._hooks.get(event, [])
                                         if getattr(cb, "shell_hook_key", None) not in stale]
    rewake.cancel_hooks(stale)  # a revoked or removed async_rewake hook stops waiting now, not at its timeout
    for _home, event, _matcher, command, _mode in stale:
        logger.info("shell hook unregistered (removed from config or allowlist): %s -> %s", event, command)


def record_consent_from_config(cfg: Optional[dict[str, Any]], *, accept_hooks: bool = False) -> list[ShellHookSpec]:
    """First-use consent without registering anything, for a client whose turns run in a gateway that
    has no TTY: the prompt (or an opt-in) for each configured hook not yet allowlisted. The gateway
    registers from the same allowlist when the client opens its session. Returns the consented specs."""
    specs, _complete = _hook_specs(cfg)
    if not specs:
        return []
    accept, allowed = _resolve_effective_accept(cfg, accept_hooks), allowlisted_pairs()
    return [spec for spec in specs if _consented(spec, accept, allowed, prompt=True)]


def accept_for_launch(cfg: Optional[dict[str, Any]]) -> None:
    """A launch's ``--accept-hooks`` on the gateway: record every configured hook in the allowlist,
    and keep the consent in memory when that write fails, so it still registers
    the hook at the next ``sync_from_config`` (main registered it in memory). Only an unsaved approval
    is held, so a later ``hermes hooks revoke`` of a saved one takes effect."""
    specs = record_consent_from_config(cfg, accept_hooks=True)
    if not specs:
        return
    saved, home_key = allowlisted_pairs(), _home_key()
    with _sync_lock:
        _launch_consent.update((home_key, spec.event, spec.command) for spec in specs
                               if (spec.event, spec.command) not in saved)


def _hook_specs(cfg: Optional[dict[str, Any]]) -> tuple[Optional[list[ShellHookSpec]], bool]:
    """``(specs, complete)``: the configured hooks, or ``None`` when this process must run none
    (safe worker, safe mode, no config) — the one guard every registrar and the consent path share;
    ``complete`` is False when a configured entry was dropped as malformed."""
    from agent.safe_worker_policy import safe_worker_enabled
    from utils import env_var_enabled
    if safe_worker_enabled() or not isinstance(cfg, dict):
        return None, True
    if env_var_enabled("HERMES_SAFE_MODE"):  # hooks are user customizations too — fire zero user-configured code
        logger.info("HERMES_SAFE_MODE=1 — shell-hook registration skipped")
        return None, True
    return _parse_hooks(cfg.get("hooks"))


def _key(home_key: str, spec: ShellHookSpec) -> tuple[str, str, Optional[str], str, bool]:
    # The mode is part of the identity: switching async_rewake re-registers the hook at the next sync.
    return (home_key, spec.event, spec.matcher, spec.command, spec.async_rewake)


def _wire(manager: Any, home_key: str, specs: list[ShellHookSpec]) -> list[ShellHookSpec]:
    """Register ``specs`` not yet wired for this home; caller holds ``_registered_lock``."""
    registered = []
    for spec in specs:
        key = _key(home_key, spec)
        if key in _registered:
            continue
        callback = _make_callback(spec)
        callback.shell_hook_key = key  # type: ignore[attr-defined]  # identifies it for sync_from_config
        manager._hooks.setdefault(spec.event, []).append(callback)
        _registered.add(key)
        registered.append(spec)
        logger.info("shell hook registered: %s -> %s (matcher=%s, timeout=%ds, fail_closed=%s, async_rewake=%s)",
                    spec.event, spec.command, spec.matcher, spec.timeout, spec.fail_closed, spec.async_rewake)
    if any(spec.async_rewake for spec in registered):
        rewake.install_housekeeping(manager)
    return registered


def re_register_config_hooks() -> None:
    """Re-register after a plugin force-reload cleared the manager's hooks; only this home's keys
    are cleared (profile A's reload never drops B), never re-prompts.

    ``PluginManager.discover_and_load(force=True)`` unloads via the ownership ledger and clears the
    manager's ``_hooks`` dict, which silently drops shell hooks that were registered from ``config.yaml`` at
    startup (they are config-owned, not plugin-owned, so the ledger cannot restore them). Clear the
    idempotence set and re-run ``register_from_config()`` so hooks are wired again (#60036 / PR #60267;
    tracking #64178 — salvaged from PR #64188).
    Only the idempotence keys for the *current* Hermes home are cleared — ``discover_and_load(force=True)``
    only unloads the manager scoped to that one home, so clearing every home's keys would make a
    force-reload in profile A drop profile B's still-live registration from the ledger and duplicate it on
    B's next registration call (#92682 review).
    """
    _forget_home_registrations(_registered, _registered_lock)
    from hermes_cli.config import load_config
    register_from_config(load_config())


def reset_for_tests() -> None:
    """Test-only: clear the idempotence set and stop every ``async_rewake`` run."""
    rewake.reset_for_tests()
    with _registered_lock:
        _registered.clear()
    _launch_consent.clear()
    _skip_logged.clear()
    consent._process_accept = False
