"""hermes fallback — manage the fallback provider chain (tried in order when the primary fails).

Subcommands: ``list`` (default), ``add`` (same picker as `hermes model`), ``remove``, ``clear``.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from hermes_cli.fallback_config import get_fallback_chain

# Normalized fallback chain (merges legacy ``fallback_model``); always a fresh copy.
_read_chain = get_fallback_chain


def _identity(entry: Dict[str, Any]):
    """BackendIdentity for a ``{provider, model, base_url?}`` entry."""
    from agent.backend_identity import BackendIdentity
    return BackendIdentity.build(provider=entry.get("provider"), model=entry.get("model"), base_url=entry.get("base_url"))


def _write_chain(config: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
    """Persist the chain to ``fallback_providers``; drop the legacy key so there is one source of truth."""
    config["fallback_providers"] = chain
    config.pop("fallback_model", None)


def _format_entry(entry: Dict[str, Any]) -> str:
    """One-line human-readable rendering of a fallback entry."""
    base = entry.get("base_url")
    return f"{entry.get('model', '?')}  (via {entry.get('provider', '?')}){f'  [{base}]' if base else ''}"


def _extract_fallback_from_model_cfg(model_cfg: Any) -> Optional[Dict[str, Any]]:
    """Pull the ``{provider, model, base_url?, api_mode?}`` dict from a ``config["model"]`` snapshot."""
    if not isinstance(model_cfg, dict):
        return None
    provider = (model_cfg.get("provider") or "").strip()
    model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()  # the picker writes ``model.default``
    if not provider or not model:
        return None
    entry: Dict[str, Any] = {"provider": provider, "model": model}
    entry.update({key: value for key in ("base_url", "api_mode") if (value := (model_cfg.get(key) or "").strip())})
    return entry


def _snapshot_auth_active_provider() -> Any:
    """Current ``active_provider`` in auth.json, or None if unavailable."""
    try:
        from hermes_cli.auth import _load_auth_store
        return _load_auth_store().get("active_provider")
    except Exception:
        return None


def _restore_auth_active_provider(value: Any) -> None:
    """Write back a snapshotted ``active_provider``; best-effort (user re-runs `hermes model`), never fails the add."""
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
        with _auth_store_lock():
            _save_auth_store({**_load_auth_store(), "active_provider": value})
    except Exception:
        pass


def _restore_model_cfg(model_before: Any) -> None:
    """Restore ``config["model"]`` to a previously-captured snapshot."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.pop("model", None)
    if model_before is not None:
        cfg["model"] = copy.deepcopy(model_before)
    save_config(cfg)


def _entries(n: int) -> str:
    return f"{n} {'entry' if n == 1 else 'entries'}"


def _print_chain(heading: str, chain: List[Dict[str, Any]]) -> None:
    print(f"  {heading} ({_entries(len(chain))}):")
    print("".join(f"    {i}. {_format_entry(entry)}\n" for i, entry in enumerate(chain, 1)))


def _load_chain(empty_message: str):
    """Load config + chain; print ``empty_message`` block and return ``(config, None)`` when empty."""
    from hermes_cli.config import load_config
    config = load_config()
    chain = _read_chain(config)
    if not chain:
        print(f"\n{empty_message}\n")
    return config, chain or None


def _describe_primary(config: Dict[str, Any]) -> Optional[str]:
    """One-line description of the primary model for display purposes."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "?").strip() or "?"
        model = (model_cfg.get("default") or model_cfg.get("model") or "?").strip() or "?"
        return f"{model}  (via {provider})"
    return model_cfg.strip() or None if isinstance(model_cfg, str) else None


def cmd_fallback_list(args) -> None:  # noqa: ARG001
    """Print the current fallback chain."""
    config, chain = _load_chain("  No fallback providers configured.")
    if chain is None:
        print("  Add one with:  hermes fallback add\n")
        return
    print()
    if primary := _describe_primary(config):
        print(f"  Primary:   {primary}\n")
    _print_chain("Fallback chain", chain)
    print("  Tried in order when the primary fails (rate-limit, 5xx, connection errors).")
    print("  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers\n")


def cmd_fallback_add(args) -> None:
    """Launch the same picker as `hermes model`, then append the selection to the chain."""
    from hermes_cli.main import _require_tty, select_provider_and_model
    from hermes_cli.config import load_config, save_config
    _require_tty("fallback add")

    # Snapshot BEFORE the picker runs: "picked" vs "cancelled" is decided by comparing before/after,
    # and the primary must be restored either way.
    model_before = copy.deepcopy(load_config().get("model"))
    active_provider_before = _snapshot_auth_active_provider()
    print("\n  Adding a fallback provider.  The picker below is the same one used by\n"
          "  `hermes model` — select the provider + model you want as a fallback.\n")

    def _restore() -> None:
        _restore_model_cfg(model_before)

        _restore_auth_active_provider(active_provider_before)
    try:
        select_provider_and_model(args=args)
    except SystemExit:  # some provider flows exit on auth failure — restore state and re-raise
        _restore()
        raise
    new_entry = _extract_fallback_from_model_cfg(load_config().get("model"))
    if not new_entry:  # picker didn't complete (user cancelled or flow bailed)
        _restore()
        print("\n  No fallback added.")
        return

    # Same deployment as the primary → nothing to add. Identity semantics are owned by
    # agent.backend_identity: same provider+model on a DIFFERENT explicit base_url is a different
    # backend (multi-endpoint pool) and a legitimate fallback.
    # Picker picked the same thing that's already the primary → nothing changed, and there's nothing useful
    # to add as a fallback to itself. See #54250, #57584, #62984.
    from agent.backend_identity import same_deployment
    new_ident = _identity(new_entry)
    primary_entry = _extract_fallback_from_model_cfg(model_before)
    if primary_entry and same_deployment(_identity(primary_entry), new_ident):
        _restore()
        print(f"\n  Selected model matches the current primary ({_format_entry(new_entry)}).")
        print("  A provider cannot be a fallback for itself — no change.")
        return

    # Restore the primary, then re-load (rather than mutating the post-picker config) because the
    # picker may have touched other top-level keys (custom_providers, credentials) we want to keep.
    _restore()
    final_cfg = load_config()
    chain = _read_chain(final_cfg)
    if any(same_deployment(_identity(existing), new_ident) for existing in chain):
        print(f"\n  {_format_entry(new_entry)} is already in the fallback chain — skipped.")
        return
    chain.append(new_entry)
    _write_chain(final_cfg, chain)
    save_config(final_cfg)
    print(f"\n  Added fallback: {_format_entry(new_entry)}")
    print(f"  Chain is now {_entries(len(chain))} long.\n")
    print("  Run `hermes fallback list` to view, or `hermes fallback remove` to delete.")


def cmd_fallback_remove(args) -> None:  # noqa: ARG001
    """Pick an entry from the chain and remove it."""
    from hermes_cli.config import save_config
    config, chain = _load_chain("  No fallback providers configured — nothing to remove.")
    if chain is None:
        return

    # The curses menu owns its own non-TTY guard and numbered fallback; -1 means cancelled.
    from hermes_cli.setup import _curses_prompt_choice
    idx = _curses_prompt_choice("Select a fallback to remove:", [_format_entry(e) for e in chain] + ["Cancel"], 0)
    if idx is None or idx < 0 or idx >= len(chain):
        print("\n  Cancelled — no change.")
        return
    removed = chain.pop(idx)
    _write_chain(config, chain)
    save_config(config)
    print(f"\n  Removed fallback: {_format_entry(removed)}")
    print(f"  Chain is now {_entries(len(chain))} long.\n" if chain else "  Fallback chain is now empty.\n")


def cmd_fallback_clear(args) -> None:  # noqa: ARG001
    """Remove all fallback entries (with confirmation)."""
    from hermes_cli.config import save_config
    config, chain = _load_chain("  No fallback providers configured — nothing to clear.")
    if chain is None:
        return
    print()
    _print_chain("Current fallback chain", chain)
    try:
        resp = input("  Clear all entries? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return
    if resp not in {"y", "yes"}:
        print("  Cancelled — no change.")
        return
    _write_chain(config, [])
    save_config(config)
    print("\n  Fallback chain cleared.\n")


def cmd_fallback(args) -> None:
    """Top-level dispatcher for ``hermes fallback [subcommand]``."""
    sub = getattr(args, "fallback_command", None)
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"Unknown fallback subcommand: {sub}")
        print("Use one of: list, add, remove, clear")
        raise SystemExit(2)
    handler(args)


_SUBCOMMANDS = {
    **dict.fromkeys((None, "", "list", "ls"), cmd_fallback_list), "add": cmd_fallback_add,
    **dict.fromkeys(("remove", "rm"), cmd_fallback_remove), "clear": cmd_fallback_clear,
}
