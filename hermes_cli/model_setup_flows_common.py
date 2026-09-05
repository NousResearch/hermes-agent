"""Shared skeleton for the ``_model_flow_*`` wizards: banner, credentials, model list, picker,
persist, confirmation — one copy of each step. Prompt strings and the config keys written (and
their insertion order = config.yaml key order) are behavior; keep them byte-identical.

main_provider_setup / auth / config imports are lazy on purpose: main.py imports the flows (import
cycle) and tests patch ``hermes_cli.config.load_config`` etc. at call time.
"""

from __future__ import annotations

import contextlib
import subprocess

from hermes_cli.cli_output import line_input
from hermes_cli.config import clear_model_endpoint_credentials

_HTTP = ("http://", "https://")


def _say(*lines: str) -> None:
    """``print`` each line (``""`` = blank line); one call per banner block."""
    print("\n".join(lines))


def _ask(prompt: str, *, secret: bool = False, raw: bool = False, cancel_msg: str | None = "\nCancelled.",
         on_cancel=None):
    """One stripped prompt answer. ``raw`` uses builtin ``input`` (numbered fallbacks),
    ``secret`` the masked prompt, else ``line_input``. Ctrl-C/EOF prints *cancel_msg*
    (``None`` = silent) and returns *on_cancel*."""
    if secret:
        from hermes_cli.secret_prompt import masked_secret_prompt as fn
    else:
        fn = input if raw else line_input
    try:
        return fn(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        if cancel_msg is not None:
            print(cancel_msg)
        return on_cancel


def _existing_api_key_for_model_flow(provider_id: str, pconfig) -> tuple[str, str]:
    """Resolve an existing wizard credential without changing its storage."""
    from hermes_cli.auth import _resolve_api_key_provider_secret
    return _resolve_api_key_provider_secret(provider_id, pconfig)


def _ensure_flow_api_key(provider_id: str, pconfig, *, missing_hint=()) -> tuple[str, str, bool]:
    """Resolve the stored key, print *missing_hint* lines when none exists, then run
    ``_prompt_api_key`` (users can replace a stale key in-flow via K/R/C).

    Returns ``(existing_key, resolved_key, abort)``.
    """
    from hermes_cli.main_provider_setup import _prompt_api_key
    existing_key, existing_source = _existing_api_key_for_model_flow(provider_id, pconfig)
    if not existing_key:
        for line in missing_hint:
            print(line)
    resolved, abort = _prompt_api_key(pconfig, existing_key, provider_id=provider_id, existing_source=existing_source)
    return existing_key, resolved, abort


def _load_config_model_section() -> tuple[dict, dict]:
    """Return ``(cfg, cfg["model"])`` with the model section coerced to a dict."""
    from hermes_cli.config import load_config
    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    return cfg, model


def _begin_model_config(selected: str, provider: str) -> tuple[dict, dict]:
    """Record *selected* as the model choice and open the config model section
    with ``provider`` set; callers set endpoint fields then ``_commit_model_config``."""
    from hermes_cli.auth import _save_model_choice
    _save_model_choice(selected)
    cfg, model = _load_config_model_section()
    model["provider"] = provider
    return cfg, model


def _commit_model_config(cfg: dict) -> None:
    """Persist *cfg* and deactivate any OAuth provider."""
    from hermes_cli.auth import deactivate_provider
    from hermes_cli.config import save_config
    save_config(cfg)
    deactivate_provider()


def _persist_model(selected: str, provider: str, *, base_url: str | None = None, api_mode: str | None = None,
                   drop_base_url: bool = False, drop_api_mode: bool = False, clear_creds: bool = True,
                   finish=None) -> dict:
    """The standard persist step: ``_save_model_choice`` → model section with ``provider``,
    ``base_url`` then ``api_mode`` (that order is the config.yaml key order) → scrub inline
    endpoint credentials (``clear_creds``; ``drop_api_mode`` also pops ``api_mode``) →
    *finish(cfg, model)* for extra sections → save + deactivate OAuth provider."""
    cfg, model = _begin_model_config(selected, provider)
    if base_url is not None:
        model["base_url"] = base_url
    if api_mode is not None:
        model["api_mode"] = api_mode
    if drop_base_url:
        model.pop("base_url", None)
    if clear_creds:
        clear_model_endpoint_credentials(model, clear_api_mode=drop_api_mode)
    elif drop_api_mode:
        model.pop("api_mode", None)
    if finish is not None:
        finish(cfg, model)
    _commit_model_config(cfg)
    return model


def _finish_model(selected, provider: str, done: str, *, no_change: str = "No change.", **persist_kw):
    """``_persist_model`` + confirmation line when *selected*, else the no-change line.
    Returns the saved model section (None when nothing was selected)."""
    if not selected:
        print(no_change)
        return None
    model = _persist_model(selected, provider, **persist_kw)
    print(done)
    return model


def _activate_provider_model(selected, provider_id: str, base_url: str, done: str,
                             no_change: str | None = "No change.") -> None:
    """OAuth-provider persist: model choice + ``_update_config_for_provider`` (which owns
    the auth-state bookkeeping), then *done*; *no_change* (``None`` = silent) otherwise."""
    from hermes_cli.auth import _save_model_choice, _update_config_for_provider
    if not selected:
        if no_change is not None:
            print(no_change)
        return
    _save_model_choice(selected)
    _update_config_for_provider(provider_id, base_url)
    print(done)


def _ensure_dict_section(cfg: dict, key: str) -> dict:
    """Return ``cfg[key]`` as a dict, replacing a missing/non-dict value."""
    section = cfg.get(key)
    if not isinstance(section, dict):
        section = {}
    cfg[key] = section
    return section


def _pick_model_or_prompt(model_list, prompt: str, **kwargs):
    """Radio picker when *model_list* is non-empty, else a free-text ``line_input``
    (None on Ctrl-C/EOF)."""
    from hermes_cli.auth import _prompt_model_selection
    if model_list:
        return _prompt_model_selection(model_list, **kwargs)
    return _ask(prompt, cancel_msg=None)


def _run_login(login_fn, *args, **kwargs) -> bool:
    """Run an OAuth login helper; print the standard failure line and return False
    on SystemExit / any exception."""
    try:
        login_fn(*args, **kwargs)
    except SystemExit:
        print("Login cancelled or failed.")
        return False
    except Exception as exc:
        print(f"Login failed: {exc}")
        return False
    return True


def _oauth_gate(logged_in: bool, name: str, login_fn, *login_args, fresh_name: str = "", recheck=None) -> bool:
    """Login-or-reuse step for OAuth providers. Not logged in → start a login. Logged in →
    use / reauthenticate (``force_new_login``; *recheck()* must then report logged in) /
    cancel. Returns False when the flow must stop."""
    if not logged_in:
        _say(f"Not logged into {name}. Starting login...", "")
        return _run_login(login_fn, *login_args)
    _say(f"  {name} credentials: ✓", "")
    choice = _prompt_auth_credentials_choice(f"{name} credentials:")
    if choice == "cancel":
        return False
    if choice == "reauth":
        _say(f"Starting a fresh {fresh_name or name} login...", "")
        if not _run_login(login_fn, *login_args, force_new_login=True):
            return False
        if recheck is not None and not recheck():
            print("Login failed.")
            return False
    return True


def _models_dev_merged(provider_id: str, curated) -> list:
    """models.dev agentic models for *provider_id* plus curated ids not yet listed
    (case-insensitive). Empty list when models.dev has nothing / is unavailable."""
    mdev_models: list = []
    with contextlib.suppress(Exception):
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models(provider_id)
    if not mdev_models:
        return []
    seen = {m.lower() for m in mdev_models}
    merged = list(mdev_models)
    for m in curated:
        if m.lower() not in seen:
            merged.append(m)
            seen.add(m.lower())
    return merged


def _show_curated(model_list) -> None:
    if model_list:
        print(f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.')


def _prune_replaced_custom_model_config_credentials(base_url: str, *, provider_name: str = "") -> None:
    """Drop stale ``model_config`` ("the credential under ``model.api_key``") entries from inactive
    custom pools: after an explicit custom-endpoint switch an old pool still carrying that source
    points at the previous endpoint and could be selected before the fresh config."""
    try:
        from agent.credential_pool import CUSTOM_POOL_PREFIX, custom_provider_pool_key_candidates
        from hermes_cli.auth import read_credential_pool, write_credential_pool

        # A keyed ``providers.<key>`` endpoint stores under the durable slug while
        # legacy pools keep ``custom:<display-name>``; every identity the active
        # endpoint may occupy must be skipped or its own legacy pool gets pruned.
        # See #100413.
        active_pool_keys = {
            str(key).strip().lower()
            for key in custom_provider_pool_key_candidates(base_url, provider_name=provider_name or None)}
        if not active_pool_keys:
            return
        pools = read_credential_pool(None)
        if not isinstance(pools, dict):
            return
        for pool_key, entries in pools.items():
            if (
                not isinstance(pool_key, str)
                or not pool_key.startswith(CUSTOM_POOL_PREFIX)
                or pool_key in active_pool_keys
                or not isinstance(entries, list)):
                continue
            retained = [e for e in entries if not (isinstance(e, dict) and e.get("source") == "model_config")]
            if len(retained) != len(entries):
                removed_ids = [
                    str(e["id"]) for e in entries
                    if isinstance(e, dict) and e.get("source") == "model_config" and e.get("id")]
                write_credential_pool(pool_key, retained, removed_ids=removed_ids)
    except Exception:
        return


def _curses_choice(title: str, rows: list, default_idx: int):
    """``_curses_prompt_choice`` index, or None when curses is unavailable (piped stdin,
    non-TTY) so the caller can fall back to a numbered prompt."""
    try:
        from hermes_cli.setup import _curses_prompt_choice
        return _curses_prompt_choice(title, rows, default_idx)
    except Exception:
        return None


def _radiolist(title: str, items: list, default_idx: int = 0, **kw):
    """``curses_radiolist`` index (-1 = cancelled), or None when curses is unavailable so the
    caller can fall back to a numbered prompt."""
    try:
        from hermes_cli.curses_ui import curses_radiolist
        return curses_radiolist(title, items, selected=default_idx, cancel_returns=-1, **kw)
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        return None


def _print_numbered(title: str, rows: list, default_idx: int) -> None:
    print(title)
    for i, row in enumerate(rows, 1):
        marker = "→" if (i - 1) == default_idx else " "
        print(f"  {marker} {i}. {row}")


def _prompt_auth_credentials_choice(title: str) -> str:
    """Prompt for reuse / reauthenticate / cancel with the standard radio UI.

    Returns one of ``"use"``, ``"reauth"``, ``"cancel"``. Falls back to a
    numbered prompt when curses is unavailable (piped stdin, non-TTY).
    """
    choices = ["Use existing credentials", "Reauthenticate (new OAuth login)", "Cancel"]
    idx = _curses_choice(title, choices, 0)
    if idx is not None and idx >= 0:
        print()
        return ("use", "reauth", "cancel")[idx]
    _print_numbered(title, choices, 0)
    print()
    choice = _ask("  Choice [1/2/3]: ", raw=True, cancel_msg=None, on_cancel="1")
    return {"2": "reauth", "3": "cancel"}.get(choice, "use")
