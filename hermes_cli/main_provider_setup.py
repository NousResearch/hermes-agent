"""Provider setup wizard helpers: provider picker, custom-provider save/remove, auxiliary-model
routing menu, API-key/reasoning prompts, Anthropic OAuth.

Split out of ``hermes_cli/main.py``. Names that still live in main are imported lazily at call time.
"""

import contextlib

from typing import Optional
from hermes_cli.model_setup_flows_common import _ask, _ensure_dict_section, _print_numbered, _radiolist, _say


def _is_profile_api_key_provider(provider_id: str) -> bool:
    """True when *provider_id* maps to a profile with ``auth_type='api_key'`` — the catch-all
    in select_provider_and_model() so plugin providers dispatch to the generic key flow."""
    try:
        from providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        return _p is not None and _p.auth_type == "api_key"
    except Exception:
        return False


_GENERIC_API_KEY_PROVIDERS = frozenset({
    "openai-api", "gemini", "deepseek", "xai", "zai", "kimi-coding-cn",
    "minimax", "minimax-cn", "kilocode", "opencode-zen", "opencode-go",
    "opencode-free", "alibaba", "huggingface", "xiaomi", "arcee", "gmi",
    "nvidia", "ollama-cloud", "tencent-tokenhub", "tencent-tokenplan", "lmstudio"})


def _short_url(url: str) -> str:
    """``https://host/path/`` -> ``host/path`` for menu labels."""
    return url.replace("https://", "").replace("http://", "").rstrip("/")


def _clear_stale_openai_base_url():
    """Remove OPENAI_BASE_URL from ~/.hermes/.env unless the active provider is 'custom' — a
    leftover value routes provider:auto auxiliary clients to the old custom endpoint."""
    from hermes_cli.config import get_env_value, save_env_value, load_config
    model_cfg = load_config().get("model", {})
    provider = (model_cfg.get("provider") or "").strip().lower() if isinstance(model_cfg, dict) else ""
    if provider == "custom" or not provider:
        return  # custom provider legitimately uses OPENAI_BASE_URL

    stale_url = get_env_value("OPENAI_BASE_URL")
    if stale_url:
        save_env_value("OPENAI_BASE_URL", "")
        shown = f"{stale_url[:40]}..." if len(stale_url) > 40 else stale_url
        print(f"Cleared stale OPENAI_BASE_URL from .env (was: {shown})")


# (task_key, display_name, short_description)
_AUX_TASKS: list[tuple[str, str, str]] = [
    ("vision", "Vision", "image/screenshot analysis"),
    ("compression", "Compression", "context summarization"),
    ("approval", "Approval", "smart command approval"),
    ("mcp", "MCP", "MCP tool reasoning"),
    ("title_generation", "Title generation", "session titles"),
    ("review", "Review", "/review reviewer subagent"),
    ("memory_query_rewrite", "Memory query rewrite", "memory retrieval queries"),
    ("tts_audio_tags", "TTS audio tags", "Gemini TTS tag insertion"),
    ("skills_hub", "Skills hub", "skills search/install"),
    ("triage_specifier", "Triage specifier", "kanban spec fleshing"),
    ("kanban_decomposer", "Kanban decomposer", "task decomposition"),
    ("profile_describer", "Profile describer", "auto profile descriptions"),
    ("curator", "Curator", "skill-usage review pass")]

# Special non-auxiliary task surfaced in the same picker: subagent delegation. Routing lives
# under top-level `delegation.*` (NOT `auxiliary.delegation`) because delegate_task spawns full
# child agents via tools/delegate_tool.py::_resolve_delegation_credentials(), which reads that
# section directly. "auto" means "inherit the parent agent" and is stored as empty strings —
# never persist the literal "auto", or it would be resolved as a provider name.
_DELEGATION_TASK_KEY = "delegation"
_DELEGATION_TASK_NAME = "Delegation"
_DELEGATION_TASK_DESC = "subagent model (delegate_task)"


def _all_aux_tasks() -> list[tuple[str, str, str]]:
    """Built-in aux tasks (in order) followed by plugin-registered ones
    (:meth:`hermes_cli.plugins.PluginContext.register_auxiliary_task`)."""
    tasks = list(_AUX_TASKS)
    # Plugin discovery failure must not break the aux config UI.
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for entry in get_plugin_auxiliary_tasks():
            tasks.append((entry["key"], entry["display_name"], entry["description"]))
    return tasks


def _format_aux_current(task_cfg: dict) -> str:
    """Render the current aux config for display in the task menu."""
    if not isinstance(task_cfg, dict):
        return "auto"
    base_url = str(task_cfg.get("base_url") or "").strip()
    provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    model = str(task_cfg.get("model") or "").strip()
    if base_url:
        return f"custom ({_short_url(base_url)})" + (f" · {model}" if model else "")
    if provider == "auto":
        return "auto" + (f" · {model}" if model else "")
    if model:
        return f"{provider} · {model}"
    return provider


def _delegation_cfg_as_task(cfg: dict) -> dict:
    """Project the top-level ``delegation`` section into aux-task shape (provider/model/
    base_url/api_key); an empty provider means "inherit parent" and renders as "auto"."""
    d = cfg.get("delegation")
    if not isinstance(d, dict):
        d = {}
    return {k: str(d.get(k) or "").strip() for k in ("provider", "model", "base_url", "api_key")}


def _aux_task_cfg(cfg: dict, task: str) -> dict:
    """The stored routing dict for *task* (delegation reads its top-level section)."""
    if task == _DELEGATION_TASK_KEY:
        return _delegation_cfg_as_task(cfg)
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    return aux.get(task, {}) if isinstance(aux.get(task), dict) else {}


def _aux_task_display_name(task: str) -> str:
    """Display name for a task key, covering the special delegation entry."""
    if task == _DELEGATION_TASK_KEY:
        return _DELEGATION_TASK_NAME
    return next((name for key, name, _ in _all_aux_tasks() if key == task), task)


def _save_aux_choice(task: str, *, provider: str, model: str = "", base_url: str = "",
                     api_key: str = "") -> None:
    """Persist an aux task's four routing fields (timeout etc. untouched; main model config never
    modified). ``delegation`` writes the top-level section, with "auto" stored as an empty provider."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    if task == _DELEGATION_TASK_KEY:
        entry = _ensure_dict_section(cfg, "delegation")
        provider = "" if provider == "auto" else provider
    else:
        entry = _ensure_dict_section(_ensure_dict_section(cfg, "auxiliary"), task)
    entry["provider"] = provider
    entry["model"] = model or ""
    entry["base_url"] = base_url or ""
    entry["api_key"] = api_key or ""
    save_config(cfg)


def _reset_aux_to_auto() -> int:
    """Reset every known aux task (built-in + plugin) back to auto/empty. Returns number reset."""
    from hermes_cli.config import load_config, save_config
    def _clear(entry: dict, auto: str) -> bool:
        # Only the routing fields; timeout/download_timeout (aux) and max_concurrent_children
        # etc. (delegation) are user-tuned and preserved. *auto* is the reset provider value
        # ("auto" for aux tasks, "" for delegation); anything else counts as a change.
        changed = False
        if entry.get("provider") not in {None, "", auto}:
            entry["provider"] = auto
            changed = True
        for field in ("model", "base_url", "api_key"):
            if entry.get(field):
                entry[field] = ""
                changed = True
        return changed

    cfg = load_config()
    aux = _ensure_dict_section(cfg, "auxiliary")
    count = sum(_clear(_ensure_dict_section(aux, task), "auto") for task, _name, _desc in _all_aux_tasks())
    dele = cfg.get("delegation")
    if isinstance(dele, dict):
        count += _clear(dele, "")
    save_config(cfg)
    return count


def _aux_config_menu() -> None:
    """Top-level auxiliary-model picker; loops until the user picks "Back"."""
    from hermes_cli.config import load_config
    while True:
        cfg = load_config()
        _say("", "  Auxiliary models — side-task routing", "",
             "  Side tasks (vision, compression, web extraction, etc.) default",
             '  to your main chat model.  "auto" means "use my main model" —',
             "  Hermes only falls back to a lightweight backend (OpenRouter,",
             "  Nous Portal) if the main model is unavailable.  Override a",
             "  task below if you want it pinned to a specific provider/model.", "")

        menu_tasks = _all_aux_tasks() + [(_DELEGATION_TASK_KEY, _DELEGATION_TASK_NAME, _DELEGATION_TASK_DESC)]
        name_col = max(len(name) for _, name, _ in menu_tasks) + 2
        desc_col = max(len(desc) for _, _, desc in menu_tasks) + 4
        entries = [
            (task_key, f"{name.ljust(name_col)}{('(' + desc + ')').ljust(desc_col)}"
                       f"{_format_aux_current(_aux_task_cfg(cfg, task_key))}")
            for task_key, name, desc in menu_tasks]
        entries.append(("__reset__", "Reset all to auto"))
        entries.append(("__back__", "Back"))

        idx = _prompt_provider_choice([label for _, label in entries], default=0)
        if idx is None:
            return
        key = entries[idx][0]
        if key == "__back__":
            return
        if key == "__reset__":
            n = _reset_aux_to_auto()
            _say(f"Reset {n} auxiliary task(s) to auto." if n else "All auxiliary tasks were already set to auto.",
                 "")
            continue
        _aux_select_for_task(key)


def _aux_select_for_task(task: str) -> None:
    """Pick a provider + model for one aux task and persist it. Rows come from
    ``build_aux_picker_rows()`` (shared substrate): only already-configured providers appear."""
    from hermes_cli.config import load_config
    from hermes_cli.inventory import build_aux_picker_rows, format_aux_picker_entries
    task_cfg = _aux_task_cfg(load_config(), task)
    current_provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    current_model = str(task_cfg.get("model") or "").strip()
    current_base_url = str(task_cfg.get("base_url") or "").strip()
    display_name = _aux_task_display_name(task)

    try:
        providers = build_aux_picker_rows(current_provider=current_provider, current_model=current_model,
                                          current_base_url=current_base_url)
    except Exception as exc:
        print(f"Could not detect authenticated providers: {exc}")
        providers = []

    # (slug, label, models); "auto" always first
    auto_marker = "  ← current" if current_provider == "auto" and not current_base_url else ""
    auto_label = "auto (inherit main agent)" if task == _DELEGATION_TASK_KEY else "auto (recommended)"
    entries: list[tuple[str, str, list[str]]] = [("__auto__", f"{auto_label}{auto_marker}", [])]
    entries.extend(format_aux_picker_entries(providers, current_provider=current_provider,
                                             current_base_url=current_base_url))
    custom_marker = "  ← current" if current_base_url else ""
    entries.append(("__custom__", f"Custom endpoint (direct URL){custom_marker}", []))
    entries.append(("__back__", "Back", []))

    _say("", f"  Configure {display_name} — current: {_format_aux_current(task_cfg)}", "")
    idx = _prompt_provider_choice([label for _, label, _ in entries], default=0)
    if idx is None:
        return
    slug, _label, models = entries[idx]
    if slug == "__back__":
        return
    if slug == "__auto__":
        _save_aux_choice(task, provider="auto", model="", base_url="", api_key="")
        print(f"{display_name}: reset to auto.")
    elif slug == "__custom__":
        _aux_flow_custom_endpoint(task, task_cfg)
    else:
        _aux_flow_provider_model(task, slug, models, current_model)


def _aux_flow_provider_model(task: str, provider_slug: str, curated_models: list,
                             current_model: str = "") -> None:
    """Prompt for a model under an already-authenticated provider, save to aux."""
    from hermes_cli.auth import _prompt_model_selection
    from hermes_cli.models_pricing import get_pricing_for_provider
    display_name = _aux_task_display_name(task)
    try:
        pricing = get_pricing_for_provider(provider_slug) or {}
    except Exception:
        pricing = {}

    model_list = list(curated_models)
    # _prompt_model_selection supports "Enter custom model name" and cancel; with no curated
    # list (rare) fall back to a raw input prompt.
    if not model_list:
        _say(f"No curated model list for {provider_slug}.", "Enter a model slug manually (blank = use provider default):")
        selected = _ask("Model: ", cancel_msg="")
        if selected is None:
            return
    else:
        selected = _prompt_model_selection(model_list, current_model=current_model, pricing=pricing,
                                           confirm_provider=provider_slug)
        if selected is None:
            print("No change.")
            return

    _save_aux_choice(task, provider=provider_slug, model=selected or "", base_url="", api_key="")
    if selected:
        print(f"{display_name}: {provider_slug} · {selected}")
    else:
        print(f"{display_name}: {provider_slug} (provider default model)")


def _aux_flow_custom_endpoint(task: str, task_cfg: dict) -> None:
    """Prompt for a direct OpenAI-compatible base_url + optional api_key/model."""
    display_name = _aux_task_display_name(task)
    current_base_url = str(task_cfg.get("base_url") or "").strip()
    current_model = str(task_cfg.get("model") or "").strip()

    _say("", f"  Custom endpoint for {display_name}",
         "  Provide an OpenAI-compatible base URL (e.g. http://localhost:11434/v1)", "")
    url = _ask(f"Base URL [{current_base_url}]: " if current_base_url else "Base URL: ", cancel_msg="")
    if url is None:
        return
    url = url or current_base_url
    if not url:
        print("No URL provided. No change.")
        return
    model = _ask(f"Model slug (optional) [{current_model}]: " if current_model else "Model slug (optional): ",
                 cancel_msg="")
    if model is None:
        return
    model = model or current_model
    api_key = _ask("API key (optional, blank = use OPENAI_API_KEY): ", secret=True, cancel_msg="")
    if api_key is None:
        return

    _save_aux_choice(task, provider="custom", model=model, base_url=url, api_key=api_key)
    print(f"{display_name}: custom ({_short_url(url)})" + (f" · {model}" if model else ""))


_CANCELLED = object()


def _ask_index(prompt: str, count: int, *, echo_cancel: bool):
    """Numbered-menu input loop: 0-based index in ``range(count)``, ``None`` on blank,
    ``_CANCELLED`` on Ctrl-C/EOF (*echo_cancel* prints a blank line first)."""
    while True:
        try:
            val = input(prompt).strip()
            if not val:
                return None
            idx = int(val) - 1
            if 0 <= idx < count:
                return idx
            print(f"Please enter 1-{count}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            if echo_cancel:
                print()
            return _CANCELLED


def _prompt_provider_choice(choices, *, default=0, title="Select provider:"):
    """Provider menu with curses arrow keys; numbered-list fallback when curses is unavailable
    (piped stdin, non-TTY). Returns the selected index, or None if the user cancels."""
    with contextlib.suppress(Exception):
        from hermes_cli.setup import _curses_prompt_choice
        idx = _curses_prompt_choice(title, choices, default)
        if idx >= 0:
            print()
            return idx

    _print_numbered(title, choices, default)
    print()
    idx = _ask_index(f"Choice [1-{len(choices)}] ({default + 1}): ", len(choices), echo_cancel=True)
    if idx is None:
        return default
    return None if idx is _CANCELLED else idx


_DEFAULT_QWEN_PORTAL_MODELS = [
    "qwen3-coder-plus", "qwen3-coder"]

# (mode value, label, description, accepted answers); "" = auto-detect
_CUSTOM_API_MODES = (
    ("", "Auto-detect", "Use Hermes URL heuristics; best for standard OpenAI-compatible endpoints.",
     ("1", "auto", "detect", "auto-detect")),
    ("chat_completions", "Chat Completions", "Use /chat/completions for standard OpenAI-compatible servers.",
     ("2", "chat", "chat_completions", "completions")),
    ("codex_responses", "Responses / Codex", "Use /responses for Codex-compatible tool-calling backends.",
     ("3", "responses", "codex", "codex_responses")),
    ("anthropic_messages", "Anthropic Messages", "Use /v1/messages for Anthropic-compatible endpoints.",
     ("4", "anthropic", "anthropic_messages", "messages")))
_CUSTOM_API_MODE_ANSWERS = {answer: value for value, _, _, answers in _CUSTOM_API_MODES for answer in answers}


def _prompt_custom_api_mode_selection(base_url: str, current_api_mode: str = "") -> Optional[str]:
    """Prompt for a custom provider API mode: an explicit mode string, or None for auto-detect."""
    from hermes_cli.runtime_provider import _detect_api_mode_for_url
    detected_mode = _detect_api_mode_for_url(base_url)
    default_mode = str(current_api_mode or "").strip().lower() or detected_mode or ""

    _say("", "Select API compatibility mode:")
    for idx, (value, label, description, _answers) in enumerate(_CUSTOM_API_MODES, 1):
        markers = [m for m, hit in (("detected", value == detected_mode), ("current", value == default_mode)) if hit]
        suffix = f" [{' / '.join(markers)}]" if markers else ""
        _say(f"  {idx}. {label}{suffix}", f"     {description}")

    try:
        raw = input("Choice [1-4, Enter to keep current/detected]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        raise

    if not raw:
        return default_mode or None
    if raw in _CUSTOM_API_MODE_ANSWERS:
        return _CUSTOM_API_MODE_ANSWERS[raw] or None
    print(f"Invalid API mode choice: {raw}. Falling back to auto-detect.")
    return None


def _auto_provider_name(base_url: str) -> str:
    """Display name from a custom endpoint URL, e.g. "Local (localhost:11434)" or
    "RunPod (xyz.runpod.io)" — the default offered during custom endpoint setup."""
    import re
    name = re.sub(r"/v1/?$", "", _short_url(base_url)).split("/")[0]
    if "localhost" in name or "127.0.0.1" in name:
        return f"Local ({name})"
    if "runpod" in name.lower():
        return f"RunPod ({name})"
    return name.capitalize()


def _custom_provider_api_key_config_value(provider_info, resolved_api_key=""):
    """Return the value that should be persisted for a custom provider key."""
    api_key_ref = str(provider_info.get("api_key_ref", "") or "").strip()
    if api_key_ref:
        return api_key_ref
    key_env = str(provider_info.get("key_env", "") or "").strip()
    if key_env and not str(provider_info.get("api_key", "") or "").strip():
        return f"${{{key_env}}}"
    return str(resolved_api_key or "").strip()


def _custom_provider_base_url_config_value(provider_info, resolved_base_url=""):
    """Return the value that should be persisted for a custom provider URL."""
    return str(provider_info.get("base_url_ref", "") or "").strip() or str(resolved_base_url or "").strip()


def _save_custom_provider(base_url, api_key="", model="", context_length=None, name=None, api_mode=None,
                          key_env=""):
    """Save a custom endpoint to ``custom_providers`` in config.yaml, deduplicated by base_url (an
    existing entry gets model / context_length / api_mode updated). *key_env* set means the caller
    already wrote the key to ``.env``; the entry references it instead of inlining the secret.

    See #69449.
    """
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list):
        providers = []
    for entry in providers:
        if not (isinstance(entry, dict) and entry.get("base_url", "").rstrip("/") == base_url.rstrip("/")):
            continue
        changed = False
        if model and entry.get("model") != model:
            entry["model"] = model
            changed = True
        if model and context_length:
            _ensure_dict_section(entry, "models")[model] = {"context_length": context_length}
            changed = True
        if api_mode:
            if entry.get("api_mode") != api_mode:
                entry["api_mode"] = api_mode
                changed = True
        elif "api_mode" in entry:
            entry.pop("api_mode", None)
            changed = True
        if key_env and (entry.get("key_env") != key_env or entry.get("api_key")):
            entry["key_env"] = key_env
            entry.pop("api_key", None)
            changed = True
        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
        return  # already saved, updated if needed

    name = name or _auto_provider_name(base_url)
    entry = {"name": name, "base_url": base_url}
    if key_env:
        entry["key_env"] = key_env
    elif api_key:
        entry["api_key"] = api_key
    if model:
        entry["model"] = model
    if api_mode:
        entry["api_mode"] = api_mode
    if model and context_length:
        entry["models"] = {model: {"context_length": context_length}}

    providers.append(entry)
    cfg["custom_providers"] = providers
    save_config(cfg)
    print(f'  💾 Saved to custom providers as "{name}" (edit in config.yaml)')


def _remove_custom_provider(config):
    """Let the user remove a saved custom provider from config.yaml."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list) or not providers:
        print("No custom providers configured.")
        return

    print("Remove a custom provider:\n")
    choices = [
        f"{entry.get('name', 'unnamed')} ({_short_url(entry.get('base_url', ''))})" if isinstance(entry, dict) else str(entry)
        for entry in providers]
    choices.append("Cancel")

    idx = _radiolist("Select provider to remove:", list(choices))
    if idx is not None:
        print()
        if idx < 0:
            idx = None
    else:
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        print()
        try:
            val = input(f"Choice [1-{len(choices)}]: ").strip()
            idx = int(val) - 1 if val else None
        except (ValueError, KeyboardInterrupt, EOFError):
            idx = None

    if idx is None or idx >= len(providers):
        print("No change.")
        return

    removed = providers.pop(idx)
    cfg["custom_providers"] = providers
    save_config(cfg)
    removed_name = removed.get("name", "unnamed") if isinstance(removed, dict) else str(removed)
    print(f'✅ Removed "{removed_name}" from custom providers.')


def _prompt_reasoning_effort_selection(efforts, current_effort=""):
    """Prompt for a reasoning effort. Returns effort, 'none', or None to keep current."""
    deduped = list(dict.fromkeys(str(effort).strip().lower() for effort in efforts if str(effort).strip()))
    canonical_order = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    ordered = [effort for effort in canonical_order if effort in deduped]
    ordered.extend(effort for effort in deduped if effort not in canonical_order)
    if not ordered:
        return None

    def _label(effort):
        return f"{effort}  ← currently in use" if effort == current_effort else effort

    disable_label = "Disable reasoning"
    skip_label = "Skip (keep current)"
    if current_effort == "none":
        default_idx = len(ordered)
    elif current_effort in ordered:
        default_idx = ordered.index(current_effort)
    elif "medium" in ordered:
        default_idx = ordered.index("medium")
    else:
        default_idx = 0

    n = len(ordered)
    idx = _radiolist("Select reasoning effort:", [_label(effort) for effort in ordered] + [disable_label, skip_label],
                     default_idx)
    if idx is not None:
        if idx < 0:
            return None
        print()
    else:
        print("Select reasoning effort:")
        for i, effort in enumerate(ordered, 1):
            print(f"  {i}. {_label(effort)}")
        _say(f"  {n + 1}. {disable_label}", f"  {n + 2}. {skip_label}", "")
        idx = _ask_index(f"Choice [1-{n + 2}] (default: keep current): ", n + 2, echo_cancel=False)
        if idx is None or idx is _CANCELLED:
            return None
    if idx < n:
        return ordered[idx]
    return "none" if idx == n else None


def _prompt_api_key(pconfig, existing_key: str, provider_id: str = "", existing_source: str = "") -> tuple:
    """API-key entry for ``hermes setup`` / ``hermes model``: first-time entry, or [K]eep / [R]eplace /
    [C]lear when a key exists (a malformed paste is recoverable without editing ``.env``).
    Returns ``(resolved_key, abort)``; ``abort=True`` means the caller must ``return`` at once."""
    from hermes_cli.auth import LMSTUDIO_NOAUTH_PLACEHOLDER
    from hermes_cli.config import save_env_value
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""

    def _prompt_new_key(*, allow_lmstudio_default: bool) -> str:
        lmstudio_default = provider_id == "lmstudio" and allow_lmstudio_default
        if lmstudio_default:
            prompt = f"{key_env} (Enter for no-auth default {LMSTUDIO_NOAUTH_PLACEHOLDER!r}): "
        else:
            prompt = f"{key_env} (or Enter to cancel): "
        entered = _ask(prompt, secret=True, cancel_msg="")
        if entered is None:
            return ""
        if not entered and lmstudio_default:
            return LMSTUDIO_NOAUTH_PLACEHOLDER
        return entered

    if not existing_key:
        print(f"No {pconfig.name} API key configured.")
        if not key_env:
            return "", True
        new_key = _prompt_new_key(allow_lmstudio_default=True)
        if not new_key:
            print("Cancelled.")
            return "", True
        save_env_value(key_env, new_key)
        _say("API key saved.", "")
        return new_key, False

    # Already configured — offer K / R / C
    from hermes_cli.env_loader import format_secret_source_suffix
    source_suffix = format_secret_source_suffix(key_env) if key_env else ""
    print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓{source_suffix}")
    if not key_env:
        # Nothing we can rewrite; just acknowledge and move on.
        print()
        return existing_key, False
    pool_backed = existing_source.startswith("credential_pool:")
    menu = "  [K]eep / [R]eplace (default K): " if pool_backed else "  [K]eep / [R]eplace / [C]lear (default K): "
    choice = _ask(menu, raw=True, cancel_msg="", on_cancel="k").lower()

    if choice.startswith("r"):
        new_key = _prompt_new_key(allow_lmstudio_default=False)
        if not new_key:
            _say("  No change.", "")
            return existing_key, False
        save_env_value(key_env, new_key)
        _say("  API key updated.", "")
        return new_key, False
    if choice.startswith("c") and not pool_backed:
        save_env_value(key_env, "")
        print(f"  API key cleared.  Re-run `hermes setup` to configure {pconfig.name} again.")
        return "", True
    # Keep (default, or any other input)
    print()
    return existing_key, False


def _infer_stepfun_region(base_url: str) -> str:
    """Infer the current StepFun region from the configured endpoint."""
    return "china" if "api.stepfun.com" in (base_url or "").strip().lower() else "international"


def _stepfun_base_url_for_region(region: str) -> str:
    from hermes_cli.auth import STEPFUN_STEP_PLAN_CN_BASE_URL, STEPFUN_STEP_PLAN_INTL_BASE_URL
    return STEPFUN_STEP_PLAN_CN_BASE_URL if region == "china" else STEPFUN_STEP_PLAN_INTL_BASE_URL


def _run_anthropic_oauth_flow(save_env_value):
    """Run the Claude OAuth setup-token flow. Returns True if credentials were saved."""
    from agent.anthropic_credentials import run_oauth_setup_token, read_claude_code_credentials, is_claude_code_token_valid
    from hermes_cli.config import save_anthropic_oauth_token, use_anthropic_claude_code_credentials

    def _activate_claude_code_credentials_if_available() -> bool:
        try:
            creds = read_claude_code_credentials()
        except Exception:
            creds = None
        if creds and (is_claude_code_token_valid(creds) or bool(creds.get("refreshToken"))):
            use_anthropic_claude_code_credentials(save_fn=save_env_value)
            print("  ✓ Claude Code credentials linked.")
            from hermes_constants import display_hermes_home as _dhh_fn
            print(f"    Hermes will use Claude's credential store directly instead of copying a setup-token into {_dhh_fn()}/.env.")
            return True
        return False

    def _paste_token(prompt: str):
        """Manual setup-token entry: True saved, False empty, None cancelled."""
        token = _ask(prompt, secret=True, cancel_msg="")
        if not token:
            return token
        save_anthropic_oauth_token(token, save_fn=save_env_value)
        print("  ✓ Setup-token saved.")
        return True

    try:
        _say("", "  Running 'claude setup-token' — follow the prompts below.",
             "  A browser window will open for you to authorize access.", "")
        token = run_oauth_setup_token()
        if token:
            if _activate_claude_code_credentials_if_available():
                return True
            save_anthropic_oauth_token(token, save_fn=save_env_value)
            print("  ✓ OAuth credentials saved.")
            return True

        # Subprocess completed but no token auto-detected — ask user to paste
        _say("", "  If the setup-token was displayed above, paste it here:", "")
        saved = _paste_token("  Paste setup-token (or Enter to cancel): ")
        if saved is None:
            return False
        if saved:
            return True
        print("  ⚠ Could not detect saved credentials.")
        return False

    except FileNotFoundError:
        # Claude CLI not installed — guide user through manual setup
        _say("", "  The 'claude' CLI is required for OAuth login.", "", "  To install and authenticate:", "",
             "    1. Install Claude Code:  npm install -g @anthropic-ai/claude-code",
             "    2. Run:                  claude setup-token",
             "    3. Follow the browser prompts to authorize",
             "    4. Re-run:               hermes model", "",
             "  Or paste an existing setup-token now (sk-ant-oat-...):", "")
        saved = _paste_token("  Setup-token (or Enter to cancel): ")
        if saved is None:
            return False
        if saved:
            return True
        print("  Cancelled — install Claude Code and try again.")
        return False


def _named_custom_provider_map(cfg) -> dict[str, dict[str, str]]:
    """Saved custom providers keyed by slug, with raw ``${ENV}`` refs preserved."""
    from hermes_cli.config import get_compatible_custom_providers, read_raw_config
    from hermes_cli.providers import custom_provider_slug

    # Raw (un-expanded) templates keyed by identity. ``get_compatible_custom_providers(
    # read_raw_config())`` is deliberately bypassed: its normalize step ``urlparse()``s
    # ``base_url`` and drops entries whose base_url is itself an env-ref template.
    raw_api_key_refs: dict[tuple, str] = {}
    raw_base_url_refs: dict[tuple, str] = {}
    raw_cfg = read_raw_config()

    raw_entries: list[tuple[str, str, dict]] = []
    raw_list = raw_cfg.get("custom_providers")
    if isinstance(raw_list, list):
        raw_entries.extend((e.get("name", ""), "", e) for e in raw_list if isinstance(e, dict))
    raw_providers = raw_cfg.get("providers")
    if isinstance(raw_providers, dict):
        raw_entries.extend((e.get("name", "") or k, k, e) for k, e in raw_providers.items() if isinstance(e, dict))
    for name, provider_key, raw_entry in raw_entries:
        template = str(raw_entry.get("api_key", "") or "").strip()
        base_template = str(raw_entry.get("base_url", "") or raw_entry.get("url", "") or raw_entry.get("api", "") or "").strip()
        name = str(name or "").strip()
        provider_key = str(provider_key or "").strip()
        model = str(raw_entry.get("model", "") or raw_entry.get("default_model", "") or "").strip()
        # Index by every identity the loaded (expanded) config might present: (name),
        # (name, model), (provider_key), (provider_key, model); case-insensitive names.
        keys = [k.lower() for k in (name, provider_key) if k]
        identities = [(k,) for k in keys] + [(k, model) for k in keys]
        for refs, tmpl in ((raw_api_key_refs, template), (raw_base_url_refs, base_template)):
            if "${" in tmpl:
                for identity in identities:
                    refs.setdefault(identity, tmpl)

    def _lookup_ref(refs: dict[tuple, str], name: str, provider_key: str, model: str) -> str:
        name_lc = str(name or "").strip().lower()
        pkey_lc = str(provider_key or "").strip().lower()
        model = str(model or "").strip()
        return next((refs[i] for i in ((pkey_lc, model), (pkey_lc,), (name_lc, model), (name_lc,)) if i[0] and i in refs), "")

    custom_provider_map = {}
    for entry in get_compatible_custom_providers(cfg):
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        base_url = (entry.get("base_url") or "").strip()
        if not name or not base_url:
            continue
        provider_key = (entry.get("provider_key") or "").strip()
        model = entry.get("model", "")
        custom_provider_map[custom_provider_slug(name, provider_key)] = {
            "name": name,
            "base_url": base_url,
            "api_key": entry.get("api_key", ""),
            "key_env": entry.get("key_env") or entry.get("api_key_env", ""),
            "model": model,
            "models": entry.get("models", {}),
            "models_discovered": entry.get("models_discovered", False),
            "extra_headers": entry.get("extra_headers", {}),
            "discover_models": entry.get("discover_models", True),
            "api_mode": entry.get("api_mode", ""),
            "provider_key": provider_key,
            "api_key_ref": _lookup_ref(raw_api_key_refs, name, provider_key, model),
            "base_url_ref": _lookup_ref(raw_base_url_refs, name, provider_key, model)}
    return custom_provider_map


def _build_provider_picker_rows(config: dict, active: str, provider_labels: dict[str, str],
                                custom_provider_map: dict[str, dict[str, str]]) -> tuple[list[tuple[str, str, list[str]]], int]:
    """Rows for the ``hermes model`` provider picker plus the pre-selected index. Canonical providers
    fold into display groups (PROVIDER_GROUPS): a group row's ``members`` drive a sub-picker, leaf
    rows have ``members == []``; saved custom providers and trailing actions stay flat. Honors
    ``model_catalog.excluded_providers`` (slug or alias, case-insensitive) like the gateway/TUI."""
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_ALIASES
    from hermes_cli.models_catalog_static import group_providers, provider_group_for_slug
    canonical_descs = {p.slug: p.tui_desc for p in CANONICAL_PROVIDERS}
    _cli_excluded = {
        str(p).strip().lower()
        for p in (config.get("model_catalog", {}) or {}).get("excluded_providers") or []
        if p}
    if _cli_excluded:
        # A canonical provider is hidden if its slug OR any alias is excluded.
        _names_for: dict[str, set[str]] = {_p.slug: {_p.slug.lower()} for _p in CANONICAL_PROVIDERS}
        for _alias, _canon in _PROVIDER_ALIASES.items():
            _names_for.setdefault(_canon, {_canon.lower()}).add(_alias.lower())
        _visible_slugs = [p.slug for p in CANONICAL_PROVIDERS if not _names_for.get(p.slug, {p.slug.lower()}) & _cli_excluded]
    else:
        _visible_slugs = [p.slug for p in CANONICAL_PROVIDERS]

    # The active provider's group when grouped, otherwise the active slug itself.
    active_group = provider_group_for_slug(active) if active else ""

    # (key, label, members): members == [] → leaf row (provider slug / action);
    # members != [] → group row, key is "group:<gid>"
    ordered: list[tuple[str, str, list[str]]] = []
    default_idx = 0

    def _add(key, label, members, is_active):
        nonlocal default_idx
        if is_active:
            label = f"{label}  ← currently active"
            default_idx = len(ordered)
        ordered.append((key, label, members))

    for row in group_providers(_visible_slugs):
        if row["kind"] == "group":
            gid = row["group_id"]
            group_desc = row.get("description", "")
            label = f"{row['label']} ▸ ({group_desc})" if group_desc else f"{row['label']} ▸"
            _add(f"group:{gid}", label, row["members"], bool(active_group) and gid == active_group)
        else:
            slug = row["slug"]
            _add(slug, canonical_descs.get(slug, provider_labels.get(slug, slug)), [], bool(active) and slug == active)

    for key, provider_info in custom_provider_map.items():
        saved_model = provider_info.get("model", "")
        model_hint = f" — {saved_model}" if saved_model else ""
        _add(key, f"{provider_info['name']} ({_short_url(provider_info['base_url'])}){model_hint}", [],
             bool(active) and key == active)

    ordered.append(("custom", "Custom endpoint (enter URL manually)", []))
    if isinstance(config.get("custom_providers"), list) and config.get("custom_providers"):
        ordered.append(("remove-custom", "Remove a saved custom provider", []))
    ordered.append(("aux-config", "Configure auxiliary models...", []))
    ordered.append(("cancel", "Leave unchanged", []))
    return ordered, default_idx
