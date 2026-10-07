"""Credential sections of `rabbit status`, run through ``status._SECTIONS`` with its shared context.
Origin helpers (``_row``, ``_first_env_value``, ...) are resolved through the ``rabbit_cli.status``
module object so tests that monkeypatch that module keep working."""

from datetime import datetime, timezone

from rabbit_cli.auth import AuthError
from rabbit_cli import config
from rabbit_time import safe_strftime


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return "(unknown)"
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except Exception:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return safe_strftime(parsed.astimezone(), "%Y-%m-%d %H:%M:%S %Z")


def _qwen_expiry(expires_at_ms) -> str:
    return datetime.fromtimestamp(int(expires_at_ms) / 1000, tz=timezone.utc).isoformat()


def _oauth_block(name: str, status: dict, hint: str, rows) -> None:
    """Print an OAuth provider row plus its conditional detail lines.

    ``rows`` are ``(label, status_key, formatter, gate)``: a detail prints when the raw value is
    truthy and ``gate`` is None or equals the logged-in state (False = only while logged out).
    """
    logged_in = bool(status.get("logged_in"))
    _status._row(name, logged_in, "logged in" if logged_in else f"not logged in (run: {hint})")
    for label, key, fmt, gate in rows:
        raw = status.get(key)
        if raw and (gate is None or gate == logged_in):
            _status._detail(label, fmt(raw) if fmt else raw)


# Values may be a single env var name (str) or a tuple of alternates (first found wins).
_API_KEYS: dict[str, str | tuple[str, ...]] = {
    "OpenRouter": "OPENROUTER_API_KEY", "OpenAI": "OPENAI_API_KEY",
    "Google / Gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "DeepSeek": "DEEPSEEK_API_KEY",
    "xAI / Grok": "XAI_API_KEY", "NVIDIA NIM": "NVIDIA_API_KEY", "Z.AI / GLM": "GLM_API_KEY",
    "Kimi": "KIMI_API_KEY", "StepFun Step Plan": "STEPFUN_API_KEY", "MiniMax": "MINIMAX_API_KEY",
    "MiniMax-CN": "MINIMAX_CN_API_KEY", "DeepInfra": "DEEPINFRA_API_KEY", "Firecrawl": "FIRECRAWL_API_KEY",
    "Tavily": "TAVILY_API_KEY", "Perplexity": "PERPLEXITY_API_KEY", "Keenable": "KEENABLE_API_KEY",
    "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
    "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
    "FAL": "FAL_KEY", "ElevenLabs": "ELEVENLABS_API_KEY", "GitHub": "GITHUB_TOKEN"}

# OAuth detail rows: (label, status key, formatter, gate) — see _oauth_block.
_FILE_REFRESH_ROWS = (
    ("Auth file:", "auth_store", None, None),
    ("Refreshed:", "last_refresh", _format_iso_timestamp, None), ("Error:", "error", None, False))

_OAUTH_BLOCKS = (
    # (row name, auth getter, login hint, detail rows)
    ("OpenAI Codex", "get_codex_auth_status", "rabbit model", _FILE_REFRESH_ROWS),
    ("Qwen OAuth", "get_qwen_auth_status", "qwen auth qwen-oauth", (
        ("Auth file:", "auth_file", None, None),
        ("Access exp:", "expires_at_ms", _qwen_expiry, None),
        ("Error:", "error", None, False))),
    ("MiniMax OAuth", "get_minimax_oauth_auth_status", "rabbit auth add minimax-oauth", (
        ("Region:", "region", None, True),
        ("Access exp:", "expires_at", None, None),
        ("Error:", "error", None, False))),
    ("xAI OAuth", "get_xai_oauth_auth_status", "rabbit auth add xai-oauth", _FILE_REFRESH_ROWS))

_APIKEY_PROVIDERS = {
    "Z.AI / GLM": ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "Kimi / Moonshot": ("KIMI_API_KEY",),
    "StepFun Step Plan": ("STEPFUN_API_KEY",), "MiniMax": ("MINIMAX_API_KEY",),
    "MiniMax (China)": ("MINIMAX_CN_API_KEY",), "DeepInfra": ("DEEPINFRA_API_KEY",)}


def _render_api_keys(ctx):
    _status._section("API Keys")
    from rabbit_cli.auth import get_anthropic_key
    # Anthropic uses the dedicated lookup (it also resolves OAuth tokens).
    for name, env_ref in (*_API_KEYS.items(), ("Anthropic", get_anthropic_key)):
        value = env_ref() if callable(env_ref) else _status._first_env_value(env_ref)
        _status._row(name, bool(value), config.redact_key(value))


def _render_auth_providers(ctx):
    _status._section("Auth Providers")
    import rabbit_cli.auth as auth
    try:
        # Read-only display: the refresh-free snapshot, so `rabbit status` never performs an OAuth
        # refresh or burns a single-use refresh token.
        statuses = {getter: getattr(auth, getter)() for _, getter, _, _ in _OAUTH_BLOCKS[:3]}
    except Exception:
        statuses = {}
    # xAI OAuth is guarded separately so an import failure there cannot disrupt the other rows.
    try:
        statuses["get_xai_oauth_auth_status"] = auth.get_xai_oauth_auth_status() or {}
    except Exception:
        statuses["get_xai_oauth_auth_status"] = {}

    for name, getter, hint, rows in _OAUTH_BLOCKS:
        _oauth_block(name, statuses.get(getter, {}), hint, rows)


def _render_apikey_providers(ctx):
    _status._section("API-Key Providers")
    for pname, env_vars in _APIKEY_PROVIDERS.items():
        configured = bool(_status._first_env_value(env_vars))
        _status._row(pname, configured, "configured" if configured else "not configured (run: rabbit model)", 16, " ")

    # LM Studio reachability: probe only when it is the active provider so users with foreign
    # configs see no noise. Auth rejection vs. a silent empty list is the common support case.
    if _status._effective_provider_label() == "LM Studio":
        from rabbit_cli.models_local import probe_lmstudio_models
        model_cfg = ctx.config.get("model")
        base = ((model_cfg.get("base_url") if isinstance(model_cfg, dict) else None)
                or _status.get_env_value("LM_BASE_URL") or "http://127.0.0.1:1234/v1")
        try:
            models = probe_lmstudio_models(api_key=_status.get_env_value("LM_API_KEY") or "",
                                           base_url=base, timeout=1.5)
            ok = models is not None
            msg = f"reachable ({len(models)} model(s)) at {base}" if ok else f"unreachable at {base}"
        except AuthError:
            ok, msg = False, "auth rejected — set LM_API_KEY"
        _status._row("LM Studio", ok, msg, 16, " ")


import rabbit_cli.status as _status  # noqa: E402  (bottom: rabbit_cli.status imports this module)
