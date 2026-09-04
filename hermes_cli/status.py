"""Status command for hermes CLI."""

import json
import os
import sys
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.auth import AuthError, resolve_provider
from hermes_cli.colors import Colors, color
from hermes_cli.config import get_env_path, get_env_value, get_hermes_home, load_config
from hermes_cli.models import provider_label
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_cli.status_auth import (  # renderers wired into _SECTIONS below
    _render_api_keys, _render_apikey_providers, _render_auth_providers, _render_nous_gateway)
from hermes_constants import OPENROUTER_MODELS_URL
from hermes_constants import is_termux as _is_termux


def check_mark(ok: bool) -> str:
    return color("✓", Colors.GREEN) if ok else color("✗", Colors.RED)


def _section(title: str) -> None:
    """Print a blank line followed by a bold cyan ``◆`` section heading."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _row(name: str, ok: bool, text: str, width: int = 12, sep: str = "  ") -> None:
    """Print one ``name  ✓/✗ text`` status row."""
    print(f"  {name:<{width}}{sep}{check_mark(ok)} {text}")


def _detail(label: str, value) -> None:
    """Print an indented ``label: value`` detail line under a status row."""
    _kv(label, value, "    ", 12)


def _kv(label: str, value, indent: str = "  ", width: int = 14) -> None:
    """Print a ``  Label:        value`` line (label padded to the 14-col status layout)."""
    print(f"{indent}{label:<{width}}{value}")


def _kv_flag(label: str, ok, on: str, off: str) -> None:
    """``_kv`` with a ✓/✗ mark followed by ``on`` or ``off`` text."""
    _kv(label, f"{check_mark(bool(ok))} {on if ok else off}")


def _configured(ok) -> str:
    return "configured" if ok else "not configured"


def _first_env_value(names) -> str:
    """Return the first non-empty env value among ``names`` (a str or tuple of names)."""
    return next((v for v in (get_env_value(n) or "" for n in ((names,) if isinstance(names, str) else names)) if v), "")


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_cfg = model_cfg.get("default") or model_cfg.get("name") or ""
    return (model_cfg.strip() if isinstance(model_cfg, str) else "") or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter":
        # A custom endpoint may live in config.yaml (model.base_url, the canonical location) or
        # the legacy OPENAI_BASE_URL env var; either way labeling it "OpenRouter" is misleading.
        try:
            model_cfg = load_config().get("model")
        except Exception:
            model_cfg = None
        config_base_url = (model_cfg.get("base_url") or "").strip() if isinstance(model_cfg, dict) else ""
        if config_base_url or get_env_value("OPENAI_BASE_URL"):
            effective = "custom"
    return provider_label(effective)


def _estop_status_line():
    """One-line pause banner for `hermes status`, or None when not paused."""
    try:
        from agent.estop import get_state
    except ImportError:
        return None
    state = get_state()
    if state is None:
        return None
    reason = state.get("reason")
    return f"⏸️  PAUSED (global emergency stop{f' — reason: {reason}' if reason else ''}; `hermes resume` to lift)"


# --- Data tables driving the per-section renderers -------------------------

# Simple env-driven terminal backends: (label, env var, default, empty-counts-as-unset).
_TERMINAL_ENV_ROWS = {
    "ssh": (("SSH Host:", "TERMINAL_SSH_HOST", "(not set)", True), ("SSH User:", "TERMINAL_SSH_USER", "(not set)", True)),
    "docker": (("Docker Image:", "TERMINAL_DOCKER_IMAGE", "python:3.11-slim", False),),
    "daytona": (("Daytona Image:", "TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20", False),),
}

_PLATFORMS = {  # name -> (token env var, home-channel env var or None)
    "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
    "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"), "WhatsApp": ("WHATSAPP_ENABLED", None),
    "Signal": ("SIGNAL_HTTP_URL", "SIGNAL_HOME_CHANNEL"),
    "Slack": ("SLACK_BOT_TOKEN", None), "Email": ("EMAIL_ADDRESS", "EMAIL_HOME_ADDRESS"),
    "SMS": ("TWILIO_ACCOUNT_SID", "SMS_HOME_CHANNEL"), "DingTalk": ("DINGTALK_CLIENT_ID", None),
    "Feishu": ("FEISHU_APP_ID", "FEISHU_HOME_CHANNEL"), "WeCom": ("WECOM_BOT_ID", "WECOM_HOME_CHANNEL"),
    "WeCom Callback": ("WECOM_CALLBACK_CORP_ID", None), "Weixin": ("WEIXIN_ACCOUNT_ID", "WEIXIN_HOME_CHANNEL"),
    "BlueBubbles": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_HOME_CHANNEL"), "QQBot": ("QQ_APP_ID", "QQ_HOME_CHANNEL"),
    "Yuanbao": ("YUANBAO_APP_ID", "YUANBAO_HOME_CHANNEL")}

# Gateway manager label when the runtime snapshot is unavailable, keyed by platform.
_GATEWAY_FALLBACK = {"termux": ("unknown", "Termux / manual process"), "linux": ("unknown", "systemd/manual"),
                     "darwin": ("unknown", "launchd")}


def _banner(lines, *styles) -> None:
    """Blank line, then each line in ``styles``."""
    print()
    for line in lines:
        print(color(line, *styles))


def _render_header(ctx):
    _banner(("┌─────────────────────────────────────────────────────────┐",
             "│                 ☤ Hermes Agent Status                  │",
             "└─────────────────────────────────────────────────────────┘"), Colors.CYAN)
    paused = _estop_status_line()
    if paused:
        _banner((paused,), Colors.YELLOW, Colors.BOLD)


def _render_environment(ctx):
    _section("Environment")
    _kv("Project:", PROJECT_ROOT)
    _kv("Python:", sys.version.split()[0])
    _kv_flag(".env file:", get_env_path().exists(), "exists", "not found")
    try:
        ctx.config = load_config()
    except Exception:
        ctx.config = {}
    _kv("Model:", _configured_model_label(ctx.config))
    _kv("Provider:", _effective_provider_label())


    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))

    # Values may be a single env var name (str) or a tuple of alternates (first found wins).
    keys: dict[str, str | tuple[str, ...]] = {
        "OpenRouter": "OPENROUTER_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
        "Google / Gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "DeepSeek": "DEEPSEEK_API_KEY",
        "xAI / Grok": "XAI_API_KEY",
        "NVIDIA NIM": "NVIDIA_API_KEY",
        "Z.AI / GLM": "GLM_API_KEY",
        "Kimi": "KIMI_API_KEY",
        "StepFun Step Plan": "STEPFUN_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "MiniMax-CN": "MINIMAX_CN_API_KEY",
        "DeepInfra": "DEEPINFRA_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Tavily": "TAVILY_API_KEY",
        "Keenable": "KEENABLE_API_KEY",
        "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
        "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
        "FAL": "FAL_KEY",
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "GitHub": "GITHUB_TOKEN",
    }

    def _resolve_env(env_ref) -> str:
        """Return first non-empty env var value from a str or tuple of names."""
        if isinstance(env_ref, tuple):
            for candidate in env_ref:
                v = get_env_value(candidate) or ""
                if v:
                    return v
            return ""
        return get_env_value(env_ref) or ""

    for name, env_ref in keys.items():
        # Anthropic already has a dedicated lookup below; keep that as the
        # single source of truth (it also resolves OAuth tokens), skip here
        # so we don't print two "Anthropic" rows.
        if name == "Anthropic":
            continue
        value = _resolve_env(env_ref)
        has_key = bool(value)
        display = redact_key(value)
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    from hermes_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    anthropic_display = redact_key(anthropic_value)
    print(f"  {'Anthropic':<12}  {check_mark(bool(anthropic_value))} {anthropic_display}")

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.auth import (
            get_nous_auth_status_local,
            get_codex_auth_status,
            get_qwen_auth_status,
            get_minimax_oauth_auth_status,
        )
        # Read-only display: use the refresh-free snapshot so `hermes status`
        # never performs an OAuth refresh or burns a single-use refresh token.
        nous_status = get_nous_auth_status_local()
        codex_status = get_codex_auth_status()
        qwen_status = get_qwen_auth_status()
        minimax_status = get_minimax_oauth_auth_status()
    except Exception:
        nous_status = {}
        codex_status = {}
        qwen_status = {}
        minimax_status = {}

    nous_account_info = None
    if (
        nous_status.get("logged_in")
        or nous_status.get("access_token")
        or nous_status.get("portal_base_url")
        or nous_status.get("inference_credential_present")
        or nous_status.get("error_code")
    ):
        try:
            nous_account_info = get_nous_portal_account_info()
        except Exception:
            nous_account_info = None

    nous_logged_in = bool(
        nous_status.get("logged_in")
        or (nous_account_info and nous_account_info.logged_in)
    )
    nous_inference_present = bool(
        nous_status.get("inference_credential_present")
        or (nous_account_info and nous_account_info.inference_credential_present)
    )
    nous_error = nous_status.get("error")
    if nous_logged_in:
        nous_label = "logged in"
    elif nous_inference_present:
        nous_label = "not logged in (Nous inference key configured)"
    else:
        nous_label = "not logged in (run: hermes portal)"
    print(
        f"  {'Nous Portal':<12}  {check_mark(nous_logged_in)} "
        f"{nous_label}"
    )
    portal_url = nous_status.get("portal_base_url") or "(unknown)"
    inference_url = (
        nous_status.get("inference_base_url")
        or (nous_account_info.inference_base_url if nous_account_info else None)
    )
    access_exp = _format_iso_timestamp(nous_status.get("access_expires_at"))
    key_exp = _format_iso_timestamp(nous_status.get("agent_key_expires_at"))
    refresh_label = "yes" if nous_status.get("has_refresh_token") else "no"
    if nous_logged_in or portal_url != "(unknown)" or nous_error:
        print(f"    Portal URL: {portal_url}")
    if nous_inference_present and inference_url:
        print(f"    Inference:  {inference_url}")
    if nous_logged_in or nous_status.get("access_expires_at"):
        print(f"    Access exp: {access_exp}")
    if nous_logged_in or nous_inference_present or nous_status.get("agent_key_expires_at"):
        print(f"    Key exp:    {key_exp}")
    if nous_logged_in or nous_status.get("has_refresh_token"):
        print(f"    Refresh:    {refresh_label}")
    if nous_error:
        print(f"    Error:      {nous_error}")

    codex_logged_in = bool(codex_status.get("logged_in"))
    print(
        f"  {'OpenAI Codex':<12}  {check_mark(codex_logged_in)} "
        f"{'logged in' if codex_logged_in else 'not logged in (run: hermes model)'}"
    )
    codex_auth_file = codex_status.get("auth_store")
    if codex_auth_file:
        print(f"    Auth file:  {codex_auth_file}")
    codex_last_refresh = _format_iso_timestamp(codex_status.get("last_refresh"))
    if codex_status.get("last_refresh"):
        print(f"    Refreshed:  {codex_last_refresh}")
    if codex_status.get("error") and not codex_logged_in:
        print(f"    Error:      {codex_status.get('error')}")

    qwen_logged_in = bool(qwen_status.get("logged_in"))
    print(
        f"  {'Qwen OAuth':<12}  {check_mark(qwen_logged_in)} "
        f"{'logged in' if qwen_logged_in else 'not logged in (run: qwen auth qwen-oauth)'}"
    )
    qwen_auth_file = qwen_status.get("auth_file")
    if qwen_auth_file:
        print(f"    Auth file:  {qwen_auth_file}")
    qwen_exp = qwen_status.get("expires_at_ms")
    if qwen_exp:
        from datetime import datetime, timezone
        print(f"    Access exp: {datetime.fromtimestamp(int(qwen_exp) / 1000, tz=timezone.utc).isoformat()}")
    if qwen_status.get("error") and not qwen_logged_in:
        print(f"    Error:      {qwen_status.get('error')}")

    minimax_logged_in = bool(minimax_status.get("logged_in"))
    print(
        f"  {'MiniMax OAuth':<12}  {check_mark(minimax_logged_in)} "
        f"{'logged in' if minimax_logged_in else 'not logged in (run: hermes auth add minimax-oauth)'}"
    )
    minimax_region = minimax_status.get("region")
    if minimax_logged_in and minimax_region:
        print(f"    Region:     {minimax_region}")
    minimax_exp = minimax_status.get("expires_at")
    if minimax_exp:
        print(f"    Access exp: {minimax_exp}")
    if minimax_status.get("error") and not minimax_logged_in:
        print(f"    Error:      {minimax_status.get('error')}")

    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the already-printed Nous/Codex/Qwen/MiniMax rows above.
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        xai_oauth_status = get_xai_oauth_auth_status() or {}
    except Exception:
        xai_oauth_status = {}

    xai_oauth_logged_in = bool(xai_oauth_status.get("logged_in"))
    print(
        f"  {'xAI OAuth':<12}  {check_mark(xai_oauth_logged_in)} "
        f"{'logged in' if xai_oauth_logged_in else 'not logged in (run: hermes auth add xai-oauth)'}"
    )
    xai_auth_file = xai_oauth_status.get("auth_store")
    if xai_auth_file:
        print(f"    Auth file:  {xai_auth_file}")
    if xai_oauth_status.get("last_refresh"):
        print(f"    Refreshed:  {_format_iso_timestamp(xai_oauth_status.get('last_refresh'))}")
    if xai_oauth_status.get("error") and not xai_oauth_logged_in:
        print(f"    Error:      {xai_oauth_status.get('error')}")

    # =========================================================================
    # Nous Subscription Features
    # =========================================================================
    if managed_nous_tools_enabled():
        features = get_nous_subscription_features(config)
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        if not features.nous_auth_present:
            print("  Nous Portal   ✗ not logged in")
        else:
            print("  Nous Portal   ✓ managed tools available")
        for feature in features.items():
            if feature.managed_by_nous:
                state = "active via Nous subscription"
            elif feature.active:
                current = feature.current_provider or "configured provider"
                state = f"active via {current}"
            elif feature.included_by_default and features.nous_auth_present:
                state = "included by subscription, not currently selected"
            elif feature.key == "modal" and features.nous_auth_present:
                state = "available via subscription (optional)"
            else:
                state = "not configured"
            print(f"  {feature.label:<15} {check_mark(feature.available or feature.active or feature.managed_by_nous)} {state}")
    elif nous_logged_in or nous_inference_present:
        # Nous OAuth without entitlement, or an opaque inference key without
        # Portal account information, cannot enable the Tool Gateway.
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        message = format_nous_portal_entitlement_message(
            nous_account_info,
            capability="managed web, image, TTS, STT, browser, and Modal tools",
        )
        if message:
            for line in message.splitlines():
                print(f"  {line}")

    # =========================================================================
    # API-Key Providers
    # =========================================================================
    print()
    print(color("◆ API-Key Providers", Colors.CYAN, Colors.BOLD))

    apikey_providers = {
        "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "Kimi / Moonshot":  ("KIMI_API_KEY",),
        "StepFun Step Plan": ("STEPFUN_API_KEY",),
        "MiniMax":          ("MINIMAX_API_KEY",),
        "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
        "DeepInfra":        ("DEEPINFRA_API_KEY",),
    }
    for pname, env_vars in apikey_providers.items():
        key_val = ""
        for ev in env_vars:
            key_val = get_env_value(ev) or ""
            if key_val:
                break
        configured = bool(key_val)
        label = "configured" if configured else "not configured (run: hermes model)"
        print(f"  {pname:<16} {check_mark(configured)} {label}")

    # LM Studio reachability — only probe when it's the active provider so
    # users with foreign configs don't see noise. Auth rejection vs. silent
    # empty list is the most common LM Studio support case.
    if _effective_provider_label() == "LM Studio":
        from hermes_cli.models import probe_lmstudio_models
        model_cfg = config.get("model")
        base = (model_cfg.get("base_url") if isinstance(model_cfg, dict) else None) or get_env_value("LM_BASE_URL") or "http://127.0.0.1:1234/v1"
        try:
            models = probe_lmstudio_models(api_key=get_env_value("LM_API_KEY") or "", base_url=base, timeout=1.5)
            if models is None:
                ok, msg = False, f"unreachable at {base}"
            else:
                ok, msg = True, f"reachable ({len(models)} model(s)) at {base}"
        except AuthError:
            ok, msg = False, "auth rejected — set LM_API_KEY"
        print(f"  {'LM Studio':<16} {check_mark(ok)} {msg}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))

    terminal_cfg = config.get("terminal", {}) if isinstance(config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        terminal_env = terminal_cfg.get("backend", "local")
    print(f"  Backend:      {terminal_env}")

    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    elif terminal_env == "daytona":
        daytona_image = os.getenv("TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20")
        print(f"  Daytona Image: {daytona_image}")
    elif terminal_env == "vercel_sandbox":
        persist = os.getenv("TERMINAL_CONTAINER_PERSISTENT")
        persist_enabled = (bool(terminal_cfg.get("container_persistent", True)) if persist is None
                           else persist.lower() in {"1", "true", "yes", "on"})
        auth_status = describe_vercel_auth()
        _kv("Runtime:", os.getenv('TERMINAL_VERCEL_RUNTIME') or terminal_cfg.get('vercel_runtime') or 'node24')
        _kv_flag("SDK:", importlib.util.find_spec("vercel") is not None, "installed",
                 "missing (install: pip install 'hermes-agent[vercel]')")
        _kv("Auth:", f"{check_mark(auth_status.ok)} {auth_status.label}")
        for line in auth_status.detail_lines:
            print(f"  Auth detail:  {line}")
        print(f"  Persistence:  {'snapshot filesystem' if persist_enabled else 'ephemeral filesystem'}")
        print("  Processes:    live processes do not survive cleanup, snapshots, or sandbox recreation")
    else:
        # Plugin-registered terminal backends: show availability via the
        # provider's doctor rows (fail-soft — never break `hermes status`).
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            from agent.terminal_env_registry import get_provider

            _provider = get_provider(terminal_env)
            if _provider is not None:
                for _ok, _label, _detail in _provider.doctor_checks():
                    print(f"  {_label}: {check_mark(bool(_ok))} {_detail}")
        except Exception:
            pass


def _render_platforms(ctx):
    _section("Messaging Platforms")
    for name, (token_var, home_var) in _PLATFORMS.items():
        has_token = bool(os.getenv(token_var, ""))
        home_channel = os.getenv(home_var, "") if home_var else ""
        _row(name, has_token, _configured(has_token) + (f" (home: {home_channel})" if home_channel else ""))

    try:  # Plugin-registered platforms
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            # Per-entry guard: one raising probe must not abort the listing of every remaining
            # plugin platform (matches the other check_fn sites).
            try:
                configured = bool(entry.check_fn())
            except Exception:
                configured = False
            _row(entry.label, configured, f"{_configured(configured)} (plugin)")
    except Exception:
        pass


def _render_gateway(ctx):
    _section("Gateway Service")
    try:
        from hermes_cli.gateway import (
            get_gateway_runtime_snapshot, _format_gateway_pids, named_profile_served_by_running_multiplexer)
        from hermes_cli.gateway_multiplex_served import multiplexer_served_secondaries
        snapshot = get_gateway_runtime_snapshot()
        # A satellite profile has no gateway.pid of its own; the default multiplexer is its live process.
        if not snapshot.running and named_profile_served_by_running_multiplexer():
            _kv_flag("Status:", True, "running (via the default-profile multiplexer)", "stopped")
            _kv("Manage with:", "hermes gateway status   # from the default profile")
            return
        _kv_flag("Status:", snapshot.running, "running", "stopped")
        _kv("Manager:", snapshot.manager)
        if snapshot.gateway_pids:
            _kv("PID(s):", _format_gateway_pids(snapshot.gateway_pids))
        if snapshot.running and (served := multiplexer_served_secondaries()):
            _kv("Serves:", ", ".join(served))
            from hermes_cli.gateway_multiplex_served import served_profile_ingress_urls
            for name, per_platform in sorted(served_profile_ingress_urls().items()):
                for platform, url in sorted(per_platform.items()):
                    _kv(f"  {name}/{platform}:", url)
        if snapshot.has_process_service_mismatch:
            _kv("Service:", "installed but not managing the current running gateway")
        elif _is_termux() and not snapshot.gateway_pids:
            _kv("Start with:", "hermes gateway")
            _kv("Note:", "Android may stop background jobs when Termux is suspended")
        elif snapshot.service_installed and not snapshot.service_running:
            _kv("Service:", "installed but stopped")
    except Exception:
        platform = "termux" if _is_termux() else "linux" if sys.platform.startswith("linux") else sys.platform
        status_text, manager = _GATEWAY_FALLBACK.get(platform, ("N/A", "(not supported on this platform)"))
        _kv("Status:", color(status_text, Colors.DIM))
        _kv("Manager:", manager)


def _load_json(path: Path, encoding: str = "utf-8"):
    with open(path, encoding=encoding) as f:
        return json.load(f)


def _render_cron(ctx):
    _section("Scheduled Jobs")
    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    if not jobs_file.exists():
        _kv("Jobs:", 0)
        return
    try:
        # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors may leave a UTF-8 BOM
        # that plain utf-8 json.load rejects.
        jobs = _load_json(jobs_file, "utf-8-sig").get("jobs", [])
        _kv("Jobs:", f"{sum(1 for j in jobs if j.get('enabled', True))} active, {len(jobs)} total")
    except Exception:
        _kv("Jobs:", "(error reading jobs file)")


def _render_sessions(ctx):
    _section("Sessions")
    # Gateway session count: state.db is the source of truth; fall back to sessions.json for
    # pre-migration installs.
    try:
        from hermes_state import SessionDB
        db = SessionDB(read_only=True)  # status only reads; never a writer beside a running gateway
        try:
            gateway_rows = db.list_gateway_sessions(active_only=True) or []
        finally:
            db.close()
    except Exception:
        gateway_rows = []

    if gateway_rows:
        _kv("Active:", f"{len(gateway_rows)} session(s)")
        freshest = max((float(r.get("last_active") or 0) for r in gateway_rows), default=0.0)
        if freshest > 0:
            from hermes_cli.timefmt import relative_time
            print(f"  Last activity:{relative_time(freshest):>13}")
    elif not (sessions_file := get_hermes_home() / "sessions" / "sessions.json").exists():
        _kv("Active:", 0)
    else:
        try:
            data = _load_json(sessions_file)
            entries = [k for k in data if not str(k).startswith("_")] if isinstance(data, dict) else []
            _kv("Active:", f"{len(entries)} session(s)")
        except Exception:
            _kv("Active:", "(error reading sessions file)")

    # Slot usage, only when max_concurrent_sessions is set. The cap is shared across CLI,
    # desktop/TUI and the messaging gateway, so the surface that gets rejected is rarely the one
    # holding the slots — without this the only way to find out is reading
    # runtime/active_sessions.json by hand.
    try:
        from hermes_cli.active_sessions import (
            active_session_registry_snapshot, format_age, resolve_max_concurrent_sessions)
        cap = resolve_max_concurrent_sessions(ctx.config)
    except Exception:
        cap = None
    if cap:
        try:
            held = active_session_registry_snapshot()
        except Exception:
            held = []
        _kv("Slots:", color(f"{len(held)}/{cap} in use", Colors.YELLOW if len(held) >= cap else Colors.GREEN))
        now = time.time()
        for entry in sorted(held, key=lambda e: e.get("started_at") or 0):
            age = format_age(now - float(entry.get("started_at") or now))
            print(f"                {entry.get('surface') or 'unknown':<17} {entry.get('session_id') or '?':<24} {age}")


def _render_deep(ctx):
    if not ctx.deep:
        return
    _section("Deep Checks")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        try:
            import httpx
            response = httpx.get(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {openrouter_key}"}, timeout=10)
            _kv_flag("OpenRouter:", response.status_code == 200, "reachable", f"error ({response.status_code})")
        except Exception as e:
            _kv("OpenRouter:", f"{check_mark(False)} error: {e}")
    try:  # gateway port, informational: in use == gateway likely running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        port_in_use = sock.connect_ex(('127.0.0.1', 18789)) == 0
        sock.close()
        _kv("Port 18789:", 'in use' if port_in_use else 'available')
    except OSError:
        pass


def _render_footer(ctx):
    _banner(("─" * 60, "  Run 'hermes doctor' for detailed diagnostics", "  Run 'hermes setup' to configure"),
            Colors.DIM)
    print()


# Print order of `hermes status`; each renderer takes the shared _StatusContext.
_SECTIONS = (
    _render_header, _render_environment, _render_api_keys, _render_auth_providers, _render_nous_gateway,
    _render_apikey_providers, _render_terminal, _render_platforms, _render_gateway, _render_cron,
    _render_sessions, _render_deep, _render_footer)


def show_status(args):
    """Show status of all Hermes Agent components."""
    # Shared by section renderers: config, --deep, and the Nous login facts Auth Providers derives
    # for the later Nous Tool Gateway section.
    ctx = SimpleNamespace(deep=getattr(args, 'deep', False), config={}, nous_logged_in=False,
                          nous_inference_present=False, nous_account_info=None)
    for render in _SECTIONS:
        render(ctx)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import subprocess  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'format_nous_portal_entitlement_message': ('hermes_cli.nous_account', 'format_nous_portal_entitlement_message'),
    'get_nous_portal_account_info': ('hermes_cli.nous_account', 'get_nous_portal_account_info'),
    'get_nous_subscription_features': ('hermes_cli.nous_subscription', 'get_nous_subscription_features'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'redact_key': ('hermes_cli.config', 'redact_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
