"""
Status command for hermes CLI.

Shows the status of all Hermes Agent components.
"""

import os
import sys
import subprocess  # noqa: F401 — re-exported for tests that monkeypatch status.subprocess to guard against regressions
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from agent.i18n import pluralize, t
from hermes_cli.auth import AuthError, resolve_provider
from hermes_cli.colors import Colors, color
from hermes_cli.config import get_env_path, get_env_value, get_hermes_home, load_config
from hermes_cli.models import provider_label
from hermes_cli.nous_subscription import get_nous_subscription_features
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_constants import OPENROUTER_MODELS_URL
from tools.tool_backend_helpers import managed_nous_tools_enabled


def _status_t(key: str, default: str, **kwargs) -> str:
    return t(f"status.{key}", default=default, **kwargs)


def _status_count(count: int, english_one: str, english_many: str, russian_one: str, russian_few: str, russian_many: str) -> str:
    from agent.i18n import get_language

    if get_language() == "ru":
        noun = pluralize(count, russian_one, russian_few, russian_many, language="ru")
    else:
        noun = english_one if abs(int(count)) == 1 else english_many
    return f"{count} {noun}"

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display.

    Thin wrapper over :func:`agent.redact.mask_secret`. Preserves the
    "(not set)" placeholder in dim color to match ``hermes config``'s
    output (previously this variant was missing the DIM color —
    consolidated via PR that also introduced ``mask_secret``).
    """
    from agent.redact import mask_secret
    return mask_secret(key, empty=color(t("common.not_set", default="(not set)"), Colors.DIM))


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    if not value or not isinstance(value, str):
        return t("common.unknown", default="(unknown)")
    from datetime import datetime, timezone
    text = value.strip()
    if not text:
        return t("common.unknown", default="(unknown)")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model = (model_cfg.get("default") or model_cfg.get("name") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()
    else:
        model = ""
    return model or t("common.not_set", default="(not set)")


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        effective = "custom"

    return provider_label(effective)


from hermes_constants import is_termux as _is_termux


def show_status(args):
    """Show status of all Hermes Agent components."""
    show_all = getattr(args, 'all', False)
    deep = getattr(args, 'deep', False)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color(f"│                 ⚕ {_status_t('title', 'Hermes Agent Status'): <30}│", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.environment', 'Environment')}", Colors.CYAN, Colors.BOLD))
    print(f"  {_status_t('labels.project', 'Project'):<12}  {PROJECT_ROOT}")
    print(f"  {_status_t('labels.python', 'Python'):<12}  {sys.version.split()[0]}")

    env_path = get_env_path()
    print(f"  {_status_t('labels.env_file', '.env file'):<12}  {check_mark(env_path.exists())} {_status_t('messages.exists', 'exists') if env_path.exists() else _status_t('messages.not_found', 'not found')}")

    try:
        config = load_config()
    except Exception:
        config = {}

    print(f"  {_status_t('labels.model', 'Model'):<12}  {_configured_model_label(config)}")
    print(f"  {_status_t('labels.provider', 'Provider'):<12}  {_effective_provider_label()}")

    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.api_keys', 'API Keys')}", Colors.CYAN, Colors.BOLD))

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
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Tavily": "TAVILY_API_KEY",
        "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
        "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
        "FAL": "FAL_KEY",
        "Tinker": "TINKER_API_KEY",
        "WandB": "WANDB_API_KEY",
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
        display = redact_key(value) if not show_all else value
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    from hermes_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    anthropic_display = redact_key(anthropic_value) if not show_all else anthropic_value
    print(f"  {'Anthropic':<12}  {check_mark(bool(anthropic_value))} {anthropic_display}")

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.auth_providers', 'Auth Providers')}", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.auth import (
            get_nous_auth_status,
            get_codex_auth_status,
            get_qwen_auth_status,
            get_minimax_oauth_auth_status,
        )
        nous_status = get_nous_auth_status()
        codex_status = get_codex_auth_status()
        qwen_status = get_qwen_auth_status()
        minimax_status = get_minimax_oauth_auth_status()
    except Exception:
        nous_status = {}
        codex_status = {}
        qwen_status = {}
        minimax_status = {}

    nous_logged_in = bool(nous_status.get("logged_in"))
    nous_error = nous_status.get("error")
    nous_label = _status_t("messages.logged_in", "logged in") if nous_logged_in else _status_t("messages.not_logged_in_nous", "not logged in (run: hermes auth add nous --type oauth)")
    print(
        f"  {'Nous Portal':<12}  {check_mark(nous_logged_in)} "
        f"{nous_label}"
    )
    portal_url = nous_status.get("portal_base_url") or t("common.unknown", default="(unknown)")
    access_exp = _format_iso_timestamp(nous_status.get("access_expires_at"))
    key_exp = _format_iso_timestamp(nous_status.get("agent_key_expires_at"))
    refresh_label = t("common.yes", default="yes") if nous_status.get("has_refresh_token") else t("common.no", default="no")
    unknown_label = t("common.unknown", default="(unknown)")
    if nous_logged_in or portal_url != unknown_label or nous_error:
        print(f"    {_status_t('labels.portal_url', 'Portal URL')}: {portal_url}")
    if nous_logged_in or nous_status.get("access_expires_at"):
        print(f"    {_status_t('labels.access_exp', 'Access exp')}: {access_exp}")
    if nous_logged_in or nous_status.get("agent_key_expires_at"):
        print(f"    {_status_t('labels.key_exp', 'Key exp')}:    {key_exp}")
    if nous_logged_in or nous_status.get("has_refresh_token"):
        print(f"    {_status_t('labels.refresh', 'Refresh')}:    {refresh_label}")
    if nous_error and not nous_logged_in:
        print(f"    {_status_t('labels.error', 'Error')}:      {nous_error}")

    codex_logged_in = bool(codex_status.get("logged_in"))
    print(
        f"  {'OpenAI Codex':<12}  {check_mark(codex_logged_in)} "
        f"{_status_t('messages.logged_in', 'logged in') if codex_logged_in else _status_t('messages.not_logged_in_codex', 'not logged in (run: hermes model)')}"
    )
    codex_auth_file = codex_status.get("auth_store")
    if codex_auth_file:
        print(f"    {_status_t('labels.auth_file', 'Auth file')}:  {codex_auth_file}")
    codex_last_refresh = _format_iso_timestamp(codex_status.get("last_refresh"))
    if codex_status.get("last_refresh"):
        print(f"    {_status_t('labels.refreshed', 'Refreshed')}:  {codex_last_refresh}")
    if codex_status.get("error") and not codex_logged_in:
        print(f"    {_status_t('labels.error', 'Error')}:      {codex_status.get('error')}")

    qwen_logged_in = bool(qwen_status.get("logged_in"))
    print(
        f"  {'Qwen OAuth':<12}  {check_mark(qwen_logged_in)} "
        f"{_status_t('messages.logged_in', 'logged in') if qwen_logged_in else _status_t('messages.not_logged_in_qwen', 'not logged in (run: qwen auth qwen-oauth)')}"
    )
    qwen_auth_file = qwen_status.get("auth_file")
    if qwen_auth_file:
        print(f"    {_status_t('labels.auth_file', 'Auth file')}:  {qwen_auth_file}")
    qwen_exp = qwen_status.get("expires_at_ms")
    if qwen_exp:
        from datetime import datetime, timezone
        print(f"    {_status_t('labels.access_exp', 'Access exp')}: {datetime.fromtimestamp(int(qwen_exp) / 1000, tz=timezone.utc).isoformat()}")
    if qwen_status.get("error") and not qwen_logged_in:
        print(f"    {_status_t('labels.error', 'Error')}:      {qwen_status.get('error')}")

    minimax_logged_in = bool(minimax_status.get("logged_in"))
    print(
        f"  {'MiniMax OAuth':<12}  {check_mark(minimax_logged_in)} "
        f"{_status_t('messages.logged_in', 'logged in') if minimax_logged_in else _status_t('messages.not_logged_in_minimax', 'not logged in (run: hermes auth add minimax-oauth)')}"
    )
    minimax_region = minimax_status.get("region")
    if minimax_logged_in and minimax_region:
        print(f"    {_status_t('labels.region', 'Region')}:     {minimax_region}")
    minimax_exp = minimax_status.get("expires_at")
    if minimax_exp:
        print(f"    {_status_t('labels.access_exp', 'Access exp')}: {minimax_exp}")
    if minimax_status.get("error") and not minimax_logged_in:
        print(f"    {_status_t('labels.error', 'Error')}:      {minimax_status.get('error')}")

    # =========================================================================
    # Nous Subscription Features
    # =========================================================================
    if managed_nous_tools_enabled():
        features = get_nous_subscription_features(config)
        print()
        print(color(f"◆ {_status_t('sections.nous_tool_gateway', 'Nous Tool Gateway')}", Colors.CYAN, Colors.BOLD))
        if not features.nous_auth_present:
            print(f"  Nous Portal   ✗ {_status_t('messages.not_logged_in', 'not logged in')}")
        else:
            print(f"  Nous Portal   ✓ {_status_t('messages.managed_tools_available', 'managed tools available')}")
        for feature in features.items():
            if feature.managed_by_nous:
                state = _status_t("messages.active_via_nous", "active via Nous subscription")
            elif feature.active:
                current = feature.current_provider or "configured provider"
                state = _status_t("messages.active_via_provider", "active via {provider}", provider=current)
            elif feature.included_by_default and features.nous_auth_present:
                state = _status_t("messages.included_by_subscription", "included by subscription, not currently selected")
            elif feature.key == "modal" and features.nous_auth_present:
                state = _status_t("messages.available_via_subscription", "available via subscription (optional)")
            else:
                state = _status_t("messages.not_configured", "not configured")
            print(f"  {feature.label:<15} {check_mark(feature.available or feature.active or feature.managed_by_nous)} {state}")
    elif nous_logged_in:
        # Logged into Nous but on the free tier — show upgrade nudge
        print()
        print(color(f"◆ {_status_t('sections.nous_tool_gateway', 'Nous Tool Gateway')}", Colors.CYAN, Colors.BOLD))
        print(_status_t("messages.free_tier_notice", "  Your free-tier Nous account does not include Tool Gateway access."))
        print(_status_t("messages.upgrade_prompt", "  Upgrade your subscription to unlock managed web, image, TTS, and browser tools."))
        try:
            portal_url = nous_status.get("portal_base_url", "").rstrip("/")
            if portal_url:
                print(f"  {_status_t('messages.upgrade_link', 'Upgrade:')} {portal_url}")
        except Exception:
            pass

    # =========================================================================
    # API-Key Providers
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.api_key_providers', 'API-Key Providers')}", Colors.CYAN, Colors.BOLD))

    apikey_providers = {
        "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "Kimi / Moonshot":  ("KIMI_API_KEY",),
        "StepFun Step Plan": ("STEPFUN_API_KEY",),
        "MiniMax":          ("MINIMAX_API_KEY",),
        "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
    }
    for pname, env_vars in apikey_providers.items():
        key_val = ""
        for ev in env_vars:
            key_val = get_env_value(ev) or ""
            if key_val:
                break
        configured = bool(key_val)
        label = _status_t("messages.configured", "configured") if configured else _status_t("messages.not_configured_model", "not configured (run: hermes model)")
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
                ok, msg = False, _status_t("messages.unreachable_at", "unreachable at {base}", base=base)
            else:
                model_word = _status_count(len(models), "model", "models", "модель", "модели", "моделей").split(" ", 1)[1]
                ok, msg = True, _status_t("messages.reachable_at", "reachable ({count} {model_word}) at {base}", count=len(models), base=base, model_word=model_word)
        except AuthError:
            ok, msg = False, _status_t("messages.auth_rejected_set_key", "auth rejected — set LM_API_KEY")
        print(f"  {'LM Studio':<16} {check_mark(ok)} {msg}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.terminal_backend', 'Terminal Backend')}", Colors.CYAN, Colors.BOLD))

    terminal_cfg = config.get("terminal", {}) if isinstance(config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        terminal_env = terminal_cfg.get("backend", "local")
    print(f"  {_status_t('labels.backend', 'Backend')}:      {terminal_env}")

    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  {_status_t('labels.ssh_host', 'SSH Host')}:     {ssh_host or t('common.not_set', default='(not set)')}")
        print(f"  {_status_t('labels.ssh_user', 'SSH User')}:     {ssh_user or t('common.not_set', default='(not set)')}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  {_status_t('labels.docker_image', 'Docker Image')}: {docker_image}")
    elif terminal_env == "daytona":
        daytona_image = os.getenv("TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20")
        print(f"  {_status_t('labels.daytona_image', 'Daytona Image')}: {daytona_image}")
    elif terminal_env == "vercel_sandbox":
        runtime = os.getenv("TERMINAL_VERCEL_RUNTIME") or terminal_cfg.get("vercel_runtime") or "node24"
        persist = os.getenv("TERMINAL_CONTAINER_PERSISTENT")
        if persist is None:
            persist_enabled = bool(terminal_cfg.get("container_persistent", True))
        else:
            persist_enabled = persist.lower() in ("1", "true", "yes", "on")
        auth_status = describe_vercel_auth()
        sdk_ok = importlib.util.find_spec("vercel") is not None
        sdk_label = (
            _status_t("labels.installed", "installed")
            if sdk_ok
            else f"{_status_t('labels.not_installed', 'not installed')} (pip install 'hermes-agent[vercel]')"
        )
        print(f"  {_status_t('labels.runtime', 'Runtime')}:      {runtime}")
        print(f"  {_status_t('labels.sdk', 'SDK')}:          {check_mark(sdk_ok)} {sdk_label}")
        print(f"  {_status_t('labels.auth', 'Auth')}:         {check_mark(auth_status.ok)} {auth_status.label}")
        for line in auth_status.detail_lines:
            print(f"  {_status_t('labels.auth_detail', 'Auth detail')}:  {line}")
        print(f"  {_status_t('labels.persistence', 'Persistence')}:  {_status_t('messages.snapshot_filesystem', 'snapshot filesystem') if persist_enabled else _status_t('messages.ephemeral_filesystem', 'ephemeral filesystem')}")
        print(_status_t("messages.processes_note", "  Processes:    live processes do not survive cleanup, snapshots, or sandbox recreation"))

    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  {_status_t('labels.sudo', 'Sudo')}:         {check_mark(bool(sudo_password))} {_status_t('messages.enabled', 'enabled') if sudo_password else _status_t('messages.disabled', 'disabled')}")

    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.messaging_platforms', 'Messaging Platforms')}", Colors.CYAN, Colors.BOLD))

    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
        "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"),
        "WhatsApp": ("WHATSAPP_ENABLED", None),
        "Signal": ("SIGNAL_HTTP_URL", "SIGNAL_HOME_CHANNEL"),
        "Slack": ("SLACK_BOT_TOKEN", None),
        "Email": ("EMAIL_ADDRESS", "EMAIL_HOME_ADDRESS"),
        "SMS": ("TWILIO_ACCOUNT_SID", "SMS_HOME_CHANNEL"),
        "DingTalk": ("DINGTALK_CLIENT_ID", None),
        "Feishu": ("FEISHU_APP_ID", "FEISHU_HOME_CHANNEL"),
        "WeCom": ("WECOM_BOT_ID", "WECOM_HOME_CHANNEL"),
        "WeCom Callback": ("WECOM_CALLBACK_CORP_ID", None),
        "Weixin": ("WEIXIN_ACCOUNT_ID", "WEIXIN_HOME_CHANNEL"),
        "BlueBubbles": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_HOME_CHANNEL"),
        "QQBot": ("QQ_APP_ID", "QQ_HOME_CHANNEL"),
        "Yuanbao": ("YUANBAO_APP_ID", "YUANBAO_HOME_CHANNEL"),
    }

    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        # Back-compat: QQBot home channel was renamed from QQ_HOME_CHANNEL to QQBOT_HOME_CHANNEL
        if not home_channel and home_var == "QQBOT_HOME_CHANNEL":
            home_channel = os.getenv("QQ_HOME_CHANNEL", "")
        
        status = _status_t("messages.configured", "configured") if has_token else _status_t("messages.not_configured", "not configured")
        if home_channel:
            status += _status_t("messages.home_channel", " (home: {home})", home=home_channel)
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")

    # Plugin-registered platforms
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            configured = entry.check_fn()
            status_str = _status_t("messages.configured", "configured") if configured else _status_t("messages.not_configured", "not configured")
            label = entry.label
            print(f"  {label:<12}  {check_mark(configured)} {status_str} (plugin)")
    except Exception:
        pass

    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.gateway_service', 'Gateway Service')}", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot, _format_gateway_pids

        snapshot = get_gateway_runtime_snapshot()
        is_running = snapshot.running
        print(f"  {_status_t('labels.status', 'Status')}:       {check_mark(is_running)} {_status_t('messages.running', 'running') if is_running else _status_t('messages.stopped', 'stopped')}")
        print(f"  {_status_t('labels.manager', 'Manager')}:      {snapshot.manager}")
        if snapshot.gateway_pids:
            print(f"  {_status_t('labels.pid', 'PID')}:       {_format_gateway_pids(snapshot.gateway_pids)}")
        if snapshot.has_process_service_mismatch:
            print(_status_t("messages.service_not_managing", "  Service:      installed but not managing the current running gateway"))
        elif _is_termux() and not snapshot.gateway_pids:
            print(_status_t("messages.start_with_gateway", "  Start with:   hermes gateway"))
            print(_status_t("messages.termux_note", "  Note:         Android may stop background jobs when Termux is suspended"))
        elif snapshot.service_installed and not snapshot.service_running:
            print(_status_t("messages.installed_but_stopped", "  Service:      installed but stopped"))
    except Exception:
        if _is_termux():
            print(f"  {_status_t('labels.status', 'Status')}:       {color(t('common.unknown', default='(unknown)'), Colors.DIM)}")
            print(_status_t("messages.manager_termux", "  Manager:      Termux / manual process"))
        elif sys.platform.startswith('linux'):
            print(f"  {_status_t('labels.status', 'Status')}:       {color(t('common.unknown', default='(unknown)'), Colors.DIM)}")
            print(_status_t("messages.manager_systemd", "  Manager:      systemd/manual"))
        elif sys.platform == 'darwin':
            print(f"  {_status_t('labels.status', 'Status')}:       {color(t('common.unknown', default='(unknown)'), Colors.DIM)}")
            print(_status_t("messages.manager_launchd", "  Manager:      launchd"))
        else:
            print(f"  {_status_t('labels.status', 'Status')}:       {color(_status_t('messages.na', 'N/A'), Colors.DIM)}")
            print(_status_t("messages.manager_not_supported", "  Manager:      (not supported on this platform)"))

    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.scheduled_jobs', 'Scheduled Jobs')}", Colors.CYAN, Colors.BOLD))

    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            with open(jobs_file, encoding="utf-8") as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  {_status_t('labels.jobs', 'Jobs')}:         {_status_t('messages.jobs_active_total', '{active} active, {total} total', active=len(enabled_jobs), total=len(jobs))}")
        except Exception:
            print(_status_t("messages.jobs_error", "  Jobs:         (error reading jobs file)"))
    else:
        print(_status_t("messages.jobs_zero", "  Jobs:         0"))

    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color(f"◆ {_status_t('sections.sessions', 'Sessions')}", Colors.CYAN, Colors.BOLD))

    sessions_file = get_hermes_home() / "sessions" / "sessions.json"
    if sessions_file.exists():
        import json
        try:
            with open(sessions_file, encoding="utf-8") as f:
                data = json.load(f)
                print(f"  {_status_t('labels.active', 'Active')}:       {_status_count(len(data), 'session', 'sessions', 'сеанс', 'сеанса', 'сеансов')}")
        except Exception:
            print(_status_t("messages.sessions_error", "  Active:       (error reading sessions file)"))
    else:
        print(_status_t("messages.sessions_zero", "  Active:       0"))

    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color(f"◆ {_status_t('sections.deep_checks', 'Deep Checks')}", Colors.CYAN, Colors.BOLD))
        
        # Check OpenRouter connectivity
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            try:
                import httpx
                response = httpx.get(
                    OPENROUTER_MODELS_URL,
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    timeout=10
                )
                ok = response.status_code == 200
                print(f"  OpenRouter:   {check_mark(ok)} {_status_t('messages.reachable', 'reachable') if ok else _status_t('messages.error_with_code', 'error ({code})', code=response.status_code)}")
            except Exception as e:
                print(f"  OpenRouter:   {check_mark(False)} {_status_t('messages.error_with_value', 'error: {value}', value=e)}")
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Порт 18789:   {_status_t('messages.in_use', 'in use') if port_in_use else _status_t('messages.available', 'available')}")
        except OSError:
            pass

    print()
    print(color("─" * 60, Colors.DIM))
    print(color(_status_t("messages.run_doctor", "  Run 'hermes doctor' for detailed diagnostics"), Colors.DIM))
    print(color(_status_t("messages.run_setup", "  Run 'hermes setup' to configure"), Colors.DIM))
    print()
