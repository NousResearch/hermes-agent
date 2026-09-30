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
from hermes_cli.config import read_raw_config, get_env_path, get_env_value, get_hermes_home, load_config
from hermes_cli.config_defaults import DEFAULT_SANDBOX_IMAGE, DEFAULT_VERCEL_IMAGE
from hermes_cli.models import provider_label
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_cli.status_auth import (  # renderers wired into _SECTIONS below
    _render_api_keys, _render_apikey_providers, _render_auth_providers, _render_nous_gateway)
from hermes_constants import OPENROUTER_MODELS_URL


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
    "docker": (("Docker Image:", "TERMINAL_DOCKER_IMAGE", DEFAULT_SANDBOX_IMAGE, False),),
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


def _raw_terminal_config() -> dict:
    """Return terminal keys explicitly present in config.yaml, excluding defaults."""
    try:
        raw = read_raw_config()
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}
    terminal = raw.get("terminal", {})
    return terminal if isinstance(terminal, dict) else {}


def _render_terminal(ctx):
    _section("Terminal Backend")
    terminal_cfg = ctx.config.get("terminal", {}) if isinstance(ctx.config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "") or terminal_cfg.get("backend", "local")
    _kv("Backend:", terminal_env)
    if terminal_env in _TERMINAL_ENV_ROWS:
        for label, var, default, empty_is_unset in _TERMINAL_ENV_ROWS[terminal_env]:
            value = (os.getenv(var, "") or default) if empty_is_unset else os.getenv(var, default)
            print(f"  {label:<13} {value}")
    elif terminal_env == "daytona":
        raw_terminal_cfg = _raw_terminal_config()
        # --- Create mode & image/snapshot ---
        create_mode = os.getenv("TERMINAL_DAYTONA_CREATE_MODE") or terminal_cfg.get("daytona_create_mode", "image")
        if create_mode == "snapshot":
            snapshot = os.getenv("TERMINAL_DAYTONA_SNAPSHOT") or terminal_cfg.get("daytona_snapshot", "")
            print(f"  Mode:         snapshot")
            print(f"  Snapshot:     {snapshot or '(not set)'}")
            # Image is fallback in snapshot mode; show only if explicitly configured.
            daytona_image = os.getenv("TERMINAL_DAYTONA_IMAGE")
            if daytona_image is None and "daytona_image" in raw_terminal_cfg:
                daytona_image = raw_terminal_cfg.get("daytona_image", "")
            if daytona_image:
                print(f"  Fallback image: {daytona_image}")
        else:
            daytona_image = (
                os.getenv("TERMINAL_DAYTONA_IMAGE")
                or terminal_cfg.get("daytona_image")
                or DEFAULT_SANDBOX_IMAGE
            )
            print(f"  Mode:         image")
            print(f"  Image:        {daytona_image}")

        # --- Language ---
        language = os.getenv("TERMINAL_DAYTONA_LANGUAGE") or terminal_cfg.get("daytona_language", "")
        if language:
            print(f"  Language:     {language}")

        # --- Profile-scoped naming ---
        name_prefix = os.getenv("TERMINAL_DAYTONA_NAME_PREFIX") or terminal_cfg.get("daytona_name_prefix", "hermes")
        name_scope = os.getenv("TERMINAL_DAYTONA_NAME_SCOPE") or terminal_cfg.get("daytona_name_scope", "task")
        print(f"  Name prefix:  {name_prefix}")
        print(f"  Name scope:   {name_scope}")

        # --- Lifecycle ---
        auto_stop = os.getenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL") or terminal_cfg.get("daytona_auto_stop_interval", 0)
        auto_archive = os.getenv("TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL") or terminal_cfg.get("daytona_auto_archive_interval", 0)
        auto_delete = os.getenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL") or terminal_cfg.get("daytona_auto_delete_interval", 0)
        ephemeral = os.getenv("TERMINAL_DAYTONA_EPHEMERAL") or terminal_cfg.get("daytona_ephemeral", False)
        ephemeral_str = str(ephemeral).lower() if isinstance(ephemeral, bool) else str(ephemeral)
        print(f"  Ephemeral:    {ephemeral_str}")
        if auto_stop not in (0, "0", ""):
            print(f"  Auto-stop:    {auto_stop} min")
        if auto_archive not in (0, "0", ""):
            print(f"  Auto-archive: {auto_archive} min")
        if auto_delete not in (0, "0", ""):
            print(f"  Auto-delete:  {auto_delete} min")

        # --- Network ---
        block_all = str(os.getenv("TERMINAL_DAYTONA_NETWORK_BLOCK_ALL", "")).lower() or str(terminal_cfg.get("daytona_network_block_all", False)).lower()
        if block_all in ("true", "1", "yes"):
            print(f"  Network:      block requested (Daytona-enforced)")
            allow_list = os.getenv("TERMINAL_DAYTONA_NETWORK_ALLOW_LIST") or terminal_cfg.get("daytona_network_allow_list", "")
            if allow_list:
                print(f"  Allow list:   {allow_list}")
        else:
            print(f"  Network:     open")

        # --- SDK & Auth ---
        sdk_ok = importlib.util.find_spec("daytona") is not None
        sdk_label = "installed" if sdk_ok else "missing (run hermes setup terminal and select Daytona, then restart Hermes)"
        print(f"  SDK:          {check_mark(sdk_ok)} {sdk_label}")
        api_key_set = bool(os.getenv("DAYTONA_API_KEY"))
        print(f"  API key:      {check_mark(api_key_set)} {('configured' if api_key_set else 'not set')}")

        # --- CWD sync ---
        sync_cwd = os.getenv("TERMINAL_DAYTONA_SYNC_CWD") or terminal_cfg.get("daytona_sync_cwd", False)
        sync_cwd_bool = str(sync_cwd).lower() in ("true", "1", "yes") if isinstance(sync_cwd, str) else bool(sync_cwd)
        if sync_cwd_bool:
            print(f"  Sync CWD:    enabled (host project dir → /workspace)")

    elif terminal_env == "vercel_sandbox":
        persist = os.getenv("TERMINAL_CONTAINER_PERSISTENT")
        persist_enabled = (bool(terminal_cfg.get("container_persistent", True)) if persist is None
                           else persist.lower() in {"1", "true", "yes", "on"})
        auth_status = describe_vercel_auth()
        _kv("Image:", os.getenv('TERMINAL_VERCEL_RUNTIME') or terminal_cfg.get('vercel_runtime')
            or os.getenv('TERMINAL_VERCEL_IMAGE') or terminal_cfg.get('vercel_image') or DEFAULT_VERCEL_IMAGE)
        _kv_flag("SDK:", importlib.util.find_spec("vercel") is not None, "installed",
                 "missing (run hermes setup terminal and select Vercel Sandbox, then restart Hermes)")
        _kv("Auth:", f"{check_mark(auth_status.ok)} {auth_status.label}")
        for line in auth_status.detail_lines:
            _kv("Auth detail:", line)
        _kv("Persistence:", 'snapshot filesystem' if persist_enabled else 'ephemeral filesystem')
        _kv("Processes:", "live processes do not survive cleanup, snapshots, or sandbox recreation")
    else:
        # Plugin-registered terminal backends: show availability via the provider's doctor rows
        # (fail-soft — never break `hermes status`).
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            from agent.terminal_env_registry import get_provider
            provider = get_provider(terminal_env)
            if provider is not None:
                for ok, label, text in provider.doctor_checks():
                    print(f"  {label}: {check_mark(bool(ok))} {text}")
        except Exception:
            pass
    _kv_flag("Sudo:", os.getenv("SUDO_PASSWORD", ""), "enabled", "disabled")


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
        elif snapshot.service_installed and not snapshot.service_running:
            _kv("Service:", "installed but stopped")
    except Exception:
        platform = "linux" if sys.platform.startswith("linux") else sys.platform
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
            data = _load_json(sessions_file, encoding="utf-8-sig")
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
