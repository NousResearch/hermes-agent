"""
Dump command for hermes CLI.

Outputs a compact, plain-text summary of the user's Hermes setup
that can be copy-pasted into Discord/GitHub/Telegram for support context.
No ANSI colors, no checkmarks — just data.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from hermes_cli.config import get_hermes_home, get_env_path, get_project_root, load_config
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import display_hermes_home
from agent.skill_utils import is_excluded_skill_path


def _dotenv_key_names() -> set[str]:
    """Return the set of env-var names assigned a non-empty value in ~/.hermes/.env.

    The managed backends (launchd / systemd / the desktop-spawned ``serve``
    process) load credentials from this file — NOT from an interactive shell's
    exports. ``hermes debug share`` runs in a terminal, so ``os.getenv`` reflects
    the shell's environment, which can include exported keys the managed backend
    never sees. Comparing against this set lets the dump flag that mismatch (the
    exact trap behind #48504-style "no web_search" reports: key exported in the
    shell, absent from .env, invisible to the launchd backend).
    """
    try:
        env_path = get_env_path()
        text = env_path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return set()

    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        # A bare `KEY=` (empty value) is effectively unset for the backend.
        if name and value.strip().strip("'\""):
            names.add(name)
    return names


def _get_git_commit(project_root: Path) -> str:
    """Return short git commit hash, or '(unknown)'.

    Source installs and dev images resolve this live via ``git rev-parse``.
    The published Docker image excludes ``.git`` from the build context, so
    that lookup always fails — we fall back to the baked-in build SHA written
    to ``<project_root>/.hermes_build_sha`` by the Dockerfile's
    ``HERMES_GIT_SHA`` build-arg (see ``hermes_cli/build_info.py``).
    The output format is identical regardless of source.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                return value
    except Exception:
        pass

    # Fall back to the build-time baked SHA (populated in published Docker
    # images, absent otherwise).  Defers the import so the dump module
    # stays cheap on non-dump code paths.
    try:
        from hermes_cli.build_info import get_build_sha
        baked = get_build_sha(short=8)
        if baked:
            return baked
    except Exception:
        pass

    return "(unknown)"


def _get_git_commit_date(project_root: Path) -> str:
    """Return the date the HEAD commit was authored (YYYY-MM-DD), or ''.

    Resolves live via ``git log`` on source installs.  The published Docker
    image excludes ``.git``, so this returns '' there — the dump line simply
    drops the date suffix in that case (the baked SHA still identifies the
    build).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                return value
    except Exception:
        pass

    return ""


def _redact(value: str) -> str:
    """Redact all but first 4 and last 4 chars.

    Thin wrapper over :func:`agent.redact.mask_secret`. Returns ``""`` for
    an empty value (matches the historical behavior of this helper —
    ``hermes dump`` formats empty values as blank, not as ``"(not set)"``).
    """
    from agent.redact import mask_secret
    return mask_secret(value)


def _gateway_status() -> str:
    """Return a short gateway status string."""
    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot

        snapshot = get_gateway_runtime_snapshot()
        if snapshot.running:
            mode = snapshot.manager
            if snapshot.has_process_service_mismatch:
                mode = "manual"
            return f"running ({mode}, pid {snapshot.gateway_pids[0]})"
        if snapshot.service_installed and not snapshot.service_running:
            return f"stopped ({snapshot.manager})"
        return f"stopped ({snapshot.manager})"
    except Exception:
        return "unknown" if sys.platform.startswith(("linux", "darwin")) else "N/A"


def _count_skills(hermes_home: Path) -> int:
    """Count installed skills."""
    skills_dir = hermes_home / "skills"
    if not skills_dir.is_dir():
        return 0
    count = 0
    for item in skills_dir.rglob("SKILL.md"):
        if is_excluded_skill_path(item):
            continue
        count += 1
    return count


def _count_mcp_servers(config: dict) -> int:
    """Count configured MCP servers."""
    mcp = config.get("mcp", {})
    servers = mcp.get("servers", {})
    return len(servers)


def _cron_summary(hermes_home: Path) -> str:
    """Return cron jobs summary."""
    jobs_file = hermes_home / "cron" / "jobs.json"
    if not jobs_file.exists():
        return "0"
    try:
        with open(jobs_file, encoding="utf-8") as f:
            data = json.load(f)
        jobs = data.get("jobs", [])
        active = sum(1 for j in jobs if j.get("enabled", True))
        return f"{active} active / {len(jobs)} total"
    except Exception:
        return "(error reading)"


def _configured_platforms() -> list[str]:
    """Return list of configured messaging platform names."""
    checks = {
        "telegram": "TELEGRAM_BOT_TOKEN",
        "discord": "DISCORD_BOT_TOKEN",
        "slack": "SLACK_BOT_TOKEN",
        "whatsapp": "WHATSAPP_ENABLED",
        "signal": "SIGNAL_HTTP_URL",
        "email": "EMAIL_ADDRESS",
        "sms": "TWILIO_ACCOUNT_SID",
        "matrix": "MATRIX_HOMESERVER_URL",
        "mattermost": "MATTERMOST_URL",
        "homeassistant": "HASS_TOKEN",
        "dingtalk": "DINGTALK_CLIENT_ID",
        "feishu": "FEISHU_APP_ID",
        "wecom": "WECOM_BOT_ID",
        "wecom_callback": "WECOM_CALLBACK_CORP_ID",
        "weixin": "WEIXIN_ACCOUNT_ID",
        "qqbot": "QQ_APP_ID",
    }
    return [name for name, env in checks.items() if os.getenv(env)]


def _memory_provider(config: dict) -> str:
    """Return the active memory provider name."""
    mem = config.get("memory", {})
    provider = mem.get("provider", "")
    return provider if provider else "built-in"


def _get_model_and_provider(config: dict) -> tuple[str, str]:
    """Extract model and provider from config."""
    model_cfg = config.get("model", "")
    if isinstance(model_cfg, dict):
        model = model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name") or "(not set)"
        provider = model_cfg.get("provider") or "(auto)"
    elif isinstance(model_cfg, str):
        model = model_cfg or "(not set)"
        provider = "(auto)"
    else:
        model = "(not set)"
        provider = "(auto)"
    return model, provider


def _config_overrides(config: dict) -> dict[str, str]:
    """Find non-default config values worth reporting.
    
    Returns a flat dict of dotpath -> value for interesting overrides.
    """
    from hermes_cli.config import DEFAULT_CONFIG

    overrides = {}

    # Sections with interesting user-facing overrides
    interesting_paths = [
        ("agent", "max_turns"),
        ("agent", "gateway_timeout"),
        ("agent", "tool_use_enforcement"),
        ("terminal", "backend"),
        ("terminal", "docker_image"),
        ("terminal", "persistent_shell"),
        ("browser", "allow_private_urls"),
        ("compression", "enabled"),
        ("compression", "threshold"),
        ("display", "streaming"),
        ("display", "skin"),
        ("display", "show_reasoning"),
        ("privacy", "redact_pii"),
        ("tts", "provider"),
    ]

    for section, key in interesting_paths:
        default_section = DEFAULT_CONFIG.get(section, {})
        user_section = config.get(section, {})
        if not isinstance(default_section, dict) or not isinstance(user_section, dict):
            continue
        default_val = default_section.get(key)
        user_val = user_section.get(key)
        if user_val is not None and user_val != default_val:
            overrides[f"{section}.{key}"] = str(user_val)

    # Toolsets (if different from default)
    default_toolsets = DEFAULT_CONFIG.get("toolsets", [])
    user_toolsets = config.get("toolsets", [])
    if user_toolsets != default_toolsets:
        overrides["toolsets"] = str(user_toolsets)

    # Fallback providers
    fallbacks = config.get("fallback_providers", [])
    if fallbacks:
        overrides["fallback_providers"] = str(fallbacks)

    return overrides


def run_dump(args):
    """Output a compact, copy-pasteable setup summary."""
    show_keys = getattr(args, "show_keys", False)

    # Load env from .env file so key checks work
    env_path = get_env_path()
    load_hermes_dotenv(
        hermes_home=env_path.parent,
        project_env=get_project_root() / ".env",
    )

    project_root = get_project_root()
    hermes_home = get_hermes_home()

    try:
        from hermes_cli import __version__
    except ImportError:
        __version__ = "(unknown)"

    commit = _get_git_commit(project_root)
    commit_date = _get_git_commit_date(project_root)

    try:
        config = load_config()
    except Exception:
        config = {}

    model, provider = _get_model_and_provider(config)

    # Profile
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name() or "(default)"
    except Exception:
        profile = "(default)"

    # Terminal backend — report the EFFECTIVE backend, not just config.yaml.
    # ``terminal.backend`` in config.yaml is bridged to the TERMINAL_ENV env var,
    # but a TERMINAL_ENV set directly in .env / the shell overrides config and is
    # what terminal_tool actually uses (tools/terminal_tool.py reads TERMINAL_ENV).
    # Reporting only the config value hides that override and sends users chasing
    # the wrong cause when the agent runs in a docker/podman sandbox even though
    # config says "local" (and vice-versa). run_dump() has already loaded .env,
    # so os.environ reflects the real override here.
    terminal_cfg = config.get("terminal", {})
    config_backend = terminal_cfg.get("backend", "local")
    env_backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if env_backend and env_backend != str(config_backend).strip().lower():
        backend = (
            f"{env_backend}  (TERMINAL_ENV overrides config.yaml "
            f"terminal.backend={config_backend})"
        )
    else:
        backend = config_backend

    # OpenAI SDK version
    try:
        import openai
        openai_ver = openai.__version__
    except ImportError:
        openai_ver = "not installed"

    # OS info
    os_info = f"{platform.system()} {platform.release()} {platform.machine()}"

    lines = []
    lines.append("--- hermes dump ---")
    # Identify the build by commit + the date that commit was made, resolved
    # live via git.  __release_date__ (the package release date) is
    # intentionally NOT shown here — it reads like a wall-clock timestamp and
    # confuses support triage.  The commit date is the real "as-of" date.
    ver_str = f"{__version__}"
    ver_str += f" [{commit}]"
    if commit_date:
        ver_str += f" ({commit_date})"
    lines.append(f"version:          {ver_str}")
    lines.append(f"os:               {os_info}")
    lines.append(f"python:           {sys.version.split()[0]}")
    lines.append(f"openai_sdk:       {openai_ver}")
    lines.append(f"profile:          {profile}")
    lines.append(f"hermes_home:      {display_hermes_home()}")
    lines.append(f"model:            {model}")
    lines.append(f"provider:         {provider}")
    lines.append(f"terminal:         {backend}")

    # API keys
    lines.append("")
    lines.append("api_keys:")
    # Each entry: (env_var, label, pool_key_or_None).  ``pool_key`` is the
    # provider id under ``auth.json::credential_pool`` whose entries should
    # be consulted when the env var is unset — this lets OAuth-backed
    # providers (Nous Portal, Anthropic via claude_code, etc.) report as
    # set even when no API-key env var is configured.  ``None`` means the
    # provider has no OAuth flow that lands in the credential pool.
    api_keys = [
        ("OPENROUTER_API_KEY", "openrouter", "openrouter"),
        ("OPENAI_API_KEY", "openai", None),
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic"),
        ("ANTHROPIC_TOKEN", "anthropic_token", "anthropic"),
        ("NOUS_API_KEY", "nous", "nous"),
        ("GOOGLE_API_KEY", "google/gemini", "gemini"),
        ("GEMINI_API_KEY", "gemini", "gemini"),
        ("GLM_API_KEY", "glm/zai", "zai"),
        ("ZAI_API_KEY", "zai", "zai"),
        ("KIMI_API_KEY", "kimi", "kimi-coding"),
        ("MINIMAX_API_KEY", "minimax", "minimax"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek"),
        ("DASHSCOPE_API_KEY", "dashscope", "alibaba"),
        ("HF_TOKEN", "huggingface", None),
        ("NVIDIA_API_KEY", "nvidia", "nvidia"),
        ("OPENCODE_ZEN_API_KEY", "opencode_zen", "opencode-zen"),
        ("OPENCODE_GO_API_KEY", "opencode_go", "opencode-go"),
        ("KILOCODE_API_KEY", "kilocode", None),
        ("FIRECRAWL_API_KEY", "firecrawl", None),
        ("TAVILY_API_KEY", "tavily", None),
        ("BROWSERBASE_API_KEY", "browserbase", None),
        ("FAL_KEY", "fal", None),
        ("ELEVENLABS_API_KEY", "elevenlabs", None),
        ("GITHUB_TOKEN", "github", "copilot"),
    ]

    dotenv_keys = _dotenv_key_names()

    # OAuth-backed providers with no env var in the table above.  Their
    # tokens live in ``credential_pool`` only, so the user previously had
    # no way to tell from ``hermes debug`` whether they were logged in.
    oauth_only_providers = [
        ("openai-codex", "openai_codex"),
        ("qwen-oauth", "qwen_oauth"),
        ("minimax-oauth", "minimax_oauth"),
        ("google-gemini-cli", "google_gemini_cli"),
    ]

    try:
        from hermes_cli.auth import read_credential_pool
        pool_data = read_credential_pool(None)
    except Exception:
        pool_data = {}

    def _pool_has_token(pool_key: str) -> bool:
        entries = pool_data.get(pool_key)
        if not isinstance(entries, list):
            return False
        return any(
            isinstance(e, dict) and str(e.get("access_token") or "").strip()
            for e in entries
        )

    for env_var, label, pool_key in api_keys:
        val = os.getenv(env_var, "")
        if val:
            display = _redact(val) if show_keys else "set (env)"
        elif pool_key and _pool_has_token(pool_key):
            display = "set (oauth)"
        else:
            display = "not set"
        # Set in this (shell) process but absent from ~/.hermes/.env: a managed
        # backend (launchd/systemd/desktop `serve`) loads .env, not the login
        # shell, so it likely can't see this key — even though the dump reads
        # "set". Flag it so support doesn't chase a phantom "key is configured"
        # (the actual cause of gated tools like web_search going missing).
        if val and env_var not in dotenv_keys:
            display += " (shell only — not in .env; managed/desktop backend may not see it)"
        # A credential added via `hermes auth add openrouter` lives in the
        # credential pool, not as an env var — surface it so the dump doesn't
        # misleadingly read "not set" while `hermes auth list` shows it (#42130).
        if display == "not set" and label == "openrouter":
            try:
                from agent.credential_pool import load_pool as _load_pool

                if _load_pool("openrouter").has_credentials():
                    display = "set (auth pool)"
            except Exception:
                pass
        lines.append(f"  {label:<20} {display}")

    for pool_key, label in oauth_only_providers:
        display = "set (oauth)" if _pool_has_token(pool_key) else "not set"
        lines.append(f"  {label:<20} {display}")

    # Features summary
    lines.append("")
    lines.append("features:")

    toolsets = config.get("toolsets", ["hermes-cli"])
    lines.append(f"  toolsets:           {', '.join(toolsets) if toolsets else '(default)'}")
    lines.append(f"  mcp_servers:        {_count_mcp_servers(config)}")
    lines.append(f"  memory_provider:    {_memory_provider(config)}")
    lines.append(f"  gateway:            {_gateway_status()}")

    platforms = _configured_platforms()
    lines.append(f"  platforms:          {', '.join(platforms) if platforms else 'none'}")
    lines.append(f"  cron_jobs:          {_cron_summary(hermes_home)}")
    lines.append(f"  skills:             {_count_skills(hermes_home)}")

    # Config overrides (non-default values)
    overrides = _config_overrides(config)
    if overrides:
        lines.append("")
        lines.append("config_overrides:")
        for key, val in overrides.items():
            lines.append(f"  {key}: {val}")

    lines.append("--- end dump ---")

    output = "\n".join(lines)
    print(output)
