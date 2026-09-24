"""Dump command for hermes CLI."""

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
    """Env-var names assigned a non-empty value in ~/.hermes/.env — what the managed backends (launchd /
    systemd / desktop ``serve``) load, as opposed to the shell exports ``os.getenv`` reflects here.

    ``hermes debug share`` runs in a terminal, so ``os.getenv`` reflects the shell's environment, which can
    include exported keys the managed backend never sees. Comparing against this set lets the dump flag that
    mismatch (the exact trap behind #48504-style "no web_search" reports: key exported in the shell, absent
    from .env, invisible to the launchd backend).
    """
    try:
        text = get_env_path().read_text(encoding="utf-8", errors="ignore")
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
        if name.strip() and value.strip().strip("'\""):  # a bare `KEY=` is effectively unset for the backend
            names.add(name.strip())
    return names


def _git_output(project_root: Path, *args: str) -> str:
    """Stripped stdout of ``git <args>`` run in *project_root*, or '' on any failure."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                                timeout=5, cwd=str(project_root))
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _get_git_commit(project_root: Path) -> str:
    """Short git commit hash, or '(unknown)'. Docker images exclude ``.git``, so fall back to the build SHA
    the Dockerfile bakes into ``<project_root>/.hermes_build_sha``."""
    value = _git_output(project_root, "rev-parse", "--short=8", "HEAD")
    if value:
        return value
    try:
        from hermes_cli.build_info import get_build_sha  # deferred: keeps dump cheap on non-dump paths
        return get_build_sha(short=8) or "(unknown)"
    except Exception:
        return "(unknown)"


def _get_git_commit_date(project_root: Path) -> str:
    """Return the date the HEAD commit was authored (YYYY-MM-DD), or '' (Docker images have no .git)."""
    return _git_output(project_root, "log", "-1", "--format=%cd", "--date=short", "HEAD")


def _redact(value: str) -> str:
    """Redact all but first 4 and last 4 chars; ``""`` for an empty value (dump formats empties as blank)."""
    from agent.redact import mask_secret
    return mask_secret(value)


def _gateway_status() -> str:
    """Return a short gateway status string."""
    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot
        snapshot = get_gateway_runtime_snapshot()
        if snapshot.running:
            mode = "manual" if snapshot.has_process_service_mismatch else snapshot.manager
            return f"running ({mode}, pid {snapshot.gateway_pids[0]})"
        return f"stopped ({snapshot.manager})"
    except Exception:
        return "unknown" if sys.platform.startswith(("linux", "darwin")) else "N/A"


def _count_skills(hermes_home: Path) -> int:
    """Count installed skills."""
    skills_dir = hermes_home / "skills"
    return sum(1 for item in skills_dir.rglob("SKILL.md") if not is_excluded_skill_path(item)) if skills_dir.is_dir() else 0


def _cron_summary(hermes_home: Path) -> str:
    """Return cron jobs summary."""
    jobs_file = hermes_home / "cron" / "jobs.json"
    if not jobs_file.exists():
        return "0"
    try:
        # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors may leave a UTF-8 BOM.
        with open(jobs_file, encoding="utf-8-sig") as f:
            jobs = json.load(f).get("jobs", [])
        active = sum(1 for j in jobs if j.get("enabled", True))
        return f"{active} active / {len(jobs)} total"
    except Exception:
        return "(error reading)"


_PLATFORM_ENV_VARS = {
    "telegram": "TELEGRAM_BOT_TOKEN", "discord": "DISCORD_BOT_TOKEN", "slack": "SLACK_BOT_TOKEN",
    "whatsapp": "WHATSAPP_ENABLED", "signal": "SIGNAL_HTTP_URL", "email": "EMAIL_ADDRESS",
    "sms": "TWILIO_ACCOUNT_SID", "matrix": "MATRIX_HOMESERVER_URL", "mattermost": "MATTERMOST_URL",
    "homeassistant": "HASS_TOKEN", "dingtalk": "DINGTALK_CLIENT_ID", "feishu": "FEISHU_APP_ID",
    "wecom": "WECOM_BOT_ID", "wecom_callback": "WECOM_CALLBACK_CORP_ID", "weixin": "WEIXIN_ACCOUNT_ID",
    "qqbot": "QQ_APP_ID",
}


def _get_model_and_provider(config: dict) -> tuple[str, str]:
    """Extract model and provider from config."""
    model_cfg = config.get("model", "")
    if isinstance(model_cfg, dict):
        return (model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name") or "(not set)", model_cfg.get("provider") or "(auto)")
    return (model_cfg if isinstance(model_cfg, str) else "") or "(not set)", "(auto)"


# Sections with interesting user-facing overrides
_INTERESTING_PATHS = (
    ("agent", "max_turns"), ("agent", "gateway_timeout"), ("agent", "session_stall_timeout"),
    ("agent", "sanitizer_heal_escalation_threshold"), ("agent", "tool_use_enforcement"),
    ("agent", "execution_guidance"), ("terminal", "backend"), ("terminal", "docker_image"),
    ("terminal", "persistent_shell"), ("browser", "allow_private_urls"), ("compression", "enabled"),
    ("compression", "threshold"), ("compression", "in_place"), ("display", "streaming"),
    ("display", "skin"), ("display", "show_reasoning"), ("privacy", "redact_pii"), ("tts", "provider"),
)


def _config_overrides(config: dict) -> dict[str, str]:
    """Find non-default config values worth reporting."""
    from hermes_cli.config import DEFAULT_CONFIG
    overrides = {}
    for section, key in _INTERESTING_PATHS:
        default_section = DEFAULT_CONFIG.get(section, {})
        user_section = config.get(section, {})
        if not isinstance(default_section, dict) or not isinstance(user_section, dict):
            continue
        user_val = user_section.get(key)
        if user_val is not None and user_val != default_section.get(key):
            overrides[f"{section}.{key}"] = str(user_val)
    user_toolsets = config.get("toolsets", [])
    if user_toolsets != DEFAULT_CONFIG.get("toolsets", []):
        overrides["toolsets"] = str(user_toolsets)
    fallbacks = config.get("fallback_providers", [])
    if fallbacks:
        overrides["fallback_providers"] = str(fallbacks)
    return overrides


# (env var, dump label) in display order.
_API_KEYS = [
    ("OPENROUTER_API_KEY", "openrouter"), ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"), ("ANTHROPIC_TOKEN", "anthropic_token"),
    ("NOUS_API_KEY", "nous"), ("GOOGLE_API_KEY", "google/gemini"), ("GEMINI_API_KEY", "gemini"),
    ("GLM_API_KEY", "glm/zai"), ("ZAI_API_KEY", "zai"), ("KIMI_API_KEY", "kimi"),
    ("MINIMAX_API_KEY", "minimax"), ("DEEPSEEK_API_KEY", "deepseek"),
    ("DASHSCOPE_API_KEY", "dashscope"), ("HF_TOKEN", "huggingface"), ("NVIDIA_API_KEY", "nvidia"),
    ("AI_GATEWAY_API_KEY", "ai_gateway"), ("OPENCODE_ZEN_API_KEY", "opencode_zen"),
    ("OPENCODE_GO_API_KEY", "opencode_go"), ("COMMANDCODE_API_KEY", "commandcode"),
    ("KILOCODE_API_KEY", "kilocode"), ("FIRECRAWL_API_KEY", "firecrawl"), ("TAVILY_API_KEY", "tavily"), ("PERPLEXITY_API_KEY", "perplexity"),
    ("KEENABLE_API_KEY", "keenable"), ("BROWSERBASE_API_KEY", "browserbase"), ("FAL_KEY", "fal"),
    ("ELEVENLABS_API_KEY", "elevenlabs"), ("GITHUB_TOKEN", "github"),
]


def _version_line(project_root: Path) -> str:
    """``<version> [<commit>] (<commit date>)`` — the commit date is the real "as-of" date; __release_date__
    is intentionally NOT shown (reads like a wall-clock timestamp, confuses triage)."""
    try:
        from hermes_cli import __version__
    except ImportError:
        __version__ = "(unknown)"
    ver_str = f"{__version__} [{_get_git_commit(project_root)}]"
    commit_date = _get_git_commit_date(project_root)
    return f"{ver_str} ({commit_date})" if commit_date else ver_str


def _effective_terminal_backend(config: dict) -> str:
    """The EFFECTIVE backend: a TERMINAL_ENV set directly in .env / the shell overrides ``terminal.backend`` and
    is what terminal_tool uses. run_dump() has already loaded .env, so os.environ reflects the override."""
    config_backend = config.get("terminal", {}).get("backend", "local")
    env_backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if env_backend and env_backend != str(config_backend).strip().lower():
        return f"{env_backend}  (TERMINAL_ENV overrides config.yaml terminal.backend={config_backend})"
    return config_backend


def _api_key_lines(show_keys: bool) -> list[str]:
    dotenv_keys = _dotenv_key_names()
    lines = []
    for env_var, label in _API_KEYS:
        val = os.getenv(env_var, "")
        display = _redact(val) if show_keys and val else ("set" if val else "not set")
        # Set in this shell but absent from ~/.hermes/.env: a managed backend loads .env, not the login
        # shell, so it likely can't see this key — flag it so support doesn't chase a phantom "configured".
        if val and env_var not in dotenv_keys:
            display += " (shell only — not in .env; managed/desktop backend may not see it)"
        # `hermes auth add openrouter` credentials live in the pool, not env — don't read "not set".
        # A credential added via `hermes auth add openrouter` lives in the credential pool, not as an env
        # var — surface it so the dump doesn't misleadingly read "not set" while `hermes auth list` shows it
        # (#42130).
        if not val and label == "openrouter":
            try:
                from agent.credential_pool import load_pool as _load_pool
                if _load_pool("openrouter").has_credentials():
                    display = "set (auth pool)"
            except Exception:
                pass
        lines.append(f"  {label:<20} {display}")
    return lines


def _openai_version() -> str:
    try:
        import openai
        return openai.__version__
    except ImportError:
        return "not installed"


def run_dump(args):
    """Output a compact, copy-pasteable setup summary."""
    show_keys = getattr(args, "show_keys", False)
    # Load env from .env file so key checks work
    load_hermes_dotenv(hermes_home=get_env_path().parent, project_env=get_project_root() / ".env")
    project_root, hermes_home = get_project_root(), get_hermes_home()
    try:
        config = load_config()
    except Exception:
        config = {}
    model, provider = _get_model_and_provider(config)
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name() or "(default)"
    except Exception:
        profile = "(default)"
    toolsets = config.get("toolsets", ["hermes-cli"])
    platforms = [name for name, env in _PLATFORM_ENV_VARS.items() if os.getenv(env)]
    lines = [
        "--- hermes dump ---",
        f"version:          {_version_line(project_root)}",
        f"os:               {platform.system()} {platform.release()} {platform.machine()}",
        f"python:           {sys.version.split()[0]}",
        f"openai_sdk:       {_openai_version()}",
        f"profile:          {profile}",
        f"hermes_home:      {display_hermes_home()}",
        f"model:            {model}",
        f"provider:         {provider}",
        f"terminal:         {_effective_terminal_backend(config)}",
        "", "api_keys:", *_api_key_lines(show_keys),
        "", "features:",
        f"  toolsets:           {', '.join(toolsets) if toolsets else '(default)'}",
        f"  mcp_servers:        {len(config.get('mcp', {}).get('servers', {}))}",
        f"  memory_provider:    {config.get('memory', {}).get('provider', '') or 'built-in'}",
        f"  gateway:            {_gateway_status()}",
        f"  platforms:          {', '.join(platforms) if platforms else 'none'}",
        f"  cron_jobs:          {_cron_summary(hermes_home)}",
        f"  skills:             {_count_skills(hermes_home)}",
    ]
    overrides = _config_overrides(config)
    if overrides:
        lines += ["", "config_overrides:"] + [f"  {key}: {val}" for key, val in overrides.items()]
    print("\n".join(lines + ["--- end dump ---"]))
