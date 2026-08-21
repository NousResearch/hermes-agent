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
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
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
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
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
        # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors
        # may leave a UTF-8 BOM that plain utf-8 json.load rejects.
        with open(jobs_file, encoding="utf-8-sig") as f:
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


# Substrings marking a fallback-provider field as carrying a secret VALUE.
# Matched against the lowercased field name so siblings of ``api_key`` —
# ``access_key``, ``auth_token``, ``client_secret`` — are covered too.
_FALLBACK_SECRET_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth",
)

# Fields matching a marker above that are nonetheless safe to show. ``key_env``
# and ``api_key_env`` NAME an environment variable rather than holding its
# value, and operators need them to diagnose routing; the ``*_tokens`` limits
# are numeric budgets. Anything not listed here fails closed: an unrecognized
# field whose name matches a marker is masked, costing a diagnostic rather
# than leaking a credential.
_FALLBACK_SAFE_FIELDS = frozenset(
    {
        "key_env",
        "api_key_env",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "token_limit",
    }
)

# Fields whose string values are URLs. Credentials hide in userinfo
# (``https://user:KEY@host``) and in query parameters (``?api_key=KEY``), so
# these are cleaned rather than shown verbatim.
_FALLBACK_URL_MARKERS = ("url", "endpoint")

# Stand-in for a value we cannot render safely. Handing an unknown object to
# ``str()`` would run its ``__repr__``, which may print the very key we are
# trying to hide.
_FALLBACK_OMITTED = "<omitted>"

# Bound on nested-container recursion, so a deep or self-referencing config
# cannot turn the dump into a stack overflow.
_FALLBACK_MAX_DEPTH = 6


def _is_fallback_secret_field(name) -> bool:
    """True when a field name marks its value as a credential."""
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    if lowered in _FALLBACK_SAFE_FIELDS:
        return False
    return any(marker in lowered for marker in _FALLBACK_SECRET_MARKERS)


def _is_fallback_url_field(name) -> bool:
    """True when a field name marks its value as a URL."""
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    return any(marker in lowered for marker in _FALLBACK_URL_MARKERS)


def _fallback_placeholder(value) -> str:
    """Describe a value by type instead of rendering it."""
    return f"<omitted: {type(value).__name__}>"


def _is_sensitive_query_param(name) -> bool:
    """True when a URL query parameter name carries a credential.

    The repository already owns this policy in ``agent.redact``, and it lists
    names the field-name markers above cannot reach — ``signature``,
    ``x-amz-signature``, ``code`` — so a pre-signed URL or an OAuth callback
    would otherwise print its secret in full. Ask that policy first, so this
    path stays in step with it as it grows, and fall back to the markers for
    names it doesn't list (``access_key``, ``auth_token``, ``client_secret``).
    """
    if not isinstance(name, str):
        return False

    from agent.redact import is_sensitive_query_param

    if is_sensitive_query_param(name):
        return True
    return _is_fallback_secret_field(name)


def _mask_url_credentials(value: str) -> str:
    """Strip userinfo and redact secret parameters from a URL.

    Both parameter-bearing components are masked. A credential does not care
    which separator introduced it: OAuth implicit-flow hands back its token
    after ``#``, and ``agent/redact.py`` classifies query and fragment pairs
    with the same set (``_STRICT_URL_PARAM_RE`` carries ``#`` among its
    delimiters), so a policy that stopped at ``?`` would just be a hole with a
    good excuse.
    """
    import re
    from urllib.parse import unquote_plus, urlsplit, urlunsplit

    try:
        parts = urlsplit(value)
        username = parts.username
        password = parts.password
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        # Unparseable (e.g. a bad port) — refuse rather than guess.
        return _FALLBACK_OMITTED
    # A scheme-relative reference (``//user:pass@host/v1``) has no scheme and
    # still carries userinfo, so the authority — not the scheme — is what makes
    # this a URL worth masking.
    if not parts.netloc:
        return value

    def _mask_params(component: str) -> tuple[str, bool]:
        """Redact credential-named pairs in one parameter component.

        Split on ``&`` *and* ``;``. ``parse_qsl`` stopped honouring ``;`` in
        CPython 3.9.2 (bpo-42967) because a *request parser* that accepts both
        separators can be induced to disagree with the proxy in front of it.
        A masker carries no such exposure — reading both can only mask more,
        never less, and its output never re-enters request handling — and
        ``agent/redact.py`` already calls ``;`` a separator
        (``_STRICT_URL_PARAM_RE``). Left to ``parse_qsl``,
        ``?region=eu;signature=<secret>`` is one pair named ``region`` and the
        secret ships whole.

        Separators and untouched pairs are copied through verbatim rather than
        re-encoded, so the endpoint stays the URL the user typed.
        """
        if not component:
            return component, False
        out: list[str] = []
        changed = False
        for token in re.split(r"([&;])", component):
            if token in ("&", ";"):
                out.append(token)
                continue
            name, sep, val = token.partition("=")
            if sep and val and _is_sensitive_query_param(unquote_plus(name)):
                out.append(f"{name}={_redact(val)}")
                changed = True
            else:
                out.append(token)
        if not changed:
            return component, False
        return "".join(out), True

    netloc = parts.netloc
    changed = False
    if username or password:
        host = hostname or ""
        if port:
            host = f"{host}:{port}"
        netloc = f"***@{host}" if host else "***"
        changed = True

    query, query_masked = _mask_params(parts.query)
    fragment, fragment_masked = _mask_params(parts.fragment)
    changed = changed or query_masked or fragment_masked

    if not changed:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def _mask_fallback_value(name, value, depth: int):
    """Return a safely renderable copy of one fallback-provider field value."""
    if depth > _FALLBACK_MAX_DEPTH:
        return _FALLBACK_OMITTED

    if _is_fallback_secret_field(name):
        # Only a non-empty string can be shown in the dump's head/tail form.
        # Bytes, numbers, nested containers and custom objects are dropped —
        # each of them still stringifies to the credential in full.
        if isinstance(value, str):
            return _redact(value) if value else value
        if value is None or isinstance(value, bool):
            return value
        return _FALLBACK_OMITTED

    if isinstance(value, str):
        return _mask_url_credentials(value) if _is_fallback_url_field(name) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return _mask_fallback_entry(value, depth + 1)
    if isinstance(value, (list, tuple)):
        return [_mask_fallback_value(name, item, depth + 1) for item in value]
    return _fallback_placeholder(value)


def _mask_fallback_entry(entry: dict, depth: int = 0):
    """Copy one fallback-provider entry with credential values masked."""
    if depth > _FALLBACK_MAX_DEPTH:
        return _FALLBACK_OMITTED
    return {
        name: _mask_fallback_value(name, value, depth) for name, value in entry.items()
    }


def _mask_fallback_secrets(fallbacks):
    """Copy the fallback-provider config with inline credential values masked.

    ``hermes dump`` output is copy-pasted into public bug reports and uploaded
    verbatim by ``hermes debug share``, so an inline ``api_key`` must never
    appear in full. Secret values are masked through :func:`_redact` (head/tail
    only), matching how the dump shows every other key.

    Both container shapes ``fallback_config._iter_fallback_entries`` accepts are
    handled: a list of entries, and a single entry given as a bare mapping.
    Whatever is left cannot be shown safely — ``str()`` on an arbitrary object
    runs its ``__repr__`` — so it is replaced by a type marker.
    """
    if isinstance(fallbacks, dict):
        return _mask_fallback_entry(fallbacks)
    if isinstance(fallbacks, (list, tuple)):
        return [
            _mask_fallback_entry(entry)
            if isinstance(entry, dict)
            else _fallback_placeholder(entry)
            for entry in fallbacks
        ]
    return _fallback_placeholder(fallbacks)


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
        ("agent", "session_stall_timeout"),
        ("agent", "tool_use_enforcement"),
        ("agent", "execution_guidance"),
        ("terminal", "backend"),
        ("terminal", "docker_image"),
        ("terminal", "persistent_shell"),
        ("browser", "allow_private_urls"),
        ("compression", "enabled"),
        ("compression", "threshold"),
        ("compression", "in_place"),
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

    # Fallback providers — mask inline credentials before serializing; this
    # block is uploaded verbatim by ``hermes debug share``.
    fallbacks = config.get("fallback_providers", [])
    if fallbacks:
        overrides["fallback_providers"] = str(_mask_fallback_secrets(fallbacks))

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
    api_keys = [
        ("OPENROUTER_API_KEY", "openrouter"),
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("ANTHROPIC_TOKEN", "anthropic_token"),
        ("NOUS_API_KEY", "nous"),
        ("GOOGLE_API_KEY", "google/gemini"),
        ("GEMINI_API_KEY", "gemini"),
        ("GLM_API_KEY", "glm/zai"),
        ("ZAI_API_KEY", "zai"),
        ("KIMI_API_KEY", "kimi"),
        ("MINIMAX_API_KEY", "minimax"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("DASHSCOPE_API_KEY", "dashscope"),
        ("HF_TOKEN", "huggingface"),
        ("NVIDIA_API_KEY", "nvidia"),
        ("AI_GATEWAY_API_KEY", "ai_gateway"),
        ("OPENCODE_ZEN_API_KEY", "opencode_zen"),
        ("OPENCODE_GO_API_KEY", "opencode_go"),
        ("COMMANDCODE_API_KEY", "commandcode"),
        ("KILOCODE_API_KEY", "kilocode"),
        ("FIRECRAWL_API_KEY", "firecrawl"),
        ("TAVILY_API_KEY", "tavily"),
        ("BROWSERBASE_API_KEY", "browserbase"),
        ("FAL_KEY", "fal"),
        ("ELEVENLABS_API_KEY", "elevenlabs"),
        ("GITHUB_TOKEN", "github"),
    ]

    dotenv_keys = _dotenv_key_names()

    for env_var, label in api_keys:
        val = os.getenv(env_var, "")
        if show_keys and val:
            display = _redact(val)
        else:
            display = "set" if val else "not set"
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
        if not val and label == "openrouter":
            try:
                from agent.credential_pool import load_pool as _load_pool

                if _load_pool("openrouter").has_credentials():
                    display = "set (auth pool)"
            except Exception:
                pass
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
