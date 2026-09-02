"""
Doctor command for hermes CLI.

Diagnoses issues with Hermes Agent setup.
"""

import os
import sys
import subprocess
import shutil
import importlib.util
from pathlib import Path

from hermes_cli.config import (
    detect_install_method,
    get_env_path,
    get_hermes_home,
    get_project_root,
    is_nix_install_method,
    recommended_update_command_for_method,
)
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import display_hermes_home
from hermes_constants import agent_browser_runnable

PROJECT_ROOT = get_project_root()
HERMES_HOME = get_hermes_home()
_DHH = display_hermes_home()  # user-facing display path (e.g. ~/.hermes or ~/.hermes/profiles/coder)

# Load environment variables from ~/.hermes/.env so API key checks work
_env_path = get_env_path()
load_hermes_dotenv(hermes_home=_env_path.parent, project_env=PROJECT_ROOT / ".env")

from hermes_cli.colors import Colors, color
from hermes_cli.models import _HERMES_USER_AGENT
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_constants import OPENROUTER_MODELS_URL
from utils import base_url_host_matches


_PROVIDER_ENV_HINTS = (
    "DEEPINFRA_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
    "NOUS_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "GMI_API_KEY",
    "FIREWORKS_API_KEY",
    "ACTUAL_API_KEY",
    "ACTUAL_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
    "AI_GATEWAY_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
    "COMMANDCODE_API_KEY",
    "XIAOMI_API_KEY",
    "TOKENHUB_API_KEY",
    "TOKENPLAN_API_KEY",
)


from hermes_constants import is_termux as _is_termux


def _python_install_cmd() -> str:
    return "python -m pip install" if _is_termux() else "uv pip install"


def _system_package_install_cmd(pkg: str) -> str:
    if _is_termux():
        return f"pkg install {pkg}"
    if sys.platform == "darwin":
        return f"brew install {pkg}"
    return f"sudo apt install {pkg}"


def _sqlite_upgrade_hint(install_method: str | None = None) -> str:
    """Return an actionable SQLite upgrade hint for this install layout."""
    method = install_method or detect_install_method(PROJECT_ROOT)
    if method == "docker":
        command = recommended_update_command_for_method(method)
        action = f"run `{command}`, then recreate all Hermes containers"
    elif is_nix_install_method(method):
        # The Nix helper is prose guidance, not a literal shell command.
        action = recommended_update_command_for_method(method)
    elif method == "apt":
        action = f"run `{recommended_update_command_for_method(method)}`"
    else:
        action = "run `hermes update`"
    return (
        f"({action}; fixed versions: 3.51.3+ / 3.50.7 / 3.44.6 — "
        "see https://sqlite.org/wal.html#walresetbug)"
    )


def _hermes_database_paths(hermes_home: Path) -> list[tuple[str, Path]]:
    """Return (display name, path) pairs for Hermes-managed SQLite databases."""
    # backup.py owns the canonical list of per-profile stores; reuse it.
    from hermes_cli.backup import _QUICK_STATE_FILES

    entries = [
        (name, hermes_home / name)
        for name in _QUICK_STATE_FILES
        if name.endswith(".db")
    ]
    # Non-default kanban boards each keep their own kanban.db.
    for board_db in sorted((hermes_home / "kanban" / "boards").glob("*/kanban.db")):
        entries.append((str(board_db.relative_to(hermes_home)), board_db))
    return entries


_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _unreadable_reason(db_path: Path) -> str:
    """Explain why a database file could not be read, without opening it.

    ``read_header_bytes_preopen`` collapses every ``OSError`` into ``None``,
    but doctor's job is to say *which* problem it hit. ``stat()`` and
    ``access()`` answer that from directory metadata alone — neither takes a
    file descriptor, so neither can cancel the file's POSIX advisory locks.
    """
    try:
        db_path.stat()
    except OSError as exc:
        return str(exc)
    if not os.access(db_path, os.R_OK):
        return f"permission denied: {db_path}"
    return "file could not be read"


def _read_journal_mode(db_path: Path) -> tuple[str | None, str | None]:
    """Return (journal mode, error) from the file header without opening the database.

    Header byte 18 is 2 for WAL and 1 for a rollback journal. Opening the
    database through the SQLite engine — even read-only — creates -wal/-shm
    sidecar files, which a diagnostic must not do.

    The byte read is routed through ``read_header_bytes_preopen`` rather than
    a bare ``open()``: closing *any* descriptor for a database file cancels
    this process's POSIX advisory locks on it, so a raw read would drop the
    locks a live connection is holding (see ``hermes_cli.sqlite_safe_read``).
    ``run_doctor`` is also called in-process by the dashboard console, which
    holds live ``SessionDB`` connections. The helper refuses in that case and
    the mode is reported as unreadable instead.
    """
    from hermes_cli.sqlite_safe_read import (
        has_live_connection,
        read_header_bytes_preopen,
    )

    header = read_header_bytes_preopen(db_path, length=20)
    if header is None:
        if has_live_connection(db_path):
            return None, "database is open in this process"
        return None, _unreadable_reason(db_path)
    if len(header) == 0:
        return None, "file is empty"
    if len(header) < 20 or not header.startswith(_SQLITE_HEADER_MAGIC):
        return None, "file is not a database"
    if header[18] == 2:
        return "wal", None
    if header[18] == 1:
        return "rollback", None
    return None, f"unrecognized file-format version {header[18]}"


def _format_db_size(db_path: Path) -> str:
    # backup.py owns human-readable size formatting; reuse it (as with
    # _QUICK_STATE_FILES above) and keep only the stat-failure wrap here.
    from hermes_cli.backup import _format_size

    try:
        nbytes = db_path.stat().st_size
    except OSError:
        return "size unknown"
    return _format_size(nbytes)


def _report_database_journal_modes(
    hermes_home: Path | None = None,
    version_info: tuple[int, ...] | None = None,
) -> None:
    """List each database's journal mode; warn on WAL under a vulnerable SQLite."""
    from hermes_state import _wal_reset_repair_hint, is_sqlite_wal_reset_vulnerable

    vulnerable = is_sqlite_wal_reset_vulnerable(version_info)
    home = hermes_home if hermes_home is not None else HERMES_HOME
    try:
        databases = _hermes_database_paths(home)
    except Exception as exc:
        check_warn(f"Could not list Hermes databases: {exc}")
        return
    exposed = []
    for name, path in databases:
        if not path.is_file():
            continue
        mode, error = _read_journal_mode(path)
        size = _format_db_size(path)
        if error is not None:
            if vulnerable:
                check_warn(
                    f"{name}: journal mode could not be read",
                    f"({error}; cannot rule out WAL exposure)",
                )
            else:
                check_info(f"{name}: journal mode could not be read ({error})")
        elif mode == "wal":
            if vulnerable:
                exposed.append(name)
                check_warn(
                    f"{name} is in WAL mode ({size})",
                    "(exposed to the WAL-reset bug until SQLite is upgraded)",
                )
            else:
                check_info(f"{name}: WAL journal mode ({size})")
        elif vulnerable:
            check_info(f"{name}: rollback journal mode ({size}, not exposed)")
        else:
            check_info(f"{name}: rollback journal mode ({size})")
    if exposed:
        check_info(f"To clear the exposure: {_wal_reset_repair_hint()}")


def _safe_which(cmd: str) -> str | None:
    """shutil.which wrapper resilient to platform monkeypatching in tests."""
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _termux_browser_setup_steps(node_installed: bool) -> list[str]:
    steps: list[str] = []
    step = 1
    if not node_installed:
        steps.append(f"{step}) pkg install nodejs")
        step += 1
    steps.append(f"{step}) npm install -g agent-browser")
    steps.append(f"{step + 1}) agent-browser install")
    return steps


def _termux_install_all_fallback_notes() -> list[str]:
    return [
        "Termux install profile: use .[termux-all] for broad compatibility (installer default on Termux).",
        "Matrix E2EE extra is excluded on Termux (python-olm currently fails to build).",
        "Local faster-whisper extra is excluded on Termux (ctranslate2/av build path unavailable).",
        "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY).",
    ]


def _has_provider_env_config(content: str) -> bool:
    """Return True when ~/.hermes/.env contains provider auth/base URL settings."""
    return any(key in content for key in _PROVIDER_ENV_HINTS)


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


def _is_kanban_worker_env_gate(item: dict) -> bool:
    """Return True when Kanban is unavailable only because this is not a worker process."""
    if item.get("name") != "kanban":
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False

    tools = item.get("tools") or []
    return bool(tools) and all(str(tool).startswith("kanban_") for tool in tools)


def _doctor_tool_availability_detail(toolset: str) -> str:
    """Optional explanatory suffix for toolsets whose doctor status needs context."""
    if toolset == "kanban" and not os.environ.get("HERMES_KANBAN_TASK"):
        return "(runtime-gated; loaded only for dispatcher-spawned workers)"
    return ""


def _doctor_web_capability_rows() -> list[tuple[str, str, str]]:
    """Return doctor rows for web search/extract provider readiness (#78412).

    Each row is ``(status, label, detail)`` where *status* is ``ok`` or ``warn``.
    Uses the same active-provider resolvers as the tools, but reports readiness
    from ``is_available()`` so an explicitly selected but unconfigured backend
    does not look healthy.
    """
    rows: list[tuple[str, str, str]] = []
    try:
        from agent.web_search_registry import (
            get_active_extract_provider,
            get_active_search_provider,
        )
        from tools.web_tools import _ensure_web_plugins_loaded, _provider_is_ready

        # Doctor runs in a fresh process — bundled web providers register
        # during plugin discovery, which nothing has triggered yet here.
        # Without this the registry is empty and every row reads
        # "no provider selected or registered" (idempotent, cheap on rerun).
        _ensure_web_plugins_loaded()
    except Exception:
        return rows

    for capability, getter in (
        ("web search", get_active_search_provider),
        ("web extract", get_active_extract_provider),
    ):
        try:
            provider = getter()
        except Exception:
            provider = None
        if provider is None:
            rows.append(
                (
                    "warn",
                    capability,
                    "(no provider selected or registered)",
                )
            )
            continue
        name = getattr(provider, "name", None) or type(provider).__name__
        if _provider_is_ready(provider):
            rows.append(("ok", capability, f"({name})"))
        else:
            rows.append(
                (
                    "warn",
                    capability,
                    f"({name} selected; provider not configured)",
                )
            )
    return rows

def _apply_doctor_tool_availability_overrides(available: list[str], unavailable: list[dict]) -> tuple[list[str], list[dict]]:
    """Adjust runtime-gated tool availability for doctor diagnostics."""
    updated_available = list(available)
    updated_unavailable = []
    for item in unavailable:
        name = item.get("name")
        if _is_kanban_worker_env_gate(item):
            if "kanban" not in updated_available:
                updated_available.append("kanban")
            continue
        if name == "honcho" and _honcho_is_configured_for_doctor():
            if "honcho" not in updated_available:
                updated_available.append("honcho")
            continue
        updated_unavailable.append(item)
    return updated_available, updated_unavailable


def _has_healthy_oauth_fallback_for_apikey_provider(provider_label: str) -> bool:
    """Return True when a direct API-key probe failure is non-blocking.

    Some provider families support both a direct API-key path and a separate
    OAuth runtime path. When the OAuth path is already healthy, doctor should
    still show a failed API-key connectivity row, but it should not promote
    that direct-key problem into the final blocking summary.
    """
    normalized = (provider_label or "").strip().lower()
    if normalized == "minimax":
        try:
            from hermes_cli.auth import get_minimax_oauth_auth_status
            return bool((get_minimax_oauth_auth_status() or {}).get("logged_in"))
        except Exception:
            return False
    if normalized == "xai":
        try:
            from hermes_cli.auth import get_xai_oauth_auth_status
            return bool((get_xai_oauth_auth_status() or {}).get("logged_in"))
        except Exception:
            return False
    return False


def check_ok(text: str, detail: str = ""):
    print(f"  {color('✓', Colors.GREEN)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_warn(text: str, detail: str = ""):
    print(f"  {color('⚠', Colors.YELLOW)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_fail(text: str, detail: str = ""):
    print(f"  {color('✗', Colors.RED)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


def _doctor_memory_config(hermes_home: Path | None = None) -> dict:
    """Return the effective memory section used by doctor diagnostics."""
    home = hermes_home if hermes_home is not None else HERMES_HOME
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        config_path = home / "config.yaml"
        if not config_path.exists():
            return {}
        config = _expand_env_vars(read_user_config_raw(config_path))
        try:
            from hermes_cli import managed_scope

            config = managed_scope.apply_managed_overlay(config)
        except Exception:
            pass
        section = config.get("memory") if isinstance(config, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# ── state.db health/stats thresholds (advisory only — module constants,
# deliberately NOT config: doctor warnings are guidance, not policy) ──
STATE_DB_SIZE_WARN_BYTES = 1 * 1024 * 1024 * 1024   # 1 GiB logical size


# Shared byte formatter, aliased to the name this module's three rendering
# call sites already use.
from hermes_cli.sizefmt import format_bytes as _human_bytes


def _render_state_db_stats(stats: dict, holders=None) -> list:
    """Turn a collect_state_db_stats() dict into doctor output lines.

    Returns a list of ``(kind, text, detail)`` tuples where kind is one of
    'info' / 'warn'. Pure formatting — no I/O — so it is unit-testable
    without spawning the doctor CLI. Tolerates None in every field.
    """
    lines: list = []
    stats = stats or {}

    logical = stats.get("logical_size_bytes")
    wal = stats.get("wal_size_bytes")
    freelist = stats.get("freelist_count")

    size_bits = []
    if logical is not None:
        size_bits.append(f"logical size {_human_bytes(logical)}")
    if stats.get("page_count") is not None:
        size_bits.append(f"{stats['page_count']:,} pages")
    if freelist is not None:
        size_bits.append(f"{freelist:,} free")
    if wal is not None:
        size_bits.append(f"WAL {_human_bytes(wal)}")
    if size_bits:
        lines.append(("info", "state.db " + ", ".join(size_bits), ""))

    row_bits = []
    if stats.get("messages") is not None:
        row_bits.append(f"{stats['messages']:,} messages")
    if stats.get("sessions") is not None:
        row_bits.append(f"{stats['sessions']:,} sessions")
    if stats.get("journal_mode"):
        row_bits.append(f"journal_mode={stats['journal_mode']}")
    if holders is not None:
        row_bits.append(f"{holders} process(es) holding the DB open")
    if row_bits:
        lines.append(("info", ", ".join(row_bits), ""))

    fts = stats.get("fts_tables")
    if fts:
        present = [t for t, ok in fts.items() if ok]
        lines.append((
            "info",
            "FTS tables: " + (", ".join(present) if present else "none"),
            "",
        ))

    deferral = stats.get("fts_rebuild_deferral")
    if isinstance(deferral, dict):
        attempts = deferral.get("attempts")
        pids = deferral.get("holder_pids") or []
        lines.append((
            "warn",
            f"state.db FTS repair is blocked after {attempts or '?'} "
            f"deferral(s) by PID(s) {pids or 'unknown'}",
            "(stop the listed processes, then run 'hermes sessions "
            "optimize-storage' with the gateway stopped)",
        ))

    # Advisory: oversized database. Suggest auto_prune, and — when the v23
    # FTS rebuild is pending OR the DB still carries the legacy inline
    # trigram layout (fts_storage_version marker absent) — the offline
    # optimize-storage pass that migrates/compacts the FTS indexes.
    if logical is not None and logical > STATE_DB_SIZE_WARN_BYTES:
        detail = (
            "consider enabling sessions.auto_prune in config.yaml "
            "to bound growth"
        )
        legacy_trigram = (
            fts is not None
            and fts.get("messages_fts_trigram")
            and stats.get("fts_storage_version") is None
        )
        if stats.get("fts_rebuild_pending") or legacy_trigram:
            detail += (
                "; run 'hermes sessions optimize-storage' offline "
                "(with the gateway stopped) to compact FTS storage"
            )
        lines.append((
            "warn",
            f"state.db is large ({_human_bytes(logical)})",
            f"({detail})",
        ))

    # WAL runaway is deliberately NOT warned here: the pre-existing WAL
    # check later in the state.db section already warns above 50 MB and
    # offers a checkpoint via --fix; a second warning at a higher threshold
    # would only duplicate it.

    return lines


def _section(title: str) -> None:
    """Print a doctor section banner: blank line + bold cyan ◆ title."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _fail_and_issue(text: str, detail: str, fix: str, issues: list[str]) -> None:
    """Emit a check_fail and append the corresponding fix instruction."""
    check_fail(text, detail)
    issues.append(fix)


# Deprecated / legacy config keys still read for back-compat. Doctor surfaces
# them as non-failing warnings with the modern replacement — it does not
# auto-migrate or delete (migrations live in config.py version steps).
_DEPRECATED_CONFIG_KEYS: tuple[tuple[str, str, str], ...] = (
    # (section, key, replacement)
    ("display", "tool_progress_overrides", "display.platforms"),
    ("delegation", "max_async_children", "delegation.max_concurrent_children"),
)

# compression.summary_* → auxiliary.compression (model/provider/base_url)
_DEPRECATED_COMPRESSION_SUMMARY_KEYS: tuple[str, ...] = (
    "summary_model",
    "summary_provider",
    "summary_base_url",
)

# Deprecated env vars (checked in the .env file, not process env, so config→env
# bridges like terminal.cwd → TERMINAL_CWD do not false-positive).
_DEPRECATED_ENV_VARS: tuple[tuple[str, str], ...] = (
    # HERMES_TOOL_PROGRESS is fully unsupported since the v12 config support
    # floor removed its only consumer (the v3→4 migration) — it is silently
    # ignored. HERMES_TOOL_PROGRESS_MODE is still read by the gateway as a
    # back-compat fallback but remains deprecated.
    ("HERMES_TOOL_PROGRESS", "display.tool_progress in config.yaml — ignored/unsupported since config floor v12"),
    ("HERMES_TOOL_PROGRESS_MODE", "display.tool_progress in config.yaml"),
    ("TERMINAL_CWD", "terminal.cwd in config.yaml"),
    ("MESSAGING_CWD", "terminal.cwd in config.yaml"),
    ("QQ_HOME_CHANNEL", "QQBOT_HOME_CHANNEL"),
    ("QQ_HOME_CHANNEL_NAME", "QQBOT_HOME_CHANNEL_NAME"),
)


def collect_deprecated_config_keys(raw_config: dict | None) -> list[tuple[str, str]]:
    """Return ``(legacy_path, replacement)`` for deprecated keys present in *raw_config*.

    Only keys that appear in the on-disk YAML are reported (raw file load, not
    merged defaults). Empty containers still count — presence of the legacy
    key is the signal that the user should migrate.
    """
    findings: list[tuple[str, str]] = []
    if not isinstance(raw_config, dict):
        return findings

    for section, key, replacement in _DEPRECATED_CONFIG_KEYS:
        section_val = raw_config.get(section)
        if isinstance(section_val, dict) and key in section_val:
            findings.append((f"{section}.{key}", replacement))

    compression = raw_config.get("compression")
    if isinstance(compression, dict):
        for key in _DEPRECATED_COMPRESSION_SUMMARY_KEYS:
            if key in compression:
                findings.append((f"compression.{key}", "auxiliary.compression"))

    return findings


def collect_deprecated_env_vars(env_map: dict | None) -> list[tuple[str, str]]:
    """Return ``(legacy_env, replacement)`` for deprecated vars present in *env_map*.

    *env_map* should come from the on-disk ``.env`` (e.g. ``load_env()``), not
    ``os.environ``, so bridged runtime vars do not trigger false positives.
    """
    findings: list[tuple[str, str]] = []
    if not isinstance(env_map, dict):
        return findings
    for name, replacement in _DEPRECATED_ENV_VARS:
        val = env_map.get(name)
        if val is not None and str(val).strip() != "":
            findings.append((name, replacement))
    return findings


def collect_relay_plugin_cutover_findings(
    raw_config: dict | None,
    env_map: dict | None,
) -> list[tuple[str, str]]:
    """Return actionable findings for the removed Hermes Relay plugin."""
    from hermes_cli.relay_plugin_cutover import (
        LEGACY_RELAY_EXPORT_ENV_VARS,
        RELAY_PLUGINS_CONFIG_ENV,
        configured_legacy_relay_env_vars,
        legacy_relay_plugin_keys,
    )

    findings: list[tuple[str, str]] = []
    if isinstance(raw_config, dict):
        plugins = raw_config.get("plugins")
        if isinstance(plugins, dict):
            for key in legacy_relay_plugin_keys(plugins.get("enabled")):
                findings.append(
                    (
                        f"plugins.enabled: {key}",
                        f"remove it and configure {RELAY_PLUGINS_CONFIG_ENV}",
                    )
                )

    effective_env = dict(env_map or {})
    # Fall through to process-level env ONLY when no explicit env_map was
    # given: run_doctor passes None and wants live-process vars included, but
    # callers (and tests) that hand in an explicit map are describing a
    # complete environment — merging os.environ on top breaks hermeticity on
    # any box that exports legacy relay vars (10-vs-2 findings, Aug 2026).
    if env_map is None:
        for name in (*LEGACY_RELAY_EXPORT_ENV_VARS, RELAY_PLUGINS_CONFIG_ENV):
            if name not in effective_env and os.environ.get(name) is not None:
                effective_env[name] = os.environ[name]
    if not str(effective_env.get(RELAY_PLUGINS_CONFIG_ENV, "")).strip():
        for name in configured_legacy_relay_env_vars(effective_env):
            findings.append(
                (
                    name,
                    f"move exporter settings to {RELAY_PLUGINS_CONFIG_ENV}; "
                    "this variable is now ignored",
                )
            )
    return findings


def report_deprecated_config_and_env(
    raw_config: dict | None = None,
    env_map: dict | None = None,
) -> list[tuple[str, str]]:
    """Emit non-failing doctor warnings for deprecated config keys and env vars.

    Returns the list of ``(legacy, replacement)`` findings that were reported
    (empty when nothing deprecated is present). Does not mutate config/env and
    does not append to the blocking ``issues`` list.
    """
    deprecated = collect_deprecated_config_keys(raw_config)
    deprecated.extend(collect_deprecated_env_vars(env_map))
    relay_cutover = collect_relay_plugin_cutover_findings(raw_config, env_map)
    findings = deprecated + relay_cutover
    if not findings:
        check_ok("No deprecated config keys or env vars")
        return findings

    for legacy, replacement in deprecated:
        check_warn(
            f"Deprecated: {legacy}",
            f"(use {replacement} instead)",
        )
        check_info(f"Replace {legacy} → {replacement} (warn-only; not auto-migrated here)")
    for legacy, replacement in relay_cutover:
        check_warn(
            f"Breaking Relay migration: {legacy}",
            f"({replacement})",
        )
        check_info(f"Migrate {legacy}: {replacement}")
    return findings


def _enabled_cli_toolsets_for_doctor() -> set[str] | None:
    """Return toolsets enabled for the CLI, or None if config resolution fails."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        return {str(toolset) for toolset in _get_platform_tools(load_config() or {}, "cli")}
    except Exception:
        return None


def _missing_api_key_toolsets_for_summary(unavailable: list[dict]) -> list[dict]:
    """Filter unavailable API-key toolsets to those enabled for the CLI."""
    api_key_unavailable = [
        item for item in unavailable
        if item.get("missing_vars") or item.get("env_vars")
    ]
    enabled_toolsets = _enabled_cli_toolsets_for_doctor()
    if enabled_toolsets is None:
        return api_key_unavailable
    return [
        item for item in api_key_unavailable
        if str(item.get("name") or "") in enabled_toolsets
    ]


def _read_pyproject_version() -> str | None:
    """Read the ``version = "..."`` from ``pyproject.toml`` at the project root.

    Returns None when running from an installed wheel (no pyproject.toml ships
    with the package) or when the file can't be parsed. Reads only the
    ``[project]`` version, ignoring any version strings that appear in other
    tables.
    """
    pyproject = PROJECT_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            value = line.split("=", 1)[1]
            value = value.split("#", 1)[0].strip().strip("\"'")
            return value or None
    return None


def _check_version_consistency(issues: list[str]) -> None:
    """Verify pyproject.toml version matches hermes_cli.__version__.

    A git conflict resolution (reset/merge) can revert one file without the
    other, leaving ``hermes --version`` reporting a stale version while
    ``pyproject.toml`` is current. Detect that drift so users can re-sync.
    Silent no-op for installed wheels where pyproject.toml isn't present.
    """
    try:
        from hermes_cli import __version__ as init_version
    except Exception:
        return
    pyproject_version = _read_pyproject_version()
    if pyproject_version is None:
        # Installed wheel or unreadable pyproject — nothing to cross-check.
        return
    if pyproject_version == init_version:
        check_ok("Version files consistent", f"({init_version})")
    else:
        _fail_and_issue(
            "Version mismatch between source files",
            f"(pyproject.toml {pyproject_version} != hermes_cli/__init__.py {init_version})",
            "Re-sync version files (e.g. run 'hermes update', or set "
            "hermes_cli/__init__.py __version__ to match pyproject.toml)",
            issues,
        )


def _check_s6_supervision(issues: list[str]) -> None:
    """Inside a container under our s6 /init, surface what s6 sees.

    Runs as a counterpart to :func:`_check_gateway_service_linger` for
    the systemd-on-host case. No-op everywhere except in the s6
    container so host runs aren't cluttered with irrelevant output.

    Reports:
      - Whether the main-hermes and dashboard static services are up
      - How many per-profile gateway slots are registered (via
        ``S6ServiceManager.list_profile_gateways()``) and how many are
        currently supervised as ``up``
    """
    try:
        from hermes_cli.service_manager import (
            S6ServiceManager,
            detect_service_manager,
        )
    except Exception:
        return

    if detect_service_manager() != "s6":
        return

    _section("s6 Supervision")

    mgr = S6ServiceManager()

    # Static services. They live under /run/service/ via s6-rc symlinks,
    # so the same s6-svstat probe works.
    for static in ("main-hermes", "dashboard"):
        if mgr.is_running(static):
            check_ok(f"{static}: up")
        else:
            check_info(f"{static}: down (expected if not enabled via env)")

    profiles = mgr.list_profile_gateways()
    if not profiles:
        check_info("No per-profile gateways registered yet — create one with `hermes profile create <name>`")
        return

    up_count = sum(1 for p in profiles if mgr.is_running(f"gateway-{p}"))
    check_ok(
        f"Per-profile gateways: {up_count}/{len(profiles)} supervised up"
        + (f" ({', '.join(sorted(profiles))})" if len(profiles) <= 8 else "")
    )


def check_certificates(should_fix: bool = False, issues: "list | None" = None) -> None:
    """Verify the certifi CA bundle is loadable.

    Surfaces the SSLConfigurationError user-friendly path before they hit
    a wall of tracebacks on the first outbound HTTPS call.

    With ``--fix``, a broken bundle (missing/corrupt ``cacert.pem`` — e.g.
    after a brew Python upgrade rebuilt the venv, #29866) is repaired by
    force-reinstalling certifi into THIS interpreter's environment and
    re-verifying.
    """
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback
        from agent.errors import SSLConfigurationError
    except Exception as e:
        check_warn("SSL certificate check skipped", str(e))
        return

    try:
        verify_ca_bundle_with_fallback()
        check_ok("SSL CA certificate bundle is valid")
        return
    except SSLConfigurationError as e:
        first_error = str(e)
    except Exception as e:
        check_warn("SSL certificate check skipped", str(e))
        return

    if not should_fix:
        check_fail("SSL CA certificate bundle is broken", first_error)
        if issues is not None:
            issues.append(
                "Repair the CA bundle: run `hermes doctor --fix`, or "
                f"`{sys.executable} -m pip install --force-reinstall certifi`"
            )
        return

    # --fix: force-reinstall certifi into the running interpreter's env and
    # re-verify. importlib caches are invalidated so certifi.where() resolves
    # the fresh install without a process restart.
    check_fail("SSL CA certificate bundle is broken", first_error)
    print("    → Repairing: force-reinstalling certifi...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "certifi"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        check_fail("certifi repair could not run pip", str(exc))
        if issues is not None:
            issues.append(
                f"Reinstall certifi manually: {sys.executable} -m pip install "
                "--force-reinstall certifi"
            )
        return
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-500:]
        check_fail("certifi reinstall failed", tail)
        if issues is not None:
            issues.append(
                f"Reinstall certifi manually: {sys.executable} -m pip install "
                "--force-reinstall certifi"
            )
        return

    # Drop any cached certifi module so where() re-resolves the new bundle.
    import importlib
    for mod_name in [m for m in sys.modules if m == "certifi" or m.startswith("certifi.")]:
        sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()

    try:
        verify_ca_bundle_with_fallback()
        check_ok("SSL CA certificate bundle repaired (certifi reinstalled)")
    except SSLConfigurationError as e:
        check_fail("SSL CA certificate bundle still broken after reinstall", str(e))
        if issues is not None:
            issues.append(
                "certifi reinstall did not restore the CA bundle — check for a "
                "custom CA env var (SSL_CERT_FILE/REQUESTS_CA_BUNDLE) pointing "
                "at a missing file, or recreate the venv."
            )


def _check_gateway_service_linger(issues: list[str]) -> None:
    """Warn when a systemd user gateway service will stop after logout.

    Skipped inside a container running under s6 — the linger concept
    (user-systemd surviving SSH logout) doesn't apply there, and the
    s6 supervision state is surfaced separately by
    ``_check_s6_supervision``.
    """
    try:
        from hermes_cli.gateway import (
            get_systemd_linger_status,
            get_systemd_unit_path,
            is_linux,
        )
        from hermes_cli.service_manager import detect_service_manager
    except Exception as e:
        check_warn("Gateway service linger", f"(could not import gateway helpers: {e})")
        return

    if not is_linux():
        return

    # Inside a container under our s6 /init, _check_s6_supervision
    # reports the live supervision state; the linger warning would be
    # confusing here (no systemd, no logout, no "lingering" concept).
    if detect_service_manager() == "s6":
        return

    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        return

    _section("Gateway Service")
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        check_ok("Systemd linger enabled", "(gateway service survives logout)")
    elif linger_enabled is False:
        check_warn("Systemd linger disabled", "(gateway may stop after logout)")
        check_info("Run: sudo loginctl enable-linger $USER")
        issues.append("Enable linger for the gateway user service: sudo loginctl enable-linger $USER")
    else:
        check_warn("Could not verify systemd linger", f"({linger_detail})")


_APIKEY_PROVIDERS_CACHE: list | None = None


def _build_apikey_providers_list() -> list:
    """Build the API-key provider health-check list once and cache it.

    Tuple format: (name, env_vars, default_url, base_env, supports_models_endpoint)
    Base list augmented with any ProviderProfile with auth_type="api_key" not
    already present — adding plugins/model-providers/<name>/ is sufficient to get into doctor.
    """
    _static = [
        ("Z.AI / GLM",      ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True),
        ("Kimi / Moonshot",  ("KIMI_API_KEY",),                              "https://api.moonshot.ai/v1/models",   "KIMI_BASE_URL", True),
        ("StepFun Step Plan", ("STEPFUN_API_KEY",),                          "https://api.stepfun.ai/step_plan/v1/models", "STEPFUN_BASE_URL", True),
        ("Kimi / Moonshot (China)", ("KIMI_CN_API_KEY",),                    "https://api.moonshot.cn/v1/models",   None, True),
        ("Arcee AI",         ("ARCEEAI_API_KEY",),                           "https://api.arcee.ai/api/v1/models",  "ARCEE_BASE_URL", True),
        ("GMI Cloud",        ("GMI_API_KEY",),                               "https://api.gmi-serving.com/v1/models", "GMI_BASE_URL", True),
        ("DeepSeek",         ("DEEPSEEK_API_KEY",),                          "https://api.deepseek.com/v1/models",  "DEEPSEEK_BASE_URL", True),
        ("Hugging Face",     ("HF_TOKEN",),                                  "https://router.huggingface.co/v1/models", "HF_BASE_URL", True),
        ("NVIDIA NIM",       ("NVIDIA_API_KEY",),                            "https://integrate.api.nvidia.com/v1/models", "NVIDIA_BASE_URL", True),
        ("Alibaba/DashScope", ("DASHSCOPE_API_KEY",),                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True),
        # MiniMax global: /v1 endpoint supports /models.
        ("MiniMax",          ("MINIMAX_API_KEY",),                           "https://api.minimax.io/v1/models",    "MINIMAX_BASE_URL", True),
        # MiniMax CN: /v1 endpoint does NOT support /models (returns 404).
        ("MiniMax (China)",  ("MINIMAX_CN_API_KEY",),                        "https://api.minimaxi.com/v1/models",  "MINIMAX_CN_BASE_URL", False),
        ("Vercel AI Gateway", ("AI_GATEWAY_API_KEY",),                       "https://ai-gateway.vercel.sh/v1/models", "AI_GATEWAY_BASE_URL", True),
        ("Kilo Code",        ("KILOCODE_API_KEY",),                          "https://api.kilo.ai/api/gateway/models", "KILOCODE_BASE_URL", True),
        ("OpenCode Zen",     ("OPENCODE_ZEN_API_KEY",),                      "https://opencode.ai/zen/v1/models",  "OPENCODE_ZEN_BASE_URL", True),
        # OpenCode Go has no shared /models endpoint; skip the health check.
        ("OpenCode Go",      ("OPENCODE_GO_API_KEY",),                       None,                                  "OPENCODE_GO_BASE_URL", False),
    ]
    _known_names = {t[0] for t in _static}
    # Also index by profile canonical name so profiles without display_name
    # don't create duplicate entries for providers already in the static list.
    _known_canonical: set[str] = set()
    _name_to_canonical = {
        "Z.AI / GLM": "zai", "Kimi / Moonshot": "kimi-coding",
        "StepFun Step Plan": "stepfun", "Kimi / Moonshot (China)": "kimi-coding-cn",
        "Arcee AI": "arcee", "GMI Cloud": "gmi", "DeepSeek": "deepseek",
        "Hugging Face": "huggingface", "NVIDIA NIM": "nvidia",
        "Alibaba/DashScope": "alibaba", "MiniMax": "minimax",
        "MiniMax (China)": "minimax-cn", "Vercel AI Gateway": "ai-gateway",
        "Kilo Code": "kilocode", "OpenCode Zen": "opencode-zen",
        "OpenCode Go": "opencode-go",
    }
    for _label, _canonical in _name_to_canonical.items():
        _known_canonical.add(_canonical)
    # Providers that already have a dedicated health check above the generic
    # API-key loop (with custom headers/auth). Skip their pluggable profiles
    # here so the generic Bearer-auth loop doesn't run a duplicate, broken
    # check (e.g. Anthropic native API requires x-api-key, not Bearer).
    _dedicated_canonical = {"anthropic", "openrouter", "bedrock"}
    _known_canonical.update(_dedicated_canonical)
    try:
        from providers import list_providers
        from providers.base import ProviderProfile as _PP
        try:
            from hermes_cli.providers import normalize_provider as _normalize_provider
        except Exception:  # pragma: no cover - normalization is best-effort
            def _normalize_provider(_name: str) -> str:
                return (_name or "").strip().lower()
        for _pp in list_providers():
            if not isinstance(_pp, _PP) or _pp.auth_type != "api_key" or not _pp.env_vars:
                continue
            _label = _pp.display_name or _pp.name
            if _label in _known_names or _pp.name in _known_canonical:
                continue
            _candidates = {_normalize_provider(_pp.name)}
            for _alias in (_pp.aliases or ()):
                _candidates.add(_normalize_provider(_alias))
            if _candidates & _dedicated_canonical:
                continue
            # Separate API-key vars from base-URL override vars — the health-check
            # loop sends the first found value as Authorization: Bearer, so a URL
            # string must never be picked.
            _key_vars = tuple(
                v for v in _pp.env_vars
                if not v.endswith("_BASE_URL") and not v.endswith("_URL")
            )
            _base_var = next(
                (v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")),
                None,
            )
            if not _key_vars:
                continue
            _models_url = (
                (_pp.models_url or (_pp.base_url.rstrip("/") + "/models"))
                if _pp.base_url else None
            )
            _hc = getattr(_pp, "supports_health_check", True)
            _static.append((_label, _key_vars, _models_url, _base_var, _hc))
    except Exception:
        pass
    return _static


def managed_scope_check() -> None:
    """Report the active managed scope (resolved dir + pinned key counts).

    Silent when no managed scope is present. When the managed directory was
    resolved from the HERMES_MANAGED_DIR override (rather than the system
    default), that is surfaced too — a redirected scope is the documented
    foot-gun (see docs/design/managed-scope.md §7) and an operator should see it.
    """
    try:
        from hermes_cli import managed_scope
        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — diagnostics must never crash
        return
    if managed_dir is None:
        return
    n_cfg = len(managed_scope.managed_config_keys())
    n_env = len(managed_scope.load_managed_env())
    check_ok(
        f"Managed scope active: {n_cfg} config key(s), {n_env} env key(s) "
        f"pinned by {managed_dir}"
    )
    if os.environ.get("HERMES_MANAGED_DIR", "").strip():
        check_info(f"managed dir set via HERMES_MANAGED_DIR={managed_dir}")


def check_macos_tcc_grants() -> None:
    """Check macOS TCC grant persistence for a locally-built desktop bundle.

    TCC keys permission grants (Screen Recording, Full Disk Access,
    Accessibility, ...) to the app's code-signing requirement. A bundle
    signed with the pre-#73681 cdhash-pinned ad-hoc identity gets a new DR on
    every rebuild, so all grants silently stop matching — and the stale row
    keeps the System Settings toggle ON while macOS re-prompts on every
    capture (issue #86385).

    Post-#73681 builds pin ``designated => identifier "com.nousresearch.hermes"``
    (no cdhash), so new grants survive rebuilds — but grants made to older
    binaries remain stale until re-granted once. The stale state is not
    directly readable (TCC.db needs Full Disk Access), so this check reports
    the DR class and, when the DR is stable, prints the exact one-time repair.
    Silent on non-macOS and when no desktop bundle is installed.
    """
    if sys.platform != "darwin":
        return
    app = _desktop_app_bundle()
    if app is None:
        return
    dr = _macos_desktop_dr(app)
    if not dr:
        check_warn(
            "macOS TCC grant check",
            "(could not read code-signing requirement of the desktop bundle)",
        )
        return
    # The DR string is the only readable signal — TCC.db itself needs Full
    # Disk Access. A cdhash anchor marks the pre-#73681 ad-hoc identity
    # (rebuild ⇒ new cdhash ⇒ stale grants); its absence marks identifier-
    # pinned. Treat the match as a proxy for the signing class, not a
    # contract on DR wording.
    if "cdhash" in dr.lower():
        check_warn(
            "macOS TCC grants will reset after every update",
            "the desktop bundle's designated requirement is cdhash-pinned "
            "(pre-#73681 build) — rebuilds invalidate all permission grants. "
            "Run `hermes update` to get the stable identifier-pinned signing "
            "identity, then re-grant permissions once.",
        )
        return
    if "certificate" in dr.lower():
        # Certificate-anchored DR (hermes desktop --setup-tcc-identity, or a
        # notarized release build): the strongest anchor TCC can key on.
        check_ok(
            "macOS TCC signing identity is stable",
            "(certificate-anchored DR; grants survive rebuilds)",
        )
    else:
        check_ok(
            "macOS TCC signing identity is stable",
            "(identifier-pinned DR; grants survive rebuilds — for the strongest "
            "anchor, see `hermes desktop --setup-tcc-identity`)",
        )
    check_info(
        "If macOS still re-prompts for permissions (toggle shows ON): the stored "
        "grant is stale — run `tccutil reset ScreenCapture com.nousresearch.hermes` "
        "(repeat per affected service), toggle it ON in System Settings, then "
        "fully quit & relaunch Hermes once."
    )


def _desktop_app_bundle() -> Path | None:
    """Locate the locally-built desktop app bundle, if any.

    Mirrors the install layout the self-updater produces
    (``apps/desktop/release/mac-<arch>/Hermes.app``) — the only layout whose
    ad-hoc re-signed bundle can invalidate TCC grants. When multiple arch
    trees coexist (stale cross-build), the newest wins, matching
    ``_desktop_packaged_executable``'s selection. ``/Applications/Hermes.app``
    is deliberately not probed: it is the separately-signed Hermes-Setup
    launcher (``com.nousresearch.hermes.setup``, certificate-anchored), whose
    grants are stable by construction and unaffected by rebuilds.
    """
    root = Path(__file__).resolve().parents[1]
    release_dir = root / "apps" / "desktop" / "release"
    candidates = [p for p in release_dir.glob("mac*/Hermes.app") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _macos_desktop_dr(app: Path) -> str | None:
    """Return the bundle's designated requirement string, or None on failure."""
    codesign = shutil.which("codesign")
    if not codesign:
        return None
    try:
        proc = subprocess.run(
            [codesign, "-d", "--requirements", "-", str(app)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Never let a hanging codesign abort the whole doctor run — the
        # caller falls through to its "could not read" warning.
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def check_macos_tcc_anchor(should_fix: bool = False) -> None:
    """Report (and optionally install) the dylib-complete TCC anchor (#95596).

    Silent on non-macOS and for interpreters that are not uv-managed.  Never
    raises — a failed check must not crash doctor.  Install is gated by the
    module's pre-install boot probe, so ``--fix`` cannot brick the CLI.
    """
    try:
        from hermes_cli import macos_tcc_anchor as tcc

        status, detail = tcc.tcc_anchor_state()
        if status == "skip":
            return
        if status == "active":
            check_ok("macOS TCC anchor active", f"({detail})")
            return
        if should_fix:
            anchored = tcc.ensure_tcc_anchor()
            if anchored is not None:
                check_ok("macOS TCC anchor installed", f"({anchored})")
                return
        check_warn(
            "macOS TCC anchor missing" if status == "missing" else "macOS TCC anchor stale",
            f"({detail})",
        )
    except Exception as e:  # diagnostics must never crash
        check_warn("macOS TCC anchor check failed", f"({e})")


def check_macos_full_disk_access() -> None:
    """One-grant guidance: Full Disk Access silences every folder prompt.

    macOS TCC prompts per-category (Desktop, then Downloads, then Documents,
    ...), so first-run agents drip-feed permission dialogs as they touch each
    folder. ONE Full Disk Access grant covers all of them, permanently — and
    with the stable signing identities now in place (#73681/#95091/#95131),
    it survives updates too. This check probes whether the terminal context
    already has FDA and, when it doesn't, prints the exact one-switch setup
    with the System Settings deep link.

    Probe: readability of ``~/Library/Application Support/com.apple.TCC`` —
    the TCC database directory itself is FDA-gated, readable ONLY with the
    grant, and (critically) probing it with os.access/listdir does NOT
    trigger a prompt: TCC prompts fire for protected-CATEGORY paths (Desktop
    etc.), while the TCC dir simply returns EPERM without one. Silent on
    non-macOS.
    """
    if sys.platform != "darwin":
        return
    tcc_dir = Path.home() / "Library" / "Application Support" / "com.apple.TCC"
    try:
        os.listdir(tcc_dir)
        has_fda = True
    except PermissionError:
        has_fda = False
    except OSError:
        # Missing dir / other error: can't tell — stay silent rather than
        # nag on an indeterminate probe.
        return
    if has_fda:
        check_ok(
            "macOS Full Disk Access granted",
            "(no per-folder permission prompts will occur)",
        )
        return
    check_info(
        "One switch silences all macOS folder prompts: grant your terminal "
        "app Full Disk Access and Hermes will never trip per-folder dialogs "
        "(Desktop/Downloads/Documents/...) again. Open: System Settings → "
        "Privacy & Security → Full Disk Access — or run:\n"
        "      open \"x-apple.systempreferences:com.apple.preference"
        ".security?Privacy_AllFiles\"\n"
        "    then enable your terminal (and Hermes.app if you use Desktop), "
        "and restart them once. With Hermes' stable signing identities the "
        "grant survives every update."
    )


def run_doctor(args):
    """Run diagnostic checks."""
    should_fix = getattr(args, 'fix', False)
    ack_target = getattr(args, 'ack', None)

    # Doctor runs from the interactive CLI, so CLI-gated tool availability
    # checks (like cronjob management) should see the same context as `hermes`.
    os.environ.setdefault("HERMES_INTERACTIVE", "1")

    # Handle `hermes doctor --ack <id>` as a fast path. Persist the ack and
    # return without running the rest of the diagnostics — the user has
    # already seen the advisory and just wants to silence it.
    if ack_target:
        from hermes_cli.security_advisories import (
            ADVISORIES,
            ack_advisory,
        )
        valid_ids = {a.id for a in ADVISORIES}
        if ack_target not in valid_ids:
            print(color(
                f"Unknown advisory ID: {ack_target!r}. Known IDs: "
                f"{', '.join(sorted(valid_ids)) or '(none)'}",
                Colors.RED,
            ))
            sys.exit(2)
        if ack_advisory(ack_target):
            print(color(
                f"  ✓ Acknowledged advisory {ack_target}. "
                f"It will no longer trigger startup banners.",
                Colors.GREEN,
            ))
        else:
            print(color(
                f"  ✗ Failed to persist ack for {ack_target}. "
                f"Check ~/.hermes/config.yaml is writable.",
                Colors.RED,
            ))
            sys.exit(1)
        return

    issues = []
    manual_issues = []  # issues that can't be auto-fixed
    fixed_count = 0

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🩺 Hermes Doctor                        │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    _section("Security Advisories")
    try:
        from hermes_cli.security_advisories import (
            detect_compromised,
            filter_unacked,
            full_remediation_text,
            get_acked_ids,
        )
        all_hits = detect_compromised()
        fresh_hits = filter_unacked(all_hits)
        if fresh_hits:
            for hit in fresh_hits:
                check_fail(
                    f"{hit.advisory.title}",
                    f"({hit.package}=={hit.installed_version})",
                )
                # Print the full remediation block, indented under the
                # check_fail header so it reads as a single section.
                for line in full_remediation_text(hit):
                    if line:
                        print(f"    {color(line, Colors.YELLOW)}")
                    else:
                        print()
                # Funnel into the action list so the summary block surfaces it
                # for users who scroll past the section.
                manual_issues.append(
                    f"Resolve security advisory {hit.advisory.id}: "
                    f"uninstall {hit.package}=={hit.installed_version} and "
                    f"rotate credentials, then run "
                    f"`hermes doctor --ack {hit.advisory.id}`."
                )
            # Acked-but-still-installed: show as informational so the user
            # knows the package is still on disk after the ack.
            acked_ids = get_acked_ids()
            for h in all_hits:
                if h.advisory.id in acked_ids:
                    check_warn(
                        f"{h.package}=={h.installed_version} still installed "
                        f"(advisory {h.advisory.id} acknowledged)",
                    )
        else:
            check_ok("No active security advisories")
    except Exception as e:
        # Never let a bug in the advisory check block the rest of doctor.
        check_warn(f"Security advisory check failed: {e}")

    _section("MCP Server Security")
    try:
        from hermes_cli.config import load_config
        from hermes_cli.mcp_security import validate_mcp_server_entry

        servers = load_config().get("mcp_servers") or {}
        suspicious = 0
        if isinstance(servers, dict):
            for name, entry in sorted(servers.items()):
                if not isinstance(entry, dict):
                    continue
                issues_found = validate_mcp_server_entry(name, entry)
                if not issues_found:
                    continue
                suspicious += 1
                check_warn(f"MCP server '{name}' has suspicious stdio command", "; ".join(issues_found))
                manual_issues.append(
                    f"Review/remove mcp_servers.{name} in config.yaml; rotate any credentials that may have been exposed."
                )
        if suspicious == 0:
            check_ok("No suspicious MCP stdio commands")
    except Exception as e:
        check_warn(f"MCP security check failed: {e}")
    
    _section("Python Environment")
    py_version = sys.version_info
    if py_version >= (3, 11):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    elif py_version >= (3, 10):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    elif py_version >= (3, 8):
        check_warn(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ recommended)")
    else:
        _fail_and_issue(
            f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
            "(3.10+ required)",
            "Upgrade Python to 3.10+",
            issues,
        )

    # Linked SQLite library (issue #69784): version + source id matter independently
    # of the Python minor — uv's python-build-standalone can keep a vulnerable
    # SQLite across Python upgrades.
    try:
        import sqlite3
        from hermes_state import is_sqlite_wal_reset_vulnerable, sqlite_source_id

        _sqlite_ver = sqlite3.sqlite_version
        _sqlite_src = sqlite_source_id()
        _sqlite_src_short = (
            (_sqlite_src[:48] + "…") if len(_sqlite_src) > 48 else _sqlite_src
        )
        if is_sqlite_wal_reset_vulnerable():
            # Warn-only: Hermes already refuses to enable WAL on fresh DBs.
            # Do not append to ``issues`` because runtime repair remains
            # best-effort and unsupported installs may need manual action.
            check_warn(
                f"SQLite {_sqlite_ver} (WAL-reset bug)",
                _sqlite_upgrade_hint(),
            )
        else:
            check_ok(f"SQLite {_sqlite_ver}")
        if _sqlite_src_short:
            check_info(f"SQLite source id: {_sqlite_src_short}")
        _report_database_journal_modes()
    except Exception as e:
        check_warn(f"SQLite version probe failed: {e}")
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        check_ok("Virtual environment active")
    else:
        check_warn("Not in virtual environment", "(recommended)")

    # macOS TCC interpreter anchor (#95596): dylib-complete re-land of the
    # mechanism reverted in #95563. Silent on non-macOS.
    check_macos_tcc_anchor(should_fix=should_fix)

    # macOS Full Disk Access (issue #52010 follow-up): one grant silences
    # every per-folder prompt permanently. Silent on non-macOS.
    check_macos_full_disk_access()

    # Detect drift between pyproject.toml and hermes_cli/__init__.py versions
    # (a git conflict resolution can silently revert one but not the other).
    _check_version_consistency(issues)

    # macOS TCC grant persistence (issue #86385): a locally-built desktop
    # bundle whose DR is cdhash-pinned loses every permission grant on each
    # rebuild; a post-#73681 identifier-pinned DR survives, but grants made
    # to older binaries stay stale (toggle shows ON while macOS re-prompts).
    check_macos_tcc_grants()

    _section("SSL / CA Certificates")
    check_certificates(should_fix=should_fix, issues=manual_issues)

    _section("Required Packages")
    required_packages = [
        ("openai", "OpenAI SDK"),
        ("rich", "Rich (terminal UI)"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("httpx", "HTTPX"),
    ]
    
    optional_packages = [
        ("croniter", "Croniter (cron expressions)"),
        ("telegram", "python-telegram-bot"),
        ("discord", "discord.py"),
    ]
    
    for module, name in required_packages:
        try:
            __import__(module)
            check_ok(name)
        except ImportError:
            _fail_and_issue(name, "(missing)", f"Install {name}: {_python_install_cmd()} {module}", issues)
    
    for module, name in optional_packages:
        try:
            __import__(module)
            check_ok(name, "(optional)")
        except ImportError:
            check_warn(name, "(optional, not installed)")
    
    _section("Configuration Files")
    # Managed scope (administrator-pinned config/env), when present.
    managed_scope_check()
    # Check ~/.hermes/.env (primary location for user config)
    env_path = HERMES_HOME / '.env'
    if env_path.exists():
        check_ok(f"{_DHH}/.env file exists")
        
        # Prefer UTF-8 (.env is written as UTF-8 elsewhere). Fall back to
        # latin-1 for Windows Notepad/cp1252 files that are not valid UTF-8 —
        # matches hermes_cli.env_loader._load_dotenv_with_fallback.
        try:
            content = env_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = env_path.read_text(encoding="latin-1")
        if _has_provider_env_config(content):
            check_ok("API key or custom endpoint configured")
        else:
            check_warn(f"No API key found in {_DHH}/.env")
            issues.append("Run 'hermes setup' to configure API keys")
    else:
        # Also check project root as fallback
        fallback_env = PROJECT_ROOT / '.env'
        if fallback_env.exists():
            check_ok(".env file exists (in project directory)")
        else:
            check_fail(f"{_DHH}/.env file missing")
            if should_fix:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                env_path.touch()
                # .env holds API keys — restrict to owner-only access from
                # creation. touch() obeys umask which is commonly 0o022,
                # leaving the file world-readable; tighten explicitly.
                try:
                    os.chmod(str(env_path), 0o600)
                except OSError:
                    pass
                check_ok(f"Created empty {_DHH}/.env")
                check_info("Run 'hermes setup' to configure API keys")
                fixed_count += 1
            else:
                check_info("Run 'hermes setup' to create one")
                issues.append("Run 'hermes setup' to create .env")
    
    # Check ~/.hermes/config.yaml (primary) or project cli-config.yaml (fallback)
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        check_ok(f"{_DHH}/config.yaml exists")

        # Validate model.provider and model.default values
        try:
            # Raw-file diagnostic: inspects what the user actually wrote.
            from hermes_cli.config import read_user_config_raw
            cfg = read_user_config_raw(config_path)
            model_section = cfg.get("model") or {}
            provider_raw = (model_section.get("provider") or "").strip()
            provider = provider_raw.lower()
            default_model = (model_section.get("default") or model_section.get("model") or "").strip()

            known_providers: set = set()
            try:
                from hermes_cli.auth import (
                    PROVIDER_REGISTRY,
                    resolve_provider as _resolve_auth_provider,
                )
                known_providers = set(PROVIDER_REGISTRY.keys()) | {"openrouter", "custom", "auto", "moa"}
            except Exception:
                _resolve_auth_provider = None
                pass
            try:
                from hermes_cli.config import get_compatible_custom_providers as _compatible_custom_providers
                from hermes_cli.providers import (
                    custom_provider_aliases as _custom_provider_aliases,
                    normalize_provider as _normalize_catalog_provider,
                    resolve_provider_full as _resolve_provider_full,
                )
            except Exception:
                _compatible_custom_providers = None
                _custom_provider_aliases = None
                _normalize_catalog_provider = None
                _resolve_provider_full = None

            custom_providers = []
            if _compatible_custom_providers is not None:
                try:
                    custom_providers = _compatible_custom_providers(cfg)
                except Exception:
                    custom_providers = []

            user_providers = cfg.get("providers")
            if isinstance(user_providers, dict):
                from hermes_cli.config import is_provider_enabled
                known_providers.update(
                    str(name).strip().lower()
                    for name, prov_cfg in user_providers.items()
                    if str(name).strip() and is_provider_enabled(prov_cfg)
                )
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                provider_key = str(entry.get("provider_key") or "").strip()
                if name and _custom_provider_aliases is not None:
                    known_providers.update(
                        _custom_provider_aliases(name, provider_key)
                    )

            valid_provider_ids = set(known_providers)
            provider_ids_to_accept = {provider} if provider else set()
            if _normalize_catalog_provider is not None:
                for known_provider in known_providers:
                    try:
                        valid_provider_ids.add(_normalize_catalog_provider(known_provider))
                    except Exception:
                        continue

            runtime_provider = provider
            if (
                provider
                and _resolve_auth_provider is not None
                and provider not in {"auto", "custom"}
            ):
                try:
                    runtime_provider = _resolve_auth_provider(provider)
                    provider_ids_to_accept.add(runtime_provider)
                except Exception:
                    runtime_provider = provider

            catalog_provider = provider
            if (
                provider
                and _resolve_provider_full is not None
                and provider not in {"auto", "custom"}
            ):
                provider_def = _resolve_provider_full(provider, user_providers, custom_providers)
                catalog_provider = provider_def.id if provider_def is not None else None
                if catalog_provider is not None:
                    provider_ids_to_accept.add(catalog_provider)

            if provider and provider != "auto":
                if catalog_provider is None or (
                    known_providers
                    and not (provider_ids_to_accept & valid_provider_ids)
                ):
                    known_list = ", ".join(sorted(known_providers)) if known_providers else "(unavailable)"
                    _fail_and_issue(
                        f"model.provider '{provider_raw}' is not a recognised provider",
                        f"(known: {known_list})",
                        (
                            f"model.provider '{provider_raw}' is unknown. "
                            f"Valid providers: {known_list}. "
                            f"Fix: run 'hermes config set model.provider <valid_provider>'"
                        ),
                        issues,
                    )

            # Warn if model is set to a provider-prefixed name on a provider that doesn't use them.
            # Vendor/model slugs are valid on aggregator-style providers and on any custom
            # provider — bare "custom" or a named "custom:<name>" that fronts an OpenAI-compatible
            # aggregator (e.g. custom:hpc-ai serving deepseek/deepseek-v4-flash) requires the prefix.
            provider_for_policy = runtime_provider or catalog_provider
            provider_policy_id = str(provider_for_policy or "").strip().lower()
            providers_accepting_vendor_slugs = {
                "openrouter",
                "auto",
                "ai-gateway",
                "kilocode",
                "opencode-zen",
                "huggingface",
                "lmstudio",
                "nous",
                "nvidia",
                # Fireworks' native model IDs are slash-form
                # (accounts/fireworks/models/... and .../routers/...), so a "/"
                # is expected, not an aggregator vendor prefix.
                "fireworks",
                # DeepInfra is an aggregator-style gateway: its catalog
                # is exclusively ``vendor/model`` slugs (Qwen/Qwen3.5-…,
                # meta-llama/Llama-3-…, anthropic/claude-opus-4-7, …).
                "deepinfra",
            }
            provider_accepts_vendor_slug = (
                provider_policy_id in providers_accepting_vendor_slugs
                or provider_policy_id == "custom"
                or provider_policy_id.startswith("custom:")
            )
            if (
                default_model
                and "/" in default_model
                and provider_policy_id
                and not provider_accepts_vendor_slug
            ):
                check_warn(
                    f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'",
                    "(vendor-prefixed slugs belong to aggregators like openrouter)",
                )
                issues.append(
                    f"model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. "
                    "Either set model.provider to 'openrouter', or drop the vendor prefix."
                )

            # Check credentials for the configured provider.
            # Limit to API-key providers in PROVIDER_REGISTRY — other provider
            # types (OAuth, SDK, anthropic/custom/auto) have their own env-var
            # checks elsewhere in doctor, and get_auth_status() returns a bare
            # {logged_in: False} for anything it doesn't explicitly dispatch,
            # which would produce false positives.
            if runtime_provider and runtime_provider not in ("auto", "custom"):
                try:
                    if runtime_provider == "openrouter":
                        from hermes_cli.config import get_env_value

                        configured = bool(
                            str(get_env_value("OPENROUTER_API_KEY") or "").strip()
                            or str(get_env_value("OPENAI_API_KEY") or "").strip()
                        )
                    else:
                        from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status

                        pconfig = PROVIDER_REGISTRY.get(runtime_provider)
                        configured = True
                        if pconfig and getattr(pconfig, "auth_type", "") == "api_key":
                            status = get_auth_status(runtime_provider) or {}
                            configured = bool(
                                status.get("configured")
                                or status.get("logged_in")
                                or status.get("api_key")
                            )
                    if not configured:
                        _fail_and_issue(
                            f"model.provider '{runtime_provider}' is set but no API key is configured",
                            "(check ~/.hermes/.env or run 'hermes setup')",
                            (
                                f"No credentials found for provider '{runtime_provider}'. "
                                f"Run 'hermes setup' or set the provider's API key in {_DHH}/.env, "
                                f"or switch providers with 'hermes config set model.provider <name>'"
                            ),
                            issues,
                        )
                except Exception:
                    pass

        except Exception as e:
            check_warn("Could not validate model/provider config", f"({e})")
    else:
        fallback_config = PROJECT_ROOT / 'cli-config.yaml'
        if fallback_config.exists():
            check_ok("cli-config.yaml exists (in project directory)")
        else:
            if should_fix:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                example_config = PROJECT_ROOT / 'cli-config.yaml.example'
                if example_config.exists():
                    shutil.copy2(str(example_config), str(config_path))
                    check_ok(f"Created {_DHH}/config.yaml from cli-config.yaml.example")
                else:
                    from hermes_cli.config import DEFAULT_CONFIG, save_config
                    save_config(DEFAULT_CONFIG)
                    check_ok(f"Created {_DHH}/config.yaml from defaults")
                fixed_count += 1
            else:
                check_warn("config.yaml not found", "(using defaults)")

    # Check config version and stale keys
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        try:
            from hermes_cli.config import check_config_version, migrate_config
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                check_warn(
                    f"Config version outdated (v{current_ver} → v{latest_ver})",
                    "(new settings available)"
                )
                if should_fix:
                    try:
                        migrate_config(interactive=False, quiet=False)
                        check_ok("Config migrated to latest version")
                        fixed_count += 1
                    except Exception as mig_err:
                        check_warn(f"Auto-migration failed: {mig_err}")
                        issues.append("Run 'hermes setup' to migrate config")
                else:
                    issues.append("Run 'hermes doctor --fix' or 'hermes setup' to migrate config")
            else:
                check_ok(f"Config version up to date (v{current_ver})")
        except Exception:
            pass

        # Detect stale root-level model keys (known bug source — PR #4329)
        try:
            # Raw-file diagnostic: stale-key detection must see the raw file.
            from hermes_cli.config import read_user_config_raw
            raw_config = read_user_config_raw(config_path)
            stale_root_keys = [k for k in ("provider", "base_url") if k in raw_config and isinstance(raw_config[k], str)]
            if stale_root_keys:
                check_warn(
                    f"Stale root-level config keys: {', '.join(stale_root_keys)}",
                    "(should be under 'model:' section)"
                )
                if should_fix:
                    # Coerce scalar/None ``model:`` into a dict before mutation —
                    # ``setdefault("model", {})`` would return an existing scalar
                    # and then ``model_section[k] = ...`` would raise TypeError.
                    raw_model = raw_config.get("model")
                    if isinstance(raw_model, dict):
                        model_section = raw_model
                    elif isinstance(raw_model, str) and raw_model.strip():
                        model_section = {"default": raw_model.strip()}
                        raw_config["model"] = model_section
                    else:
                        model_section = {}
                        raw_config["model"] = model_section
                    for k in stale_root_keys:
                        if not model_section.get(k):
                            model_section[k] = raw_config.pop(k)
                        else:
                            raw_config.pop(k)
                    from hermes_cli.config import atomic_config_write
                    atomic_config_write(config_path, raw_config)
                    check_ok("Migrated stale root-level keys into model section")
                    fixed_count += 1
                else:
                    issues.append("Stale root-level provider/base_url in config.yaml — run 'hermes doctor --fix'")
        except Exception:
            pass

        # Detect stale HERMES_MAX_ITERATIONS ghost in .env shadowing
        # agent.max_turns in config.yaml (issue #17534). The setup wizard
        # used to dual-write the iteration budget to both stores; users who
        # later edit only config.yaml are left with a .env ghost. The gateway
        # bridge normally derives HERMES_MAX_ITERATIONS from agent.max_turns
        # at startup, but if that bridge bails (any earlier config-parse
        # error), the stale .env value silently wins and the agent runs at the
        # wrong budget — e.g. config says 400 but the activity line reads N/90.
        # Read the .env FILE directly (load_env), not get_env_value/os.environ,
        # which the startup bridge may already have overridden.
        try:
            from hermes_cli.config import load_env, read_user_config_raw, remove_env_value
            # Raw-file diagnostic: drift check against the raw file.
            raw_config = read_user_config_raw(config_path)
            agent_cfg = raw_config.get("agent")
            cfg_max_turns = (
                agent_cfg.get("max_turns")
                if isinstance(agent_cfg, dict)
                else None
            )
            # Legacy root-level key counts too.
            if cfg_max_turns is None:
                cfg_max_turns = raw_config.get("max_turns")
            env_ghost = load_env().get("HERMES_MAX_ITERATIONS")
            drift = (
                cfg_max_turns is not None
                and env_ghost is not None
                and str(cfg_max_turns).strip() != str(env_ghost).strip()
            )
            if drift:
                check_warn(
                    f"HERMES_MAX_ITERATIONS={env_ghost} in .env shadows "
                    f"agent.max_turns={cfg_max_turns} in config.yaml",
                    "(stale ghost from an earlier `hermes setup` run)",
                )
                if should_fix:
                    if remove_env_value("HERMES_MAX_ITERATIONS"):
                        check_ok(
                            "Removed stale HERMES_MAX_ITERATIONS from .env "
                            f"(config.yaml agent.max_turns={cfg_max_turns} is now authoritative)"
                        )
                        fixed_count += 1
                    else:
                        check_warn("Could not remove HERMES_MAX_ITERATIONS from .env")
                        manual_issues.append(
                            "Manually delete the HERMES_MAX_ITERATIONS line from "
                            f"{_DHH}/.env — config.yaml agent.max_turns is authoritative."
                        )
                else:
                    issues.append(
                        "Stale HERMES_MAX_ITERATIONS in .env shadows config.yaml — "
                        "run 'hermes doctor --fix'"
                    )
        except Exception:
            pass

        # Surface deprecated/legacy config keys and env vars (warn-only).
        # Migrations may still live in config.py version steps; doctor does
        # not auto-delete here — only tells the user the modern replacement.
        try:
            from hermes_cli.config import load_env as _load_env_depr
            from hermes_cli.config import read_user_config_raw as _read_raw_depr

            # Raw-file diagnostic: deprecation sweep inspects the raw file.
            _raw_for_depr = _read_raw_depr(config_path)
            # Prefer the on-disk .env so bridged process env (e.g. TERMINAL_CWD
            # from terminal.cwd) does not false-positive.
            try:
                _env_for_depr = _load_env_depr()
            except Exception:
                _env_for_depr = {}
            report_deprecated_config_and_env(_raw_for_depr, _env_for_depr)
        except Exception:
            pass

        # Validate config structure (catches malformed custom_providers, etc.)
        try:
            from hermes_cli.config import validate_config_structure
            config_issues = validate_config_structure()
            if config_issues:
                _section("Config Structure")
                for ci in config_issues:
                    if ci.severity == "error":
                        check_fail(ci.message)
                    else:
                        check_warn(ci.message)
                    # Show the hint indented
                    for hint_line in ci.hint.splitlines():
                        check_info(hint_line)
                    issues.append(ci.message)
        except Exception:
            pass

    if not config_path.exists():
        # No config.yaml — still surface deprecated env vars from .env.
        try:
            from hermes_cli.config import load_env as _load_env_depr

            try:
                _env_for_depr = _load_env_depr()
            except Exception:
                _env_for_depr = {}
            report_deprecated_config_and_env({}, _env_for_depr)
        except Exception:
            pass

    _section("xAI Model Retirement (May 15, 2026)")

    try:
        from hermes_cli.config import load_config
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            find_retired_xai_refs,
            format_issue,
        )

        _xai_cfg = load_config()
        retired_refs = find_retired_xai_refs(_xai_cfg)
        if not retired_refs:
            check_ok("No retired xAI models in config")
        else:
            for ref in retired_refs:
                check_warn(format_issue(ref))
            check_info(f"Migration guide: {MIGRATION_GUIDE_URL}")
            manual_issues.append(
                f"Update {len(retired_refs)} retired xAI model reference(s) "
                f"in config.yaml — see {MIGRATION_GUIDE_URL}"
            )
    except Exception as _xai_check_err:
        check_warn("xAI retirement check skipped", f"({_xai_check_err})")

    _section("Auth Providers")

    try:
        from hermes_cli.auth import (
            get_nous_auth_status_local,
            get_codex_auth_status,
            get_minimax_oauth_auth_status,
        )

        # Read-only display: refresh-free snapshot — doctor must never
        # trigger an OAuth refresh as a side effect of a health check.
        nous_status = get_nous_auth_status_local()
        if nous_status.get("logged_in"):
            check_ok("Nous Portal auth", "(logged in)")
        else:
            check_warn("Nous Portal auth", "(not logged in)")

        codex_status = get_codex_auth_status()
        if codex_status.get("logged_in"):
            check_ok("OpenAI Codex auth", "(logged in)")
        else:
            check_warn("OpenAI Codex auth", "(not logged in)")
            if codex_status.get("error"):
                check_info(codex_status["error"])
            # Native OAuth uses Hermes' own device-code flow — the Codex CLI is
            # only needed to import existing tokens from ~/.codex/auth.json.
            # Attach the hint to the Codex auth row so it doesn't read as
            # remediation for whichever provider happens to print next (#27975).
            if not _safe_which("codex"):
                check_info(
                    "codex CLI not installed "
                    "(optional — only required to import tokens "
                    "from an existing Codex CLI login)"
                )

        minimax_status = get_minimax_oauth_auth_status()
        if minimax_status.get("logged_in"):
            region = minimax_status.get("region", "global")
            check_ok("MiniMax OAuth", f"(logged in, region={region})")
        else:
            check_warn("MiniMax OAuth", "(not logged in)")
    except Exception as e:
        check_warn("Auth provider status", f"(could not check: {e})")

    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the already-printed Nous/Codex/Gemini/MiniMax rows above.
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        xai_oauth_status = get_xai_oauth_auth_status() or {}
        if xai_oauth_status.get("logged_in"):
            check_ok("xAI OAuth", "(logged in)")
        else:
            check_warn("xAI OAuth", "(not logged in)")
            if xai_oauth_status.get("error"):
                check_info(xai_oauth_status["error"])
    except Exception:
        pass

    _section("Directory Structure")
    hermes_home = HERMES_HOME
    if hermes_home.exists():
        check_ok(f"{_DHH} directory exists")
    elif should_fix:
        hermes_home.mkdir(parents=True, exist_ok=True)
        check_ok(f"Created {_DHH} directory")
        fixed_count += 1
    else:
        check_warn(f"{_DHH} not found", "(will be created on first use)")
    
    from tools.memory_tool import get_builtin_memory_store_flags

    _memory_config = _doctor_memory_config(hermes_home)
    _memory_enabled, _user_profile_enabled = get_builtin_memory_store_flags(
        {"memory": _memory_config}
    )

    # Check expected subdirectories. The built-in file store does not create or
    # consume memories/ when both targets are disabled, so stale migration files
    # are not an active diagnostic surface.
    expected_subdirs = ["cron", "sessions", "logs", "skills"]
    if _memory_enabled or _user_profile_enabled:
        expected_subdirs.append("memories")
    for subdir_name in expected_subdirs:
        subdir_path = hermes_home / subdir_name
        if subdir_path.exists():
            check_ok(f"{_DHH}/{subdir_name}/ exists")
        elif should_fix:
            subdir_path.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/{subdir_name}/")
            fixed_count += 1
        else:
            check_warn(f"{_DHH}/{subdir_name}/ not found", "(will be created on first use)")
    
    # Check for SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Check if it's just the template comments (no real content)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith(("<!--", "-->", "#"))]
        if lines:
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(
                "# Hermes Agent Persona\n\n"
                "<!-- Edit this file to customize how Hermes communicates. -->\n\n"
                "You are Hermes, a helpful AI assistant.\n",
                encoding="utf-8",
            )
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            fixed_count += 1
    
    # Check only enabled built-in stores. External providers are additive, but
    # users can explicitly disable either legacy file target; stale files left
    # by a migration must not be presented as active memory usage.
    memories_dir = hermes_home / "memories"
    if not (_memory_enabled or _user_profile_enabled):
        check_info("Built-in memory files disabled by config")
    elif memories_dir.exists():
        check_ok(f"{_DHH}/memories/ directory exists")
        memory_file = memories_dir / "MEMORY.md"
        user_file = memories_dir / "USER.md"
        if _memory_enabled:
            if memory_file.exists():
                size = len(memory_file.read_text(encoding="utf-8").strip())
                check_ok(f"MEMORY.md exists ({size} chars)")
            else:
                check_info("MEMORY.md not created yet (will be created when the agent first writes a memory)")
        if _user_profile_enabled:
            if user_file.exists():
                size = len(user_file.read_text(encoding="utf-8").strip())
                check_ok(f"USER.md exists ({size} chars)")
            else:
                check_info("USER.md not created yet (will be created when the agent first writes a memory)")
    else:
        check_warn(f"{_DHH}/memories/ not found", "(will be created on first use)")
        if should_fix:
            memories_dir.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/memories/")
            fixed_count += 1
    
    # Check SQLite session store
    state_db_path = hermes_home / "state.db"
    if state_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(state_db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            conn.close()
            check_ok(f"{_DHH}/state.db exists ({count} sessions)")

            # FTS write-health probe (#50502): `SELECT COUNT(*)` above succeeds
            # even when the FTS index is corrupt and every message write fails
            # through the triggers. `_db_opens_cleanly` now drives a rolled-back
            # write so this otherwise-silent corruption class is surfaced (and
            # repaired in place with --fix).
            from hermes_state import _db_opens_cleanly, repair_state_db_schema

            _write_reason = _db_opens_cleanly(state_db_path)
            if _write_reason is not None:
                check_warn(
                    f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)",
                    f"({_write_reason})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            "Repaired state.db FTS write health",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        fixed_count += 1
                    else:
                        check_warn(
                            "state.db FTS write-health repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db FTS write corruption and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db FTS write corruption — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to rebuild the FTS index"
                    )
        except Exception as e:
            from hermes_state import is_malformed_db_error, repair_state_db_schema

            if is_malformed_db_error(e):
                # sqlite_master itself is malformed (e.g. duplicate
                # messages_fts) — every statement fails before it runs, so
                # this is NOT a plain FTS-index rebuild. Repair sqlite_master
                # in place (backup first; sessions/messages preserved).
                check_warn(
                    f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)",
                    f"({e})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        try:
                            conn = sqlite3.connect(str(state_db_path))
                            count = conn.execute(
                                "SELECT COUNT(*) FROM sessions"
                            ).fetchone()[0]
                            conn.close()
                        except Exception:
                            count = "?"
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            f"Repaired state.db schema ({count} sessions recovered)",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        fixed_count += 1
                    else:
                        check_warn(
                            "state.db schema repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db schema malformed and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db schema malformed — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to recover hidden sessions"
                    )
            else:
                check_warn(f"{_DHH}/state.db exists but has issues: {e}")

        # Health/stats snapshot (#statedb-visibility): a multi-GB state.db
        # with a runaway WAL was previously invisible to every Hermes
        # surface. Strictly read-only (mode=ro) so it is safe against a
        # live DB held by the gateway; any failure degrades to one info
        # line rather than failing doctor.
        try:
            from hermes_state import collect_state_db_stats, count_db_holders

            _db_stats = collect_state_db_stats(state_db_path)
            _db_holders = count_db_holders(state_db_path)
            for _kind, _text, _detail in _render_state_db_stats(
                _db_stats, holders=_db_holders
            ):
                if _kind == "warn":
                    check_warn(_text, _detail)
                    if "auto_prune" in _detail:
                        issues.append(
                            "state.db is large — enable sessions.auto_prune "
                            "in config.yaml"
                            + (
                                " and run 'hermes sessions optimize-storage' "
                                "offline (gateway stopped)"
                                if "optimize-storage" in _detail else ""
                            )
                        )
                else:
                    check_info(_text + (f" {_detail}" if _detail else ""))
        except Exception as _stats_exc:
            check_info(f"state.db stats unavailable ({_stats_exc})")
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")

    # Check WAL file size (unbounded growth indicates missed checkpoints)
    wal_path = hermes_home / "state.db-wal"
    if wal_path.exists():
        try:
            wal_size = wal_path.stat().st_size
            if wal_size > 50 * 1024 * 1024:  # 50 MB
                check_warn(
                    f"WAL file is large ({wal_size // (1024*1024)} MB)",
                    "(may indicate missed checkpoints)"
                )
                if should_fix:
                    import sqlite3
                    conn = sqlite3.connect(str(state_db_path))
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    conn.close()
                    new_size = wal_path.stat().st_size if wal_path.exists() else 0
                    check_ok(f"WAL checkpoint performed ({wal_size // 1024}K → {new_size // 1024}K)")
                    fixed_count += 1
                else:
                    issues.append("Large WAL file — run 'hermes doctor --fix' to checkpoint")
            elif wal_size > 10 * 1024 * 1024:  # 10 MB
                check_info(f"WAL file is {wal_size // (1024*1024)} MB (normal for active sessions)")
        except Exception:
            pass

    _check_gateway_service_linger(issues)
    _check_s6_supervision(issues)

    if sys.platform != "win32":
        _section("Command Installation")
        # Determine the venv entry point location
        _venv_bin = None
        for _venv_name in ("venv", ".venv"):
            _candidate = PROJECT_ROOT / _venv_name / "bin" / "hermes"
            if _candidate.exists():
                _venv_bin = _candidate
                break

        # Determine the expected command link directory (mirrors install.sh logic)
        _prefix = os.environ.get("PREFIX", "")
        _is_termux_env = bool(os.environ.get("TERMUX_VERSION")) or "com.termux/files/usr" in _prefix
        if _is_termux_env and _prefix:
            _cmd_link_dir = Path(_prefix) / "bin"
            _cmd_link_display = "$PREFIX/bin"
        else:
            _cmd_link_dir = Path.home() / ".local" / "bin"
            _cmd_link_display = "~/.local/bin"
        _cmd_link = _cmd_link_dir / "hermes"

        if _venv_bin is None:
            check_warn(
                "Venv entry point not found",
                "(hermes not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')"
            )
            manual_issues.append(
                f"Reinstall entry point: cd {PROJECT_ROOT} && source venv/bin/activate && pip install -e '.[all]'"
            )
        else:
            check_ok(f"Venv entry point exists ({_venv_bin.relative_to(PROJECT_ROOT)})")

            # Check the symlink at the command link location
            if _cmd_link.is_symlink():
                _target = _cmd_link.resolve()
                _expected = _venv_bin.resolve()
                if _target == _expected:
                    check_ok(f"{_cmd_link_display}/hermes → correct target")
                else:
                    check_warn(
                        f"{_cmd_link_display}/hermes points to wrong target",
                        f"(→ {_target}, expected → {_expected})"
                    )
                    if should_fix:
                        _cmd_link.unlink()
                        _cmd_link.symlink_to(_venv_bin)
                        check_ok(f"Fixed symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                        fixed_count += 1
                    else:
                        issues.append(f"Broken symlink at {_cmd_link_display}/hermes — run 'hermes doctor --fix'")
            elif _cmd_link.exists():
                # It's a regular file, not a symlink — possibly a wrapper script
                check_ok(f"{_cmd_link_display}/hermes exists (non-symlink)")
            else:
                check_fail(
                    f"{_cmd_link_display}/hermes not found",
                    "(hermes command may not work outside the venv)"
                )
                if should_fix:
                    _cmd_link_dir.mkdir(parents=True, exist_ok=True)
                    _cmd_link.symlink_to(_venv_bin)
                    check_ok(f"Created symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                    fixed_count += 1

                    # Check if the link dir is on PATH
                    _path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    if str(_cmd_link_dir) not in _path_dirs:
                        check_warn(
                            f"{_cmd_link_display} is not on your PATH",
                            "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")"
                        )
                        manual_issues.append(f"Add {_cmd_link_display} to your PATH")
                else:
                    issues.append(f"Missing {_cmd_link_display}/hermes symlink — run 'hermes doctor --fix'")

    _section("External Tools")
    # Git
    if _safe_which("git"):
        check_ok("git")
    else:
        check_warn("git not found", "(optional)")
    
    # ripgrep (optional, for faster file search)
    if _safe_which("rg"):
        check_ok("ripgrep (rg)", "(faster file search)")
    else:
        check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        check_info(f"Install for faster search: {_system_package_install_cmd('ripgrep')}")
    
    # Docker (optional)
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    try:
        from hermes_constants import is_container as _is_container
        running_in_container = _is_container()
    except Exception:
        running_in_container = False

    if running_in_container:
        # Inside our container the Docker terminal backend is not
        # configured by default (Docker-in-Docker isn't set up); the
        # local backend is the intended one. Skip the noisy "docker
        # not found" warning. If the user has explicitly chosen
        # TERMINAL_ENV=docker inside the container they likely mounted
        # /var/run/docker.sock, so fall through to the normal check.
        if terminal_env != "docker":
            check_info(
                "Running inside a container — using local terminal backend "
                "(docker-in-docker is not configured by default)"
            )
            # Skip to next section; Docker isn't relevant here.
            terminal_env = "local"
    if terminal_env == "docker":
        if _safe_which("docker"):
            # Check if docker daemon is running
            try:
                result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok("docker", "(daemon running)")
            else:
                _fail_and_issue("docker daemon not running", "", "Start Docker daemon", issues)
        else:
            _fail_and_issue(
                "docker not found",
                "(required for TERMINAL_ENV=docker)",
                "Install Docker or change TERMINAL_ENV",
                issues,
            )
    elif _safe_which("docker"):
        check_ok("docker", "(optional)")
    elif _is_termux():
        check_info("Docker backend is not available inside Termux (expected on Android)")
    elif running_in_container:
        pass  # already explained above
    else:
        check_warn("docker not found", "(optional)")
    
    # SSH (if using ssh backend)
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST")
        if ssh_host:
            ssh_user = os.getenv("TERMINAL_SSH_USER")
            ssh_port = os.getenv("TERMINAL_SSH_PORT")
            ssh_key = os.getenv("TERMINAL_SSH_KEY")
            target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
            cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
            if ssh_port:
                cmd += ["-p", ssh_port]
            if ssh_key:
                cmd += ["-i", os.path.expanduser(ssh_key)]
            cmd += [target, "echo ok"]
            # Try to connect
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=15
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok(f"SSH connection to {ssh_host}")
            else:
                _fail_and_issue(f"SSH connection to {ssh_host}", "", f"Check SSH configuration for {ssh_host}", issues)
        else:
            _fail_and_issue(
                "TERMINAL_SSH_HOST not set",
                "(required for TERMINAL_ENV=ssh)",
                "Set TERMINAL_SSH_HOST in .env",
                issues,
            )
    
    # Daytona (if using daytona backend)
    if terminal_env == "daytona":
        daytona_key = os.getenv("DAYTONA_API_KEY")
        if daytona_key:
            check_ok("Daytona API key", "(configured)")
        else:
            _fail_and_issue(
                "DAYTONA_API_KEY not set",
                "(required for TERMINAL_ENV=daytona)",
                "Set DAYTONA_API_KEY environment variable",
                issues,
            )
        try:
            from daytona import Daytona  # noqa: F401 — SDK presence check
            check_ok("daytona SDK", "(installed)")
        except ImportError:
            _fail_and_issue(
                "daytona SDK not installed",
                "(pip install daytona)",
                "Install daytona SDK: pip install daytona",
                issues,
            )

    # Vercel Sandbox (if using vercel_sandbox backend)
    if terminal_env == "vercel_sandbox":
        runtime = os.getenv("TERMINAL_VERCEL_RUNTIME", "node24").strip() or "node24"
        from tools.terminal_tool import _SUPPORTED_VERCEL_RUNTIMES
        if runtime in _SUPPORTED_VERCEL_RUNTIMES:
            check_ok("Vercel runtime", f"({runtime})")
        else:
            supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
            _fail_and_issue(
                "Vercel runtime unsupported",
                f"({runtime}; use {supported})",
                f"Set TERMINAL_VERCEL_RUNTIME to one of: {supported}",
                issues,
            )

        disk = os.getenv("TERMINAL_CONTAINER_DISK", "51200").strip()
        if disk in {"", "0", "51200"}:
            check_ok("Vercel disk setting", "(uses platform default)")
        else:
            _fail_and_issue(
                "Vercel custom disk unsupported",
                "(reset terminal.container_disk to 51200)",
                "Vercel Sandbox does not support custom container_disk; use the shared default 51200",
                issues,
            )

        if importlib.util.find_spec("vercel") is not None:
            check_ok("vercel SDK", "(installed)")
        else:
            _fail_and_issue(
                "vercel SDK not installed",
                "(pip install 'hermes-agent[vercel]')",
                "Install the Vercel optional dependency: pip install 'hermes-agent[vercel]'",
                issues,
            )

        auth_status = describe_vercel_auth()
        if auth_status.ok:
            check_ok("Vercel auth", f"({auth_status.label})")
        elif auth_status.label.startswith("partial"):
            _fail_and_issue(
                "Vercel auth incomplete",
                f"({auth_status.label})",
                "Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together",
                issues,
            )
        else:
            _fail_and_issue(
                "Vercel auth not configured",
                f"({auth_status.label})",
                "Configure Vercel Sandbox auth with VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID",
                issues,
            )
        for line in auth_status.detail_lines:
            check_info(f"Vercel auth {line}")

        persistent = os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"1", "true", "yes", "on"}
        if persistent:
            check_info("Vercel persistence: snapshot filesystem only; live processes do not survive sandbox recreation")
        else:
            check_info("Vercel persistence: ephemeral filesystem")

    # Plugin-registered terminal backends (if one is the active backend)
    if terminal_env not in {
        "local", "docker", "singularity", "modal", "managed_modal",
        "daytona", "vercel_sandbox", "ssh",
    }:
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            from agent.terminal_env_registry import get_provider

            _provider = get_provider(terminal_env)
        except Exception:
            _provider = None
        if _provider is None:
            _fail_and_issue(
                f"Unknown terminal backend '{terminal_env}'",
                "(no built-in or plugin backend by that name)",
                "Fix terminal.backend in config.yaml, or install/enable the plugin that provides it",
                issues,
            )
        else:
            for _ok, _label, _detail in _provider.doctor_checks():
                if _ok:
                    check_ok(_label, _detail)
                else:
                    _fail_and_issue(_label, _detail, _detail.strip("()"), issues)

    # Node.js + agent-browser (for browser automation tools)
    if _safe_which("node"):
        check_ok("Node.js")
        # agent-browser is no longer a root package.json dependency (#43564)
        # — it resolves lazily via npx (or a global/Hermes-managed install)
        # at first use. Mirror tools.browser_tool._find_agent_browser's own
        # resolution cascade here so doctor can't diverge from what browser
        # tools will actually find; validate=False keeps this a cheap
        # existence check with no subprocess spawn or install side effects.
        agent_browser_ok = False
        try:
            from tools.browser_tool import _find_agent_browser, _is_npx_agent_browser_sentinel
            _resolved_ab = _find_agent_browser(validate=False)
        except Exception:
            _resolved_ab = None

        if _resolved_ab and _is_npx_agent_browser_sentinel(_resolved_ab):
            check_ok("agent-browser", "(resolves via npx on first use)")
            agent_browser_ok = True
            if should_fix:
                # Doctor can't tell from here whether npx's cache already
                # has agent-browser warm — just fire the same warm-up
                # `hermes update` does, so a session's first browser call
                # doesn't pay the registry fetch either way.
                from tools.browser_tool import warm_agent_browser_npx_cache
                if warm_agent_browser_npx_cache():
                    check_info("  Warmed npx cache for agent-browser")
                else:
                    check_info("  Could not warm npx cache (offline or npx unavailable)")
        elif _resolved_ab and agent_browser_runnable(_resolved_ab):
            check_ok("agent-browser", "(browser automation)")
            agent_browser_ok = True
        elif _resolved_ab:
            # Found on PATH but won't run — almost always a dangling global
            # symlink left behind by agent-browser's npm postinstall after a
            # `hermes update` wiped node_modules (issue #48521).
            check_warn(
                "agent-browser found but not runnable",
                f"(broken symlink at {_resolved_ab}? run: npx agent-browser --version)",
            )
        elif _is_termux():
            check_info("agent-browser is not installed (expected in the tested Termux path)")
            check_info("Install it manually later with: npm install -g agent-browser && agent-browser install")
            check_info("Termux browser setup:")
            for step in _termux_browser_setup_steps(node_installed=True):
                check_info(step)
        else:
            check_warn("agent-browser not installed", "(requires npm/npx on PATH)")

        # Chromium presence — the browser tools silently fail to register when
        # agent-browser is found but no Playwright-managed Chromium is on disk
        # (tools/browser_tool.py::check_browser_requirements filters them out
        # before the agent ever sees them).  Reuse the exact predicate it uses
        # so the two checks cannot diverge.  Skip on Termux (not a tested
        # path).
        if agent_browser_ok and not _is_termux():
            try:
                # Lazy import: browser_tool is a ~150KB module we don't want
                # to eagerly load in every `hermes doctor` invocation.
                from tools.browser_tool import (
                    _chromium_installed,
                    _is_camofox_mode,
                    _get_cloud_provider,
                    _get_cdp_override_raw,
                    _using_lightpanda_engine,
                )
            except Exception:
                # If browser_tool can't even import, that's a separate bug
                # surfaced elsewhere; don't crash doctor.
                pass
            else:
                # Only warn about Chromium if the installed engine actually
                # requires it: Camofox, CDP override, a cloud provider, or
                # Lightpanda all bypass the local Chromium requirement.
                skip_chromium_check = (
                    _is_camofox_mode()
                    or bool(_get_cdp_override_raw())
                    or _get_cloud_provider() is not None
                    or _using_lightpanda_engine()
                )
                if not skip_chromium_check:
                    if _chromium_installed():
                        check_ok("Playwright Chromium", "(browser engine)")
                    else:
                        check_warn(
                            "Playwright Chromium not installed",
                            "(browser_* tools will be hidden from the agent)",
                        )
                        if sys.platform == "win32":
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install chromium"
                            )
                        else:
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install --with-deps chromium"
                            )
    elif _is_termux():
        check_info("Node.js not found (browser tools are optional in the tested Termux path)")
        check_info("Install Node.js on Termux with: pkg install nodejs")
        check_info("Termux browser setup:")
        for step in _termux_browser_setup_steps(node_installed=False):
            check_info(step)
    else:
        check_warn("Node.js not found", "(optional, needed for browser tools)")

    # Lightpanda engine (browser.engine / AGENT_BROWSER_ENGINE). Independent
    # of Node: Browser Use mode spawns ``lightpanda serve`` itself.
    try:
        from tools.browser_tool import _using_lightpanda_engine, lightpanda_engine_status
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary
    except Exception:
        pass
    else:
        # _using_lightpanda_engine() is a cached config read — a failure
        # there would be exceptional, not something to silently hide.
        if _using_lightpanda_engine():
            try:
                _lp_used, _lp_reason = lightpanda_engine_status()
            except Exception as e:
                _lp_used, _lp_reason = False, f"status check failed: {e}"
            if not _lp_used:
                check_warn("browser.engine=lightpanda is shadowed", f"({_lp_reason})")
                check_info(
                    "Fix: pick Lightpanda in `hermes tools` → Browser Automation, "
                    "or set browser.engine: auto"
                )
            elif find_lightpanda_binary():
                check_ok("Lightpanda", f"({_lp_reason})")
            else:
                check_warn(
                    "Lightpanda selected but binary not found",
                    "(browser tools will fail until it is installed)",
                )
                check_info(LIGHTPANDA_INSTALL_HINT)

    # npm audit for all Node.js packages
    _npm_bin = _safe_which("npm")
    if _npm_bin:
        # Each entry: (cwd, label, extra_audit_args)
        # PROJECT_ROOT is audited with --workspaces=false so that the apps/*
        # glob (which pulls in Electron, node-pty, etc.) is never resolved
        # for a routine security check. The web and ui-tui workspaces are
        # audited separately via --workspace flags. See #38772.
        # The WhatsApp bridge may live under a writable HERMES_HOME mirror
        # instead of the (possibly read-only) install tree in Docker — resolve
        # it through the shared helper so we audit the dir that actually holds
        # node_modules. See #49561.
        try:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            _whatsapp_bridge_dir = resolve_whatsapp_bridge_dir()
        except Exception:
            _whatsapp_bridge_dir = PROJECT_ROOT / "scripts" / "whatsapp-bridge"
        npm_audit_targets = [
            (PROJECT_ROOT, "Browser tools (agent-browser)", ["--workspaces=false"]),
            (PROJECT_ROOT, "web workspace", ["--workspace", "web"]),
            (PROJECT_ROOT, "ui-tui workspace", ["--workspace", "ui-tui"]),
            (_whatsapp_bridge_dir, "WhatsApp bridge", []),
        ]
        for npm_dir, label, audit_extra in npm_audit_targets:
            # For workspace-scoped audits run from PROJECT_ROOT the
            # node_modules check must use the workspace root; standalone dirs
            # (whatsapp-bridge) check their own node_modules.
            check_dir = PROJECT_ROOT if audit_extra else npm_dir
            if not (check_dir / "node_modules").exists():
                continue
            try:
                # Use resolved absolute path so Windows can execute
                # npm.cmd (CreateProcessW can't run bare .cmd names).
                audit_result = subprocess.run(
                    [_npm_bin, "audit", "--json", *audit_extra],
                    cwd=str(npm_dir),
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                )
                import json as _json
                audit_data = _json.loads(audit_result.stdout) if audit_result.stdout.strip() else {}
                vuln_count = audit_data.get("metadata", {}).get("vulnerabilities", {})
                critical = vuln_count.get("critical", 0)
                high = vuln_count.get("high", 0)
                moderate = vuln_count.get("moderate", 0)
                total = critical + high + moderate
                # Determine a scoped fix command for the remediation hint.
                if audit_extra and audit_extra[0] == "--workspace":
                    # Detection (`npm audit --workspace <name>`) is read-only and
                    # safe, but `npm audit fix --workspace <name>` crashes on
                    # current npm with "Cannot read properties of null (reading
                    # 'edgesOut')" — an arborist bug with workspace-filtered
                    # audit fix. The root-level `npm audit fix` can crash on the
                    # same tree with "isDescendantOf", so do not hand the user a
                    # manual fix command for these build-tool advisories.
                    fix_cmd = None
                elif audit_extra == ["--workspaces=false"]:
                    fix_cmd = f"cd {npm_dir} && npm audit fix --workspaces=false"
                else:
                    fix_cmd = f"cd {npm_dir} && npm audit fix"
                if total == 0:
                    check_ok(f"{label} deps", "(no known vulnerabilities)")
                elif critical > 0 or high > 0:
                    if fix_cmd:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — run: {fix_cmd}"
                        )
                    else:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — "
                            "build-tool advisory; clears via lockfile bump"
                        )
                    check_warn(
                        f"{label} deps",
                        f"({vuln_detail})"
                    )
                    if audit_extra and audit_extra[0] == "--workspace":
                        # The web/ui-tui workspace advisories are in build-time
                        # tooling (esbuild/vite, etc.), not runtime code that ships
                        # to users. Manual npm remediation may error with a known
                        # arborist crash (edgesOut / isDescendantOf) on this monorepo
                        # tree — in that case it is an npm bug, not a Hermes one.
                        check_info(
                            "  ^ build-time tooling (not runtime); if manual npm remediation "
                            "errors with an arborist crash it's a known npm bug — clears "
                            "via a lockfile bump"
                        )
                    issues.append(
                        f"{label} has {total} npm "
                        f"{'vulnerability' if total == 1 else 'vulnerabilities'}"
                    )
                else:
                    check_ok(
                        f"{label} deps",
                        f"({moderate} moderate "
                        f"{'vulnerability' if moderate == 1 else 'vulnerabilities'})",
                    )
            except Exception:
                pass

    if _is_termux():
        check_info("Termux compatibility fallbacks:")
        for note in _termux_install_all_fallback_notes():
            check_info(note)

    _section("API Connectivity")
    # Refactor: every connectivity probe below is HTTP-bound and fully
    # independent. Running them in series spent ~5s wall on a typical
    # workstation (2s of that was boto3's IMDS lookup for AWS credentials,
    # which times out unless you're actually on EC2). Threading them with
    # a small executor pool collapses the section to roughly the slowest
    # single probe — about 2s — without changing the output format.
    #
    # Each ``_probe_*`` helper is a pure function: takes its inputs,
    # makes one HTTP/SDK call, returns a ``_ConnectivityResult`` carrying
    # the line(s) to print and any issue strings to append. No globals,
    # no shared mutable state, no printing inside the workers.
    import concurrent.futures as _futures
    from collections import namedtuple as _namedtuple

    _ConnectivityResult = _namedtuple(
        "_ConnectivityResult", ["label", "lines", "issues"]
    )
    _probes: list = []  # list of (label, callable) submitted in display order

    def _probe_openrouter() -> _ConnectivityResult:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("⚠", Colors.YELLOW), "OpenRouter API",
                  color("(not configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            r = httpx.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✓", Colors.GREEN), "OpenRouter API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(invalid API key)", Colors.DIM))],
                    ["Check OPENROUTER_API_KEY in .env"],
                )
            if r.status_code == 402:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(out of credits — payment required)", Colors.DIM))],
                    ["OpenRouter account has insufficient credits. "
                     "Fix: run 'hermes config set model.provider <provider>' "
                     "to switch providers, or fund your OpenRouter account "
                     "at https://openrouter.ai/settings/credits"],
                )
            if r.status_code == 429:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(rate limited)", Colors.DIM))],
                    ["OpenRouter rate limit hit — consider switching to "
                     "a different provider or waiting"],
                )
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"({e})", Colors.DIM))],
                ["Check network connectivity"],
            )

    def _probe_anthropic() -> _ConnectivityResult:
        from hermes_cli.auth import get_anthropic_key
        key = get_anthropic_key()
        if not key:
            return _ConnectivityResult("Anthropic API", [], [])
        try:
            import httpx
            from agent.anthropic_adapter import (
                _is_oauth_token,
                _COMMON_BETAS,
                _OAUTH_ONLY_BETAS,
                _CONTEXT_1M_BETA,
            )
            headers = {"anthropic-version": "2023-06-01"}
            is_oauth = _is_oauth_token(key)
            if is_oauth:
                headers["Authorization"] = f"Bearer {key}"
                headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
            else:
                headers["x-api-key"] = key
            r = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers=headers, timeout=10,
            )
            # Reactive recovery: OAuth subscriptions without 1M context reject the
            # request with 400 "long context beta is not yet available for this
            # subscription". Retry once with that beta stripped so the doctor
            # check doesn't falsely report Anthropic as unreachable.
            if (
                is_oauth
                and r.status_code == 400
                and "long context beta" in r.text.lower()
                and "not yet available" in r.text.lower()
            ):
                headers["anthropic-beta"] = ",".join(
                    [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                    + list(_OAUTH_ONLY_BETAS)
                )
                r = httpx.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✓", Colors.GREEN), "Anthropic API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✗", Colors.RED), "Anthropic API",
                      color("(invalid API key)", Colors.DIM))],
                    [],
                )
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color("(couldn't verify)", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_apikey_provider(pname, env_vars, default_url, base_env,
                               supports_health_check) -> _ConnectivityResult:
        key = ""
        for ev in env_vars:
            key = os.getenv(ev, "")
            if key:
                break
        if not key:
            return _ConnectivityResult(pname, [], [])
        label = pname.ljust(20)
        if not supports_health_check:
            return _ConnectivityResult(
                pname,
                [(color("✓", Colors.GREEN), label,
                  color("(key configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            base = os.getenv(base_env, "") if base_env else ""
            # Auto-detect Kimi Code keys (sk-kimi-) → api.kimi.com/coding/v1
            # (OpenAI-compat surface, which exposes /models for health check).
            if not base and key.startswith("sk-kimi-"):
                base = "https://api.kimi.com/coding/v1"
            # Anthropic-compat endpoints (/anthropic, api.kimi.com/coding
            # with no /v1) don't support /models. Rewrite to OpenAI-compat
            # /v1 surface for health checks.
            if base and base.rstrip("/").endswith("/anthropic"):
                from agent.auxiliary_client import _to_openai_base_url
                base = _to_openai_base_url(base)
            if base_url_host_matches(base, "api.kimi.com") and base.rstrip("/").endswith("/coding"):
                base = base.rstrip("/") + "/v1"
            url = (base.rstrip("/") + "/models") if base else default_url
            headers = {
                "Authorization": f"Bearer {key}",
                "User-Agent": _HERMES_USER_AGENT,
            }
            if base_url_host_matches(base, "api.kimi.com"):
                headers["User-Agent"] = "claude-code/0.1.0"
            # Google's Generative Language API (generativelanguage.googleapis.com)
            # rejects ``Authorization: Bearer <api-key>`` with 401
            # ``ACCESS_TOKEN_TYPE_UNSUPPORTED`` — that header is reserved for
            # OAuth 2 access tokens, not plain API keys. Plain keys use
            # ``x-goog-api-key`` (or ``?key=``). Without this, a perfectly valid
            # GOOGLE_API_KEY/GEMINI_API_KEY always shows red in ``hermes doctor``.
            if url and base_url_host_matches(url, "generativelanguage.googleapis.com"):
                headers.pop("Authorization", None)
                headers["x-goog-api-key"] = key
            r = httpx.get(url, headers=headers, timeout=10)
            if (
                pname == "Alibaba/DashScope"
                and not base
                and r.status_code == 401
            ):
                r = httpx.get(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    pname,
                    [(color("✓", Colors.GREEN), label, "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    pname,
                    [(color("✗", Colors.RED), label,
                      color("(invalid API key)", Colors.DIM))],
                    [f"Check {env_vars[0]} in .env"],
                )
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_bedrock() -> _ConnectivityResult:
        try:
            from agent.bedrock_adapter import (
                has_aws_credentials,
                resolve_aws_auth_env_var,
                resolve_bedrock_region,
            )
        except ImportError:
            return _ConnectivityResult("AWS Bedrock", [], [])
        if not has_aws_credentials():
            return _ConnectivityResult("AWS Bedrock", [], [])
        auth_var = resolve_aws_auth_env_var()
        region = resolve_bedrock_region()
        label = "AWS Bedrock".ljust(20)
        try:
            import boto3
            from botocore.config import Config as _BotoConfig
            # Trim retries on the actual Bedrock API call so a transient
            # failure doesn't pad the doctor run by 30+ seconds.
            cfg = _BotoConfig(
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 1},
            )
            client = boto3.client("bedrock", region_name=region, config=cfg)
            resp = client.list_foundation_models()
            n = len(resp.get("modelSummaries", []))
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("✓", Colors.GREEN), label,
                  color(f"({auth_var}, {region}, {n} models)", Colors.DIM))],
                [],
            )
        except ImportError:
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(boto3 not installed — {sys.executable} -m pip install boto3)",
                        Colors.DIM))],
                [f"Install boto3 for Bedrock: {sys.executable} -m pip install boto3"],
            )
        except Exception as e:
            err_name = type(e).__name__
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({err_name}: {e})", Colors.DIM))],
                [f"AWS Bedrock: {err_name} — check IAM permissions for "
                 f"bedrock:ListFoundationModels"],
            )

    def _probe_azure_entra() -> _ConnectivityResult:
        """Probe Azure Foundry Entra ID auth, parallel to ``_probe_bedrock``.

        Skipped unless the active config has ``model.provider:
        azure-foundry`` AND ``model.auth_mode: entra_id`` — we don't probe
        the token-service / CLI chain for users on plain API-key Azure.

        Bounded by a 10s timeout (via
        :func:`agent.azure_identity_adapter.describe_active_credential`)
        so a slow token service can't pad the doctor run.
        """
        label = "Azure Foundry (Entra ID)".ljust(28)
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
            if not isinstance(model_cfg, dict):
                return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            auth_mode = str(model_cfg.get("auth_mode") or "").strip().lower()
            if cfg_provider != "azure-foundry" or auth_mode != "entra_id":
                return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])
        except Exception:
            return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])

        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig,
                SCOPE_AI_AZURE_DEFAULT,
                describe_active_credential,
                has_azure_identity_installed,
            )
        except Exception as exc:
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(adapter import failed: {exc})", Colors.DIM))],
                [f"Azure Foundry adapter import failed: {exc}"],
            )

        if not has_azure_identity_installed():
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("⚠", Colors.YELLOW), label,
                  color("(azure-identity not installed)", Colors.DIM))],
                [f"Install azure-identity: {sys.executable} -m pip install azure-identity"],
            )

        entra_cfg = model_cfg.get("entra") or {}
        if not isinstance(entra_cfg, dict):
            entra_cfg = {}
        scope = (
            str(entra_cfg.get("scope") or "").strip()
            or SCOPE_AI_AZURE_DEFAULT
        )
        config = EntraIdentityConfig(
            scope=scope,
        )
        info = describe_active_credential(config=config, timeout_seconds=10.0)
        if info.get("ok"):
            env_sources = info.get("env_sources") or []
            tag = ", ".join(env_sources) if env_sources else "default credential chain"
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("✓", Colors.GREEN), label,
                  color(f"({tag}, scope={scope})", Colors.DIM))],
                [],
            )
        err = info.get("error") or "credential chain exhausted"
        hint = info.get("hint") or (
            "Run `az login`, set AZURE_TENANT_ID/AZURE_CLIENT_ID/"
            "AZURE_CLIENT_SECRET, or attach a managed identity to this VM."
        )
        return _ConnectivityResult(
            "Azure Foundry (Entra ID)",
            [(color("⚠", Colors.YELLOW), label,
              color(f"({err})", Colors.DIM))],
            [f"Azure Foundry Entra: {err}. {hint}"],
        )

    # Build the probe submission list in display order
    _probes.append(("OpenRouter API", _probe_openrouter))
    _probes.append(("Anthropic API", _probe_anthropic))

    global _APIKEY_PROVIDERS_CACHE
    if _APIKEY_PROVIDERS_CACHE is None:
        _APIKEY_PROVIDERS_CACHE = _build_apikey_providers_list()
    for _entry in _APIKEY_PROVIDERS_CACHE:
        _pname, _env_vars, _default_url, _base_env, _supports = _entry
        # Capture loop vars by binding default args — without this, all closures
        # would share the final iteration's values and every probe would hit
        # the last provider's URL.
        _probes.append((_pname, lambda p=_pname, e=_env_vars, u=_default_url,
                                       b=_base_env, s=_supports:
                                _probe_apikey_provider(p, e, u, b, s)))

    _probes.append(("AWS Bedrock", _probe_bedrock))
    _probes.append(("Azure Foundry (Entra ID)", _probe_azure_entra))

    # Print a single status line so users see something happening, then
    # fan out. ``\r`` clears it once the first real result line lands.
    print(f"  {color(f'Running {len(_probes)} connectivity checks in parallel…', Colors.DIM)}",
          end="", flush=True)

    # Disable boto3's EC2 instance-metadata-service probe for the duration
    # of the parallel block. boto's default credential chain tries
    # 169.254.169.254 with a multi-second timeout when we're not on EC2,
    # which dominated the section's wall time before this fix
    # (~2s on a developer laptop, even with the rest parallelized).
    # Set on the parent thread before submitting work so the env-var
    # mutation never races with another worker. has_aws_credentials() in
    # the bedrock probe already gates on real env-var creds, so IMDS is
    # never the legitimate source for `hermes doctor`.
    _imds_prev = os.environ.get("AWS_EC2_METADATA_DISABLED")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    try:
        # 8 workers is plenty — each probe is a single HTTP call plus a TLS
        # handshake. More than that wastes thread-startup cost and risks
        # noisy output if anything ever printed from inside a worker.
        with _futures.ThreadPoolExecutor(max_workers=8,
                                         thread_name_prefix="doctor-probe") as _ex:
            _futures_in_order = [_ex.submit(_fn) for _, _fn in _probes]
            _results = [_f.result() for _f in _futures_in_order]
    finally:
        if _imds_prev is None:
            os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
        else:
            os.environ["AWS_EC2_METADATA_DISABLED"] = _imds_prev

    # Clear the "Running …" line and print all results in submission order.
    print("\r" + " " * 70 + "\r", end="")
    for _r in _results:
        for _glyph, _label, _detail in _r.lines:
            if _detail:
                print(f"  {_glyph} {_label} {_detail}")
            else:
                print(f"  {_glyph} {_label}")
        _issues_to_add = list(_r.issues)
        if _issues_to_add and _has_healthy_oauth_fallback_for_apikey_provider(_r.label):
            _issues_to_add = []
        for _issue in _issues_to_add:
            issues.append(_issue)

    _section("Tool Availability")
    try:
        # Add project root to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
        
        available, unavailable = check_tool_availability()
        available, unavailable = _apply_doctor_tool_availability_overrides(available, unavailable)

        # Web is split into search/extract readiness rows so an explicitly
        # selected but unconfigured backend cannot look healthy (#78412).
        web_rows = []
        if "web" in available or any(item.get("name") == "web" for item in unavailable):
            web_rows = _doctor_web_capability_rows()
            if web_rows:
                available = [tid for tid in available if tid != "web"]
                unavailable = [item for item in unavailable if item.get("name") != "web"]

        for tid in available:
            info = TOOLSET_REQUIREMENTS.get(tid, {})
            check_ok(info.get("name", tid), _doctor_tool_availability_detail(tid))

        for status, label, detail in web_rows:
            if status == "ok":
                check_ok(label, detail)
            else:
                check_warn(label, detail)

        for item in unavailable:
            env_vars = item.get("missing_vars") or item.get("env_vars") or []
            if env_vars:
                vars_str = ", ".join(env_vars)
                check_warn(item["name"], f"(missing {vars_str})")
            else:
                check_warn(item["name"], "(system dependency not met)")

        # Count missing API-key requirements only for toolsets enabled in the
        # current CLI platform. Default-off or explicitly disabled toolsets may
        # still show warnings above, but should not pollute the final summary.
        api_disabled = _missing_api_key_toolsets_for_summary(unavailable)
        web_not_ready = any(status != "ok" for status, _, _ in web_rows)
        if api_disabled or web_not_ready:
            issues.append("Run 'hermes setup' to configure missing API keys for full tool access")
    except Exception as e:
        check_warn("Could not check tool availability", f"({e})")
    
    _section("Skills Hub")
    hub_dir = HERMES_HOME / "skills" / ".hub"
    if hub_dir.exists():
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                count = len(lock_data.get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    else:
        check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")

    from hermes_cli.config import get_env_value

    def _gh_authenticated() -> bool:
        """Check if gh CLI is authenticated via token file or device flow."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status", "--json", "authenticated"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    github_token = get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN")
    if github_token:
        check_ok("GitHub token configured (authenticated API access)")
    elif _gh_authenticated():
        check_ok("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)")
    else:
        check_warn("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)")

    _section("Memory Provider")
    _active_memory_provider = _doctor_memory_config().get("provider", "")

    if not _active_memory_provider:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
    elif _active_memory_provider == "honcho":
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
            hcfg = HonchoClientConfig.from_global_config()
            _honcho_cfg_path = resolve_config_path()

            if not _honcho_cfg_path.exists():
                # Config file missing — but env var fallback may have resolved it.
                # Only warn if the config didn't actually resolve from env vars.
                if hcfg.api_key or hcfg.base_url:
                    check_ok(
                        "Honcho configured via environment variables",
                        f"config file {_honcho_cfg_path} not found, using HONCHO_API_KEY env var",
                    )
                else:
                    check_warn("Honcho config not found", "run: hermes memory setup")
            elif not hcfg.enabled:
                check_info(f"Honcho disabled (set enabled: true in {_honcho_cfg_path} to activate)")
            elif not (hcfg.api_key or hcfg.base_url):
                _fail_and_issue(
                    "Honcho API key or base URL not set",
                    "run: hermes memory setup",
                    "No Honcho API key — run 'hermes memory setup'",
                    issues,
                )
            else:
                from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
                reset_honcho_client()
                try:
                    get_honcho_client(hcfg)
                    check_ok(
                        "Honcho connected",
                        f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}",
                    )
                except Exception as _e:
                    _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)
        except ImportError:
            _fail_and_issue(
                "honcho-ai not installed",
                "pip install honcho-ai",
                "Honcho is set as memory provider but honcho-ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Honcho check failed", str(_e))
    elif _active_memory_provider == "mem0":
        try:
            from plugins.memory.mem0 import _load_config as _load_mem0_config
            mem0_cfg = _load_mem0_config()
            mem0_key = mem0_cfg.get("api_key", "")
            if mem0_key:
                check_ok("Mem0 API key configured")
                check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
            else:
                _fail_and_issue(
                    "Mem0 API key not set",
                    "(set MEM0_API_KEY in .env or run hermes memory setup)",
                    "Mem0 is set as memory provider but API key is missing",
                    issues,
                )
        except ImportError:
            _fail_and_issue(
                "Mem0 plugin not loadable",
                "pip install mem0ai",
                "Mem0 is set as memory provider but mem0ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Mem0 check failed", str(_e))
    else:
        # Generic check for other memory providers (openviking, hindsight, etc.)
        try:
            from plugins.memory import load_memory_provider
            _provider = load_memory_provider(_active_memory_provider)
            if _provider and _provider.is_available():
                check_ok(f"{_active_memory_provider} provider active")
            elif _provider:
                check_warn(f"{_active_memory_provider} configured but not available", "run: hermes memory status")
            else:
                check_warn(f"{_active_memory_provider} plugin not found", "run: hermes memory setup")
        except Exception as _e:
            check_warn(f"{_active_memory_provider} check failed", str(_e))

    try:
        from hermes_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
        import re as _re

        named_profiles = [p for p in list_profiles() if not p.is_default]
        if named_profiles:
            _section("Profiles")
            check_ok(f"{len(named_profiles)} profile(s) found")
            wrapper_dir = _get_wrapper_dir()
            for p in named_profiles:
                parts = []
                if p.gateway_running:
                    parts.append("gateway running")
                if p.model:
                    parts.append(p.model[:30])
                if not (p.path / "config.yaml").exists():
                    parts.append("⚠ missing config")
                if not (p.path / ".env").exists():
                    parts.append("no .env")
                wrapper = wrapper_dir / p.name
                if not wrapper.exists():
                    parts.append("no alias")
                status = ", ".join(parts) if parts else "configured"
                check_ok(f"  {p.name}: {status}")

            # Check for orphan wrappers
            if wrapper_dir.is_dir():
                for wrapper in wrapper_dir.iterdir():
                    if not wrapper.is_file():
                        continue
                    try:
                        content = wrapper.read_text(encoding="utf-8")
                        if "hermes -p" in content:
                            _m = _re.search(r"hermes -p (\S+)", content)
                            if _m and not profile_exists(_m.group(1)):
                                check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
                    except Exception:
                        pass
    except ImportError:
        pass
    except Exception:
        pass

    # Opt-in live backend probes run AFTER all static checks, only with
    # `hermes doctor --live` (real network calls; bounded + read-only).
    try:
        from hermes_cli.doctor_live import maybe_run_live_checks
        maybe_run_live_checks(args, manual_issues)
    except Exception:
        pass

    print()
    remaining_issues = issues + manual_issues
    if should_fix and fixed_count > 0:
        print(color("─" * 60, Colors.GREEN))
        print(color(f"  Fixed {fixed_count} issue(s).", Colors.GREEN, Colors.BOLD), end="")
        if remaining_issues:
            print(color(f" {len(remaining_issues)} issue(s) require manual intervention.", Colors.YELLOW, Colors.BOLD))
        else:
            print()
        print()
        if remaining_issues:
            for i, issue in enumerate(remaining_issues, 1):
                print(f"  {i}. {issue}")
            print()
    elif remaining_issues:
        print(color("─" * 60, Colors.YELLOW))
        print(color(f"  Found {len(remaining_issues)} issue(s) to address:", Colors.YELLOW, Colors.BOLD))
        print()
        for i, issue in enumerate(remaining_issues, 1):
            print(f"  {i}. {issue}")
        print()
        if not should_fix:
            print(color("  Tip: run 'hermes doctor --fix' to auto-fix what's possible.", Colors.DIM))
    else:
        print(color("─" * 60, Colors.GREEN))
        print(color("  All checks passed! 🎉", Colors.GREEN, Colors.BOLD))
    
    print()
