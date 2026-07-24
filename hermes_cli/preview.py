"""Zero-side-effect Hermes runtime configuration preview."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import get_hermes_home, get_env_path, get_env_value, load_config
from hermes_cli.colors import Colors, color
from hermes_cli.models import provider_label
from hermes_constants import display_hermes_home

HERMES_HOME = get_hermes_home()


# ─────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────


@dataclass
class PreviewItem:
    name: str
    status: str  # "ok" | "warn" | "error"
    detail: str = ""
    fix: str = ""


@dataclass
class PreviewReport:
    verdict: str  # "ready" | "warning" | "blocked"
    items: List[PreviewItem] = field(default_factory=list)

    def add_ok(self, name: str, detail: str = "") -> None:
        self.items.append(PreviewItem(name, "ok", detail))

    def add_warn(self, name: str, detail: str = "", fix: str = "") -> None:
        self.items.append(PreviewItem(name, "warn", detail, fix))

    def add_error(self, name: str, detail: str = "", fix: str = "") -> None:
        self.items.append(PreviewItem(name, "error", detail, fix))

    def json_output(self) -> str:
        """Return machine-readable JSON representation of the report."""
        return json.dumps(
            {
                "verdict": self.verdict,
                "items": [
                    {
                        "name": it.name,
                        "status": it.status,
                        "detail": it.detail,
                        "fix": it.fix,
                    }
                    for it in self.items
                ],
            },
            ensure_ascii=False,
            indent=2,
        )


# ─────────────────────────────────────────────────────────
# Provider → auth env-var mapping (only match the key for
# the configured provider, not every key in the process).
# ─────────────────────────────────────────────────────────

_PROVIDER_AUTH_KEYS: Dict[str, List[str]] = {
    "deepseek": ["DEEPSEEK_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
    "nous": ["NOUS_API_KEY"],
    "glm": ["GLM_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "dashscope": ["DASHSCOPE_API_KEY"],
    "tokenhub": ["TOKENHUB_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "sensenova": ["SENSENOVA_API_KEY"],
    "custom": ["OPENAI_API_KEY", "OPENAI_BASE_URL"],  # common patterns
}


def _resolve_provider_info() -> Tuple[str, str, str]:
    """Return (model_name, provider_key, provider_label).

    Reads config.yaml once; cached result avoids triple-loading in
    run_preview().
    """
    try:
        config = load_config()
    except Exception:
        return "(not set)", "", "(not set)"

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model = (model_cfg.get("default") or model_cfg.get("name") or "").strip()
        provider_key = (model_cfg.get("provider") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()
        provider_key = ""
    else:
        model = ""
        provider_key = ""

    model_name = model or "(not set)"

    if provider_key:
        provider_label_str = provider_label(provider_key)
    else:
        # Try to resolve from runtime (without reloading config)
        try:
            from hermes_cli.auth import resolve_provider
            from hermes_cli.runtime_provider import resolve_requested_provider

            requested = resolve_requested_provider()
            try:
                resolved = resolve_provider(requested)
                provider_key = resolved or provider_key
                provider_label_str = provider_label(provider_key)
            except Exception:
                provider_label_str = provider_label(requested or "auto")
        except Exception:
            provider_label_str = "(unknown)"

    return model_name, provider_key, provider_label_str


def _check_auth_for_provider(provider_key: str) -> Tuple[str, str]:
    """Check auth specifically for the configured provider.

    Only checks API keys relevant to the given provider_key, so a
    DEEPSEEK_API_KEY won't falsely satisfy an OpenAI provider.
    """
    # 1. Check provider-specific env vars
    for env_var in _PROVIDER_AUTH_KEYS.get(provider_key, []):
        val = get_env_value(env_var)
        if val and len(val) > 5:
            return "ok", f"Authenticated via {env_var}"

    # 2. Check OAuth (provider-independent; applies to nous/codex)
    try:
        from hermes_cli.auth import get_codex_auth_status, get_nous_auth_status

        nous = get_nous_auth_status()
        codex = get_codex_auth_status()
        if nous.get("logged_in") or nous.get("inference_credential_present"):
            return "ok", "Nous Portal authenticated"
        if codex.get("logged_in"):
            return "ok", "OpenAI Codex OAuth authenticated"
    except Exception:
        pass

    # 3. Custom provider — check if base_url is configured
    if provider_key.startswith("custom"):
        try:
            config = load_config()
            model_cfg = config.get("model")
            if isinstance(model_cfg, dict):
                base_url = (model_cfg.get("base_url") or "").strip()
                if base_url:
                    truncated = (
                        base_url[:30] + "..." if len(base_url) > 30 else base_url
                    )
                    return "ok", f"Custom provider configured ({truncated})"
        except Exception:
            pass

    # 4. Fallback: custom/auto with no verifiable key
    if provider_key in ("custom", "auto", ""):
        return "ok", "Provider configured (auth assumed)"

    return "error", "No API key found"


# ─────────────────────────────────────────────────────────
# Check functions (pure — no side effects)
# ─────────────────────────────────────────────────────────


def _get_configured_model() -> Tuple[str, str]:
    """Return (model_name, provider_label) from config.yaml."""
    model_name, provider_key, provider_label_str = _resolve_provider_info()
    return model_name, provider_label_str


def _check_auth() -> Tuple[str, str]:
    """Check if the effective provider has authentication.

    Returns (status, detail).  status: "ok" | "error" | "unknown".
    Only checks API keys for the configured provider, not all keys.
    """
    model_name, provider_key, _ = _resolve_provider_info()

    if model_name == "(not set)":
        return "error", "No model configured"

    return _check_auth_for_provider(provider_key)


def _check_skills() -> Tuple[int, int]:
    """Count available skills (bundled + user + project).

    Returns (total_count, error_count).
    """
    try:
        from hermes_cli.skills import discover_skills

        skills = discover_skills()
        return len(skills), 0
    except Exception:
        return 0, 1


def _check_tools() -> Tuple[int, int]:
    """Count available tools from tool backend.

    Returns (count, error_count).  Tools load lazily, so 0 is normal.
    """
    try:
        from tools.tool_backend import get_tool_backend

        backend = get_tool_backend()
        tools = backend.get_tools()
        return len(tools), 0
    except Exception:
        return 0, 0  # Don't fail on this; tools are loaded lazily


def _check_mcp() -> Tuple[int, int, int]:
    """Count MCP servers: configured / connected / disconnected.

    Returns (total, connected, disconnected).  Only counts explicit
    "connected" / "disconnected" statuses; intermediate states are
    reported separately.
    """
    try:
        from tools.mcp_server import discover_mcp_servers

        servers = discover_mcp_servers()
        total = len(servers)
        connected = sum(1 for s in servers if s.get("status") == "connected")
        disconnected = sum(1 for s in servers if s.get("status") == "disconnected")
        return total, connected, disconnected
    except Exception:
        return 0, 0, 0


def _check_python() -> Tuple[str, str]:
    """Check Python version."""
    version = sys.version.split()[0]
    try:
        major, minor = map(int, version.split(".")[:2])
        if major >= 3 and minor >= 10:
            return "ok", f"Python {version}"
        else:
            return "warn", f"Python {version} (minimum recommended: 3.10+)"
    except Exception:
        return "warn", f"Python {version} (version could not be parsed)"


def _check_hermes_home() -> Tuple[str, str, str]:
    """Check Hermes home directory exists. Returns (status, detail, fix)."""
    if HERMES_HOME.exists():
        return "ok", f"Hermes home at {display_hermes_home()}", ""
    else:
        return "error", "Hermes home does not exist", "Run 'hermes setup' to initialize"


def _check_env_file() -> Tuple[str, str, str]:
    """Check .env file exists. Returns (status, detail, fix)."""
    env_path = get_env_path()
    if env_path.exists():
        return "ok", ".env file found", ""
    else:
        return (
            "warn",
            ".env file not found",
            "Create .env with your API keys, or use `hermes auth add`",
        )


# ─────────────────────────────────────────────────────────
# Main check function
# ─────────────────────────────────────────────────────────


def run_preview() -> PreviewReport:
    """Run all preview checks and return a report."""
    report = PreviewReport("ready")

    # 1. Hermes home
    status, detail, fix = _check_hermes_home()
    if status == "ok":
        report.add_ok("Hermes home", detail)
    elif status == "error":
        report.add_error("Hermes home", detail, fix=fix)
        report.verdict = "blocked"
    else:
        report.add_warn("Hermes home", detail, fix=fix)

    # 2. Python version
    status, detail = _check_python()
    if status == "ok":
        report.add_ok("Python", detail)
    else:
        report.add_warn("Python", detail)

    # 3. Model / Provider
    model_name, provider_label_str = _get_configured_model()
    if model_name == "(not set)":
        report.add_error(
            "Model", "No model configured", "Run 'hermes model' to set a model"
        )
        report.verdict = "blocked"
    else:
        report.add_ok("Model", f"{model_name} (provider: {provider_label_str})")

    # 4. Auth (checks only keys for the configured provider)
    auth_status, auth_detail = _check_auth()
    if auth_status == "ok":
        report.add_ok("Auth", auth_detail)
    elif auth_status == "error":
        report.add_error(
            "Auth",
            auth_detail,
            "Configure API key via 'hermes auth add' or set env var",
        )
        report.verdict = "blocked"
    else:
        report.add_warn("Auth", auth_detail)

    # 5. .env file
    env_status, env_detail, env_fix = _check_env_file()
    if env_status == "ok":
        report.add_ok(".env file", env_detail)
    elif env_status == "warn":
        report.add_warn(".env file", env_detail, env_fix)
    else:
        report.add_ok(".env file", env_detail)

    # 6. Skills
    skill_count, _ = _check_skills()
    if skill_count > 0:
        report.add_ok("Skills", f"{skill_count} available")
    else:
        report.add_warn("Skills", "No skills loaded")

    # 7. Tools
    tool_count, _ = _check_tools()
    if tool_count > 0:
        report.add_ok("Tools", f"{tool_count} available")
    else:
        report.add_ok("Tools", "Loaded on demand")

    # 8. MCP
    mcp_total, mcp_connected, mcp_disconnected = _check_mcp()
    if mcp_total == 0:
        report.add_ok("MCP", "No MCP servers configured")
    elif mcp_disconnected == 0:
        report.add_ok("MCP", f"{mcp_total} configured, all connected")
    elif mcp_connected == 0:
        report.add_warn(
            "MCP",
            f"{mcp_total} configured, 0 connected",
            "Check MCP server URLs and auth",
        )
    else:
        report.add_warn(
            "MCP",
            f"{mcp_total} configured, {mcp_connected} connected, {mcp_disconnected} disconnected",
        )

    return report


# ─────────────────────────────────────────────────────────
# Display functions
# ─────────────────────────────────────────────────────────


def _check_mark(ok: bool) -> str:
    if ok:
        return color("[OK]", Colors.GREEN)
    return color("[!!]", Colors.RED)


def display_report(report: PreviewReport) -> None:
    """Display a PreviewReport in human-readable format."""

    verdict_map = {
        "ready": ("All configuration normal", Colors.GREEN),
        "warning": ("Some issues detected", Colors.YELLOW),
        "blocked": ("Cannot run Hermes", Colors.RED),
    }

    label, label_color = verdict_map.get(report.verdict, ("Unknown", Colors.DIM))

    print()
    print(color("--- Hermes Preview - Runtime Check ---", Colors.CYAN))
    print()

    for item in report.items:
        if item.status == "ok":
            mark = color("[OK]", Colors.GREEN)
            detail_str = f" {color(item.detail, Colors.DIM)}" if item.detail else ""
            print(f"  {mark} {item.name}{detail_str}")
        elif item.status == "warn":
            mark = color("[!!]", Colors.YELLOW)
            detail_str = f" {color(item.detail, Colors.DIM)}" if item.detail else ""
            fix_str = f" {color('[fix] ' + item.fix, Colors.DIM)}" if item.fix else ""
            print(f"  {mark} {item.name}{detail_str}{fix_str}")
        else:  # error
            mark = color("[XX]", Colors.RED)
            detail_str = f" {color(item.detail, Colors.DIM)}" if item.detail else ""
            fix_str = f" {color('[fix] ' + item.fix, Colors.DIM)}" if item.fix else ""
            print(f"  {mark} {item.name}{detail_str}{fix_str}")

    print()
    print(color(f">> {label}", label_color, Colors.BOLD))
    print()


def cmd_preview(args: Any) -> None:
    """Entry point for 'hermes preview' command."""
    report = run_preview()
    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        print(report.json_output())
    else:
        display_report(report)


# (build_preview_parser lives in hermes_cli/subcommands/preview.py)
