"""Admission lint: name a plugin's imports of private Hermes core modules (a warning, never a refusal).

Catalog rule 9 ("public surfaces only") is enforced for runtime rebinds by
``plugin_validate_core_override``; an import of a core internal (``tools.tirith_security``,
``hermes_cli.cli_config_load._parse_service_tier_config``) passes that check yet breaks the plugin on
the release that moves or removes the internal. This static pass lists those imports, with
file:line, so reviewers see the coupling at admission. Imports of the plugin's own modules, the
standard library and third-party packages are never flagged; neither are ``TYPE_CHECKING`` imports.
"""

from __future__ import annotations

import ast
import tomllib
from functools import lru_cache
from pathlib import Path

from hermes_cli.plugin_validate_core_override import _hermes_top_level, _python_files

# WHY: "public" must mean what Hermes documents for plugin authors, not a reviewer's taste.
# Derived from every Hermes-module ``import`` in website/docs/developer-guide/**/*.md and
# website/docs/user-guide/features/plugins*.md code blocks (plus extending-the-dashboard.md, which
# desktop-plugin-sdk.md names as the plugin backend reference), and what PluginContext hands out in
# hermes_cli/plugins.py (property/return types, base classes its ``register_*`` methods check), and
# the names those plugin pages tell authors to call, raise, return or subclass in prose. ``module`` entries allow the module (underscore names excepted);
# ``module:Name`` entries allow one name. Re-harvest when those docs or PluginContext change.
_PUBLIC = frozenset({
    # developer-guide / plugins docs
    "agent.account_usage", "agent.browser_provider", "agent.context_engine", "agent.credential_pool",
    "agent.image_gen_provider", "agent.memory_manager", "agent.memory_provider", "agent.secret_scope",
    "agent.secret_sources.base", "agent.subagent_lifecycle", "agent.terminal_env_provider",
    "agent.transports", "agent.transports.chat_completions", "agent.video_gen_provider",
    "agent.web_search_provider", "cli", "gateway.config", "gateway.platform_registry",
    "gateway.platforms._shared", "gateway.platforms.base", "gateway.platforms.event",
    "gateway.platforms.helpers", "gateway.status", "hermes_cli.auth_oauth_pkce_plugin",
    "hermes_cli.config", "hermes_cli.plugin_events", "hermes_cli.plugin_provider_requests",
    "hermes_constants", "hermes_platform", "hermes_state", "plugins.plugin_storage", "plugins.plugin_utils", "pm",
    "providers", "providers.base", "tools.browser_supervisor", "tools.browser_supervisor_capture",
    "tools.computer_use.backend", "tools.registry",
    # PluginContext (hermes_cli/plugins.py)
    "agent.context_references", "agent.plugin_llm", "agent.transcription_provider", "agent.tts_provider",
    "cron.execution_identity",
    "hermes_cli.dashboard_auth", "hermes_cli.platform_actions", "hermes_cli.plugins:PluginContext",
    "hermes_cli.plugins_ledger:PluginRegistration", "hermes_cli.plugins_state:PluginState",
    # prose in memory-provider / model-provider / terminal-environment plugin pages
    "agent.runtime_cwd:resolve_agent_cwd", "agent.usage_pricing:CostResult", "hermes_cli.auth_constants:AuthError",
    "tools.environments.base:BaseEnvironment", "tools.environments:BaseEnvironment",  # package ``__all__``
    "utils:atomic_json_write", "utils:read_json_or_empty",
    # adding-platform-adapters.md § Public helpers for platform plugins
    "gateway.platforms.access_policy_mixin:OwnAccessPolicyMixin", "tools.approval:resolve_gateway_approval",
    "tools.clarify_gateway:get_pending", "tools.clarify_gateway:resolve_gateway_clarify",
    "tools.clarify_gateway:mark_awaiting_text", "tools.clarify_gateway:SKIPPED",
    "tools.url_safety:is_safe_url", "tools.url_safety:create_ssrf_safe_async_client",
    "gateway.platforms._http_client_limits:platform_httpx_limits",
    "hermes_cli.gateway_setup_wizard:print_setup_header", "hermes_cli.gateway_setup_wizard:confirm_reconfigure",
    "hermes_cli.gateway_setup_wizard:offer_home_channel", "hermes_cli.gateway_setup_wizard:save_env_values",
    "hermes_cli.gateway_setup_wizard:prompt_csv", "hermes_cli.setup:prompt", "hermes_cli.setup:prompt_choice",
    "hermes_cli.setup:prompt_yes_no", "hermes_cli.cli_output",
    # plugins/index.md § Read the skill catalog
    "tools.skills_tool:find_all_skills", "tools.skills_tool:sort_skills", "tools.skills_tool:skill_search_dirs",
    "tools.skills_tool:locate_skill", "tools.skills_tool:skill_lookup_path_error",
    "tools.skills_sync:dir_hash", "tools.skill_usage:read_skill_name",
})

_SHOWN = 10

# Private names that now have a public ctx replacement: the warning names it.
_REPLACEMENTS = {
    "hermes_cli.models._provider_has_credentials": "ctx.models.ready(provider)",
    "hermes_cli.models.provider_model_ids": "ctx.models.list(provider)",
    "hermes_cli.models.cached_provider_model_ids": "ctx.models.list(provider)",
    "hermes_cli.models.parse_model_input": "ctx.models.parse(spec)",
    "hermes_cli.models.list_available_providers": "ctx.models.providers()",
}


@lru_cache(maxsize=1)
def _hermes_packages() -> frozenset:
    """Hermes-owned top-level names: pyproject's ``packages.find`` includes plus the root modules
    setup.py ships as ``py-modules`` (every root ``*.py``)."""
    root = Path(__file__).resolve().parents[1]
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
        packages = {name.split(".")[0] for name in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]}
    except (OSError, KeyError, ValueError):
        packages = set(_hermes_top_level()) - {"hermes_plugins", "tests", "evals"}
    return frozenset((packages | {p.stem for p in root.glob("*.py")}) - {"setup", "conftest"})


def _local_modules(rels) -> set[str]:
    """Every dotted name under which one of the plugin's own files could be imported."""
    names = set()
    for rel in rels:
        parts = rel.with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        names |= {".".join(parts[start:end]) for start in range(len(parts)) for end in range(start + 1, len(parts) + 1)}
    return names


def _type_checking_nodes(tree: ast.AST) -> set[int]:
    skipped: set[int] = set()
    for node in ast.walk(tree):
        test = getattr(node, "test", None)
        if isinstance(node, ast.If) and (getattr(test, "id", None) or getattr(test, "attr", None)) == "TYPE_CHECKING":
            skipped |= {id(n) for stmt in node.body for n in ast.walk(stmt)}
    return skipped


def _imported(node: ast.AST) -> list[tuple[str, str]]:
    """``(module, name)`` pairs an import node binds; ``name`` is "" for ``import module``."""
    if isinstance(node, ast.Import):
        return [(alias.name, "") for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [(node.module, alias.name) for alias in node.names]
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
            and isinstance(node.args[0].value, str) and ast.unparse(node.func) in ("importlib.import_module", "import_module"):
        return [(node.args[0].value, "")]
    return []


def _private_target(module: str, name: str, core: frozenset, local: set[str]) -> str:
    """The dotted target to report, or "" when the import is public, local or not Hermes's."""
    target = f"{module}.{name}" if name else module
    hermes = module.split(".")[0] in core and module not in local
    public = target in _PUBLIC or f"{module}:{name}" in _PUBLIC \
        or (module in _PUBLIC and not name.startswith("_"))  # ``from gateway.platforms import base`` too
    return target if hermes and not public else ""


def private_import_findings(plugin_dir: Path) -> list[str]:
    """``["<rel>:<line> <module>[.<name>][ (use ctx...)]", ...]`` for every import of a private Hermes core module:
    the first site of each distinct module first (so a truncated list still shows the breadth),
    then the rest, each group in file/line order."""
    plugin_dir = Path(plugin_dir)
    # Hidden dirs (``.github/`` CI helpers) never load inside Hermes.
    files = [(path, rel) for path, rel in _python_files(plugin_dir) if not any(p.startswith(".") for p in rel.parts[:-1])]
    local = _local_modules(rel for _path, rel in files)
    core = _hermes_packages()
    found: set[tuple[str, int, str, str]] = set()
    for path, rel in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        skipped = _type_checking_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            for module, name in _imported(node):
                target = _private_target(module, name, core, local)
                if target:
                    found.add((rel.as_posix(), getattr(node, "lineno", 0), target, module))
    seen: set[str] = set()
    first = [site for site in sorted(found) if site[3] not in seen and not seen.add(site[3])]
    rest = sorted(found.difference(first))
    return [f"{rel}:{line} {target}" + (f" (use {_REPLACEMENTS[target]})" if target in _REPLACEMENTS else "")
            for rel, line, target, _module in first + rest]


def check_private_imports(report, plugin_dir: Path) -> None:
    """Warn (never fail) when the plugin imports private Hermes core modules."""
    hits = private_import_findings(plugin_dir)
    if hits:
        more = f"; and {len(hits) - _SHOWN} more" if len(hits) > _SHOWN else ""
        report.warn(
            "private core imports (catalog rule 9: public surfaces only; these can break on any release): "
            + "; ".join(hits[:_SHOWN]) + more
            + ". Use ctx.* / documented plugin APIs, or open an issue for the seam you need."
        )
