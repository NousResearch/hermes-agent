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
    "agent.account_usage", "agent.acp_client", "agent.browser_provider", "agent.context_engine", "agent.credential_pool",
    "agent.delegation_backends", "agent.delegation_provider", "hermes_cli.auth_model_picker:prompt_model_selection",
    "hermes_cli.auth:resolve_external_process_provider_credentials", "hermes_cli.model_setup",
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
    # plugins/index.md § platform plugins: read session identity through the task-local context
    "gateway.session_context:get_session_env",
})

_SHOWN = 10


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


@lru_cache(maxsize=512)
def _core_module_exists(module: str) -> bool:
    path = Path(__file__).resolve().parents[1].joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def _is_own_module_path(module: str, local: set[str]) -> bool:
    """A Hermes-shaped path that core does not ship but whose tail is one of the plugin's own modules:
    the fallback spelling of the plugin's install location (``plugins.memory.openltm._db`` beside
    ``from . import _db``). A path core DOES ship stays a core import."""
    parts = module.split(".")
    return any(".".join(parts[i:]) in local for i in range(1, len(parts))) and not _core_module_exists(module)


def _private_target(module: str, name: str, core: frozenset, local: set[str]) -> str:
    """The dotted target to report, or "" when the import is public, local or not Hermes's."""
    target = f"{module}.{name}" if name else module
    own = _is_own_module_path(module, local) or bool(name) and _is_own_module_path(target, local)
    hermes = module.split(".")[0] in core and module not in local and not own
    public = target in _PUBLIC or f"{module}:{name}" in _PUBLIC \
        or (module in _PUBLIC and not name.startswith("_"))  # ``from gateway.platforms import base`` too
    return target if hermes and not public else ""


def _parsed_files(plugin_dir: Path) -> list[tuple[Path, ast.AST]]:
    """``(rel, tree)`` for every plugin module that can load inside Hermes. Hidden dirs (``.github/``
    CI helpers) never do; tests and vendored envs are skipped by ``_python_files``."""
    parsed = []
    for path, rel in _python_files(plugin_dir):
        if any(p.startswith(".") for p in rel.parts[:-1]):
            continue
        try:
            parsed.append((rel, ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))))
        except (SyntaxError, ValueError):
            continue
    return parsed


def private_import_findings(plugin_dir: Path) -> list[str]:
    """``["<rel>:<line> <module>[.<name>]", ...]`` for every import of a private Hermes core module:
    the first site of each distinct module first (so a truncated list still shows the breadth),
    then the rest, each group in file/line order."""
    files = _parsed_files(Path(plugin_dir))
    local = _local_modules(rel for rel, _tree in files)
    core = _hermes_packages()
    found: set[tuple[str, int, str, str]] = set()
    for rel, tree in files:
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
    return [f"{rel}:{line} {target}" for rel, line, target, _module in first + rest]


# The core objects a plugin holds a live handle to — ``SessionDB`` (``ctx`` session APIs, the
# dashboard), ``AIAgent`` and ``GatewayRunner`` (hook kwargs), ``PluginContext`` — split across their
# facade + sibling files. Calling one of their private methods (``db._execute_write(...)``,
# ``runner._evict_cached_agent(...)``) couples the plugin to core internals exactly like a private
# import, but no import names it, so the import pass above cannot see it.
_CORE_HANDLE_SOURCES = ("hermes_state.py", "hermes_state_*.py", "run_agent.py", "agent/turn_*.py",
                        "gateway/run.py", "gateway/run_*.py", "hermes_cli/plugins.py")


@lru_cache(maxsize=1)
def _core_private_methods() -> frozenset:
    root = Path(__file__).resolve().parents[1]
    names: set[str] = set()
    for pattern in _CORE_HANDLE_SOURCES:
        for path in root.glob(pattern):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (OSError, SyntaxError, ValueError):
                continue
            names |= {item.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for item in node.body
                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and item.name.startswith("_") and not item.name.startswith("__")}
    return frozenset(names)


def _own_names(files: list[tuple[Path, ast.AST]]) -> set[str]:
    """Every name the plugin defines or assigns, so its own ``obj._helper()`` never reads as core."""
    own: set[str] = set()
    for _rel, tree in files:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                own.add(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                own.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                own.add(node.id)
    return own


def private_attribute_findings(plugin_dir: Path) -> list[str]:
    """``["<rel>:<line> <expr>", ...]`` for every access to a private method of a core handle class
    that the plugin does not define itself (``self``/``cls`` receivers excluded)."""
    files = _parsed_files(Path(plugin_dir))
    private = _core_private_methods() - _own_names(files)
    found = {(rel.as_posix(), node.lineno, ast.unparse(node)[:80]) for rel, tree in files for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and node.attr in private
             and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))}
    return [f"{rel}:{line} {expr}" for rel, line, expr in sorted(found)]


def _warn_sites(report, label: str, hits: list[str]) -> None:
    more = f"; and {len(hits) - _SHOWN} more" if len(hits) > _SHOWN else ""
    report.warn(
        f"{label} (catalog rule 9: public surfaces only; these can break on any release): "
        + "; ".join(hits[:_SHOWN]) + more
        + ". Use ctx.* / documented plugin APIs, or open an issue for the seam you need."
    )


def check_private_imports(report, plugin_dir: Path) -> None:
    """Warn (never fail) when the plugin imports private Hermes core modules or calls private
    methods on the core objects it is handed."""
    if hits := private_import_findings(plugin_dir):
        _warn_sites(report, "private core imports", hits)
    if hits := private_attribute_findings(plugin_dir):
        _warn_sites(report, "private core attribute access", hits)
