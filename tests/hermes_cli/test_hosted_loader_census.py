"""The loader census: the primary hosted code-provenance guarantee, as a test.

In hosted mode the guarantee that no tenant-reachable file executes in the shared
process is three parts: (1) PRIMARY — every Hermes code path that loads Python from
a path consults the one predicate (:func:`hermes_cli.hosted_code_fence.hosted_code_refusal`)
on the path or module BEFORE it loads, and refuses in hosted mode; (2) the hosted
start refuses to arm unless the interpreter itself started isolated from user and
data site paths (see the fence module's startup-isolation check); (3) the installed
audit hook is defence in depth behind both. This file enforces (1) structurally: an
AST scan of the repository (tests excluded) finds every dynamic code-loading
primitive — ``runpy.run_path``/``runpy.run_module``, ``importlib.util.spec_from_file_location``,
``importlib.machinery`` file loaders, ``zipimport.zipimporter``, ``importlib.import_module``
with a non-literal argument, ``exec``/``compile`` of file contents, ``sys.path`` insertions,
and site-directory processing (``site.addsitedir``/``site.addpackage``, whose ``.pth``
files execute import-time code) — and each hit must either sit in a function that calls
the predicate on the path first, or carry a named exemption explaining why the code it
loads is image-only. A NEW unguarded loader fails this census: the guarantee does not
depend on reviewers noticing a new import site.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Whole development trees that no deployed Hermes process imports. Their loaders run
# in standalone developer interpreters (eval harnesses, build tooling, probe scripts)
# or in a skill script's own subprocess — never in the shared hosted process.
EXEMPT_TREES: dict[str, str] = {
    "evals/": "evaluation harnesses run standalone against checkouts; never imported by a deployed process",
    "scripts/": "development, build, CI and release tooling; never imported by a deployed process",
    "apps/": "desktop app development tooling; not part of the runtime image",
    "ui-tui/scripts/": "standalone probe scripts, each running its own interpreter",
    "skills/": "skill scripts execute in their own interpreter subprocess; their import surface is not the shared process's",
    "optional-skills/": "optional skill scripts execute in their own interpreter subprocess; their import surface is not the shared process's",
}

# Runtime loader sites whose inputs are image-only by construction. Keyed by
# (repo-relative path, qualname of the enclosing function; "<module>" for module
# level). A new loader in a NEW function is never covered by these entries — it
# fails the census until it guards with the predicate or earns its own entry.
EXEMPT_LOADERS: dict[tuple[str, str], str] = {
    ("acp_adapter/entry.py", "main"):
        "the adapter's own project root, derived from this file's location — inside the image by construction",
    ("agent/agent_init.py", "_lazy_headers.factory"):
        "Hermes's own module names, bound as literals at the call sites",
    ("agent/bedrock_adapter.py", "_stale_error_types"):
        "the literal SDK module table in this function",
    ("agent/inline_tool_executors.py", "_call_tool"):
        "core tool module names, bound as literals at every call site",
    ("agent/lazy_forward.py", "lazy_attr"):
        "Hermes's own module names, bound as literals at the call sites",
    ("agent/monitoring/gateway_health_export.py", "_count"):
        "core module names, bound as literals at the call sites",
    ("agent/monitoring/otlp_exporter.py", "_require_sdk"):
        "the literal _SDK_SYMBOLS table in this module",
    ("agent/transports/__init__.py", "_discover_transports"):
        "the literal _TRANSPORT_MODULES table in this module",
    ("cli.py", "_lazy_shim.shim"):
        "Hermes module names, bound as literals at every _lazy_shim call site",
    ("cron/scheduler.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("gateway/__init__.py", "__getattr__"):
        "the literal _EXPORTS table in this module",
    ("gateway/platform_env.py", "bundled_sibling"):
        "the bundled platform plugin's own file under this module's tree — inside the image by construction",
    ("gateway/platforms/base.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("gateway/run.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("gateway/run.py", "_builtin_adapter_import"):
        "adapter module names from the literal _BUILTIN_ADAPTERS table in this module",
    ("gateway/run_agent_cache.py", "GatewayAgentCacheMixin._clear_session_boundary_security_state"):
        "core module names from the literal table in this function",
    ("hermes_bootstrap.py", "harden_import_path"):
        "the bootstrap's own image root, or an operator env override — this function may run nothing beyond the stdlib before the root is on sys.path, so it guards nothing itself; an entry it inserts that resolves outside the image is refused by the armed fence's import-time sys.path check before any import can use it",
    ("hermes_cli/_launchers.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("hermes_cli/_startup_fast.py", "ensure_project_root_on_path"):
        "the project root this module computed from its own location — inside the image by construction",
    ("hermes_cli/cli_commands_session_tools.py", "_probe"):
        "optional-subsystem module names, bound as literals at every call site",
    ("hermes_cli/console_engine.py", "_CliSurface.build"):
        "command module names from the literal command tables in this module",
    ("hermes_cli/console_engine.py", "_simple_command.handler"):
        "command module names from the literal command tables in this module",
    ("hermes_cli/credential_lifecycle.py", "_for_each_provider"):
        "Hermes module paths, bound as literals at every call site",
    ("hermes_cli/cron.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("hermes_cli/doctor_tools.py", "_check_tool_availability"):
        "the project root of the deployment's own checkout — inside the image by construction",
    ("hermes_cli/gateway.py", "run_gateway"):
        "the project root of the deployment's own checkout — inside the image by construction",
    ("hermes_cli/gateway_migrate.py", "platform_serves_profile_prefix"):
        "the __module__ of an already-imported adapter factory — the module is already in the process",
    ("hermes_cli/local_runtime/__init__.py", "__getattr__"):
        "submodules of this image package, selected by the literal _SOURCE table",
    ("hermes_cli/main.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("hermes_cli/main.py", "__getattr__"):
        "module names from the literal _FROZEN_ATTR_SOURCES table in this module",
    ("hermes_cli/main.py", "_cleanup_oneshot_runtime"):
        "module names from the literal _ONESHOT_CLEANUPS table in this module",
    ("hermes_cli/main.py", "_forward_command._cmd"):
        "command module names from the literal command registry in this module",
    ("hermes_cli/plugin_host.py", "_import_ref"):
        "base-class refs from the literal HOST_OBJECT_BASES table and the literal refs at every category-loader call site",
    ("hermes_cli/plugins.py", "PluginManager._re_register_config_hooks_after_force"):
        "core module names from the literal table in this function",
    ("hermes_cli/plugins_registrars.py", "_resolve"):
        "refs from the literal registrar table in this module",
    ("hermes_cli/plugins_registrars.py", "make_scoped_provider_registrar.register"):
        "the registry module from the literal registrar table in this module",
    ("hermes_cli/plugins_loader.py", "PluginLoaderMixin._register_deferred_platform_tools"):
        "a submodule of the plugin module the guarded directory loader just imported — its search path is the plugin directory the predicate already judged",
    ("hermes_cli/service_manager.py", "_HostServiceManager._backend_module"):
        "backend names enumerated by this module's own literal table",
    ("hermes_cli/setup_summary.py", "_first_available_plugin_provider"):
        "registry names, bound as literals at the call sites",
    ("hermes_cli/source_completion.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("hermes_cli/tools_config_providers.py", "_plugin_registry"):
        "registry module names, bound as literals at every call site",
    ("hermes_cli/update_completion.py", "main"):
        "this checkout's own root, verified against the request — inside the image by construction",
    ("hermes_cli/update_host_obligation.py", "_marker_mutex"):
        "the file of this module's own update_lock module — already loaded in this process",
    ("hermes_cli/web_deps.py", "_server"):
        "the literal WEB_SERVER name in this module",
    ("hermes_cli/web_routers/status.py", "_advisory_pressure"):
        "core module names from the literal table in this function",
    ("hermes_cli/web_server.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("hermes_cli/windows_ssh_runtime.py", "_win32"):
        "the literal pywin32 module table in this function",
    ("mini_swe_runner.py", "create_environment"):
        "environment types validated against a literal set, resolving inside the image's own tools package",
    ("plugins/kanban/dashboard/plugin_api.py", "_run_aux._run"):
        "a submodule of the image's own hermes_cli package (the literal prefix in the f-string)",
    ("plugins/memory/__init__.py", "import_memory_provider_module"):
        "the literal _NATIVE_WARM_IMPORTS table in this module",
    ("plugins/memory/__init__.py", "import_provider_module"):
        "a submodule of the provider package the guarded loader resolved — its search path is the provider directory the predicate already judged",
    ("plugins/platforms/buzz/adapter.py", "_load_sibling"):
        "a sibling module inside this bundled plugin's own tree — inside the image by construction",
    ("plugins/platforms/buzz/adapter.py", "_load_sibling_file"):
        "a sibling file inside this bundled plugin's own tree — inside the image by construction",
    ("plugins/platforms/discord/adapter.py", "<module>"):
        "this plugin's own checkout root, derived from __file__ — inside the image by construction",
    ("plugins/platforms/feishu/adapter.py", "_load_lark_oapi"):
        "the literal _LARK_SDK_IMPORTS table in this module",
    ("plugins/platforms/google_chat/adapter.py", "_load_google_modules"):
        "the literal _GOOGLE_IMPORTS table in this module",
    ("plugins/platforms/raft/adapter.py", "<module>"):
        "this plugin's own checkout root, derived from __file__ — inside the image by construction",
    ("plugins/platforms/slack/adapter.py", "<module>"):
        "this plugin's own checkout root, derived from __file__ — inside the image by construction",
    ("plugins/platforms/teams/adapter.py", "check_teams_requirements._import"):
        "the literal _SDK_IMPORTS table in this module",
    ("plugins/platforms/telegram/adapter.py", "<module>"):
        "this plugin's own checkout root, derived from __file__ — inside the image by construction",
    ("plugins/platforms/telegram/adapter.py", "check_telegram_requirements"):
        "the literal SDK module table in this function",
    ("plugins/platforms/whatsapp/adapter.py", "<module>"):
        "this plugin's own checkout root, derived from __file__ — inside the image by construction",
    ("plugins/platforms/whatsapp_cloud/adapter.py", "_optional_module"):
        "resolver module names, bound as literals at every call site",
    ("pm/_venv_entry.py", "main"):
        "the pm-managed venv's console entry point: it launches the requested command in that venv's own interpreter (its addsitedir processes that venv's .pth files there), never inside a Hermes process",
    ("pm/launch.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("pm/registry.py", "_builtins_loaded"):
        "module names from the literal _BUILTIN_MODULES table in this module",
    ("pm/worker.py", "main"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("pm/worker_operations.py", "Operation.resolve"):
        "module names from the literal OPERATIONS table in this module",
    ("providers/__init__.py", "bundled_profiles"):
        "the bundled provider plugins directory inside the image's own tree",
    ("providers/__init__.py", "_run_discovery_steps"):
        "modules enumerated over the image's own providers package path (pkgutil)",
    ("tools/approval.py", "clear_session"):
        "core module names from the literal table in this function",
    ("tools/bot_mode_dm.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction (the __main__ delivery lane)",
    ("tools/browser_use_cli.py", "_lazy_call"):
        "core module names, bound as literals at every call site",
    ("tools/connectors/__init__.py", "__getattr__"):
        "submodules of this image package, selected by the literal _LAZY_EXPORTS table",
    ("tools/cronjob_tools.py", "<module>"):
        "this module's own project root, derived from __file__ — inside the image by construction",
    ("tools/mcp_tool.py", "_import_sdk_names"):
        "SDK module names, bound as literals at every call site",
    ("tools/mcp_tool.py", "sdk_httpx"):
        "the literal httpx fallback table in this function",
    ("tools/mcp_tool_errors.py", "_optional_types"):
        "optional SDK module names, bound as literals at every call site",
    ("tools/tts_tool.py", "_sdk_importer._import"):
        "SDK module names, bound as literals at every _sdk_importer call site",
    ("tui_gateway/agent_factory.py", "_make_agent"):
        "core module names from the literal table in this function",
    ("tui_gateway/methods_session.py", "_billing_view._"):
        "module and builder names from the literal billing view table in this module",
    ("tui_gateway/methods_tools.py", "_tools_mod"):
        "core module names, bound as literals at every call site",
}

_SCAN_SKIP_DIRS = {"tests", "tests-js", "node_modules", ".git", "__pycache__", "website", "docs"}

_CALLED_PRIMITIVES = {
    ("runpy", "run_path"): "runpy.run_path",
    ("runpy", "run_module"): "runpy.run_module",
    ("importlib.util", "spec_from_file_location"): "importlib.util.spec_from_file_location",
    ("importlib.machinery", "SourceFileLoader"): "importlib.machinery.SourceFileLoader",
    ("importlib.machinery", "SourcelessFileLoader"): "importlib.machinery.SourcelessFileLoader",
    ("importlib.machinery", "ExtensionFileLoader"): "importlib.machinery.ExtensionFileLoader",
    ("importlib", "import_module"): "importlib.import_module",
    ("zipimport", "zipimporter"): "zipimport.zipimporter",
    ("site", "addsitedir"): "site.addsitedir",
    ("site", "addpackage"): "site.addpackage",
}

_PREDICATE = "hosted_code_refusal"


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _reads_file_contents(expr: ast.expr | None) -> bool:
    """Whether *expr* reads file contents (an ``open``/``read*`` call feeds it)."""
    if expr is None:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            tail = dotted.rsplit(".", 1)[-1]
            if tail in ("read", "read_text", "read_bytes", "readlines") or dotted == "open":
                return True
    return False


class _Hit:
    def __init__(self, relpath: str, qualname: str, lineno: int, primitive: str, guarded: bool):
        self.relpath, self.qualname, self.lineno, self.primitive = relpath, qualname, lineno, primitive
        self.guarded = guarded

    def where(self) -> str:
        return f"{self.relpath}:{self.lineno} ({self.qualname}) {self.primitive}"


def _resolved_dotted(func: ast.expr, aliases: dict[str, str]) -> str:
    """The full dotted name *func* refers to, with ``from X import Y`` aliases resolved."""
    dotted = _dotted(func)
    head = dotted.split(".")[0]
    if head in aliases:
        rest = dotted.split(".", 1)[1] if "." in dotted else ""
        return aliases[head] + ("." + rest if rest else "")
    return dotted


def _primitive_of(node: ast.Call, dotted: str) -> str | None:
    """The dynamic-code-loading primitive this call is, or ``None`` when it is not one."""
    primitive = next((label for (module, attr), label in _CALLED_PRIMITIVES.items()
                      if dotted == f"{module}.{attr}"), None)
    if primitive is None and dotted.startswith("sys.path."):
        tail = dotted.rsplit(".", 1)[-1]
        if tail in ("insert", "append"):
            primitive = f"sys.path.{tail}"
    if primitive == "importlib.import_module":
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant):
            primitive = None  # a literal module name is a static import, not a loader
    if primitive is None:
        base = dotted.rsplit(".", 1)[-1]
        if base in ("exec", "compile") and node.args and _reads_file_contents(node.args[0]):
            primitive = f"{base}(file contents)"
    return primitive


def _enclosing_function(functions: list[tuple[int, int, str]], lineno: int) -> tuple[int, int, str] | None:
    """The innermost function whose span holds *lineno*, or ``None`` at module level."""
    enclosing = None
    for start, end, qualname in functions:
        if start <= lineno <= end and (enclosing is None or start > enclosing[0]):
            enclosing = (start, end, qualname)
    return enclosing


def _scan_file(path: Path, relpath: str) -> list[_Hit]:
    """The census hits in one repository file: import aliases, the function spans, the
    predicate's call sites, and every call that resolves to a loader primitive."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    functions: list[tuple[int, int, str]] = []

    def collect(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                functions.append((child.lineno, child.end_lineno or child.lineno, qualname))
                collect(child, qualname)
            elif isinstance(child, ast.ClassDef):
                collect(child, f"{prefix}.{child.name}" if prefix else child.name)
            else:
                collect(child, prefix)

    collect(tree, "")
    predicate_linenos = {
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _resolved_dotted(node.func, aliases).endswith(_PREDICATE)}
    hits: list[_Hit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        primitive = _primitive_of(node, _resolved_dotted(node.func, aliases))
        if primitive is None:
            continue
        enclosing = _enclosing_function(functions, node.lineno)
        qualname = enclosing[2] if enclosing else "<module>"
        guarded = enclosing is not None and any(
            enclosing[0] <= guard_lineno <= enclosing[1] and guard_lineno < node.lineno
            for guard_lineno in predicate_linenos)
        hits.append(_Hit(relpath, qualname, node.lineno, primitive, guarded))
    return hits


def _scan_repo() -> list[_Hit]:
    """Every dynamic-code-loading primitive outside the exempt trees, with its enclosing
    function and whether that function calls the predicate on the path FIRST — the guard
    must precede the load, so the refusal happens before any code can run."""
    hits: list[_Hit] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _SCAN_SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = Path(root) / fn
            relpath = str(path.relative_to(REPO_ROOT))
            if any(relpath.startswith(prefix) for prefix in EXEMPT_TREES):
                continue
            hits.extend(_scan_file(path, relpath))
    return hits


def test_the_loader_census_finds_no_unguarded_loader():
    """The primary guarantee: every dynamic code load in the repository either calls
    :func:`hermes_cli.hosted_code_fence.hosted_code_refusal` on its path first (so a
    hosted process refuses out-of-image code before it loads), or carries a named
    exemption above explaining why its input is image-only. A new loader without
    either fails here — add the predicate call, not an exemption, unless the input
    really is image-only by construction."""
    unguarded = [hit for hit in _scan_repo()
                 if not hit.guarded and (hit.relpath, hit.qualname) not in EXEMPT_LOADERS]
    assert not unguarded, (
        "Unguarded dynamic code loaders (guard each with hosted_code_refusal on the "
        "path before it loads, or add a named exemption entry): \n  "
        + "\n  ".join(hit.where() for hit in unguarded))


def test_the_predicate_call_detector_is_alive():
    """The census must be seeing the predicate, or the guard check would silently pass
    everything as unguarded-by-omission the other way: at least one hit is guarded,
    and a guarded hit's guard really precedes its load (the scan encodes the ordering)."""
    hits = _scan_repo()
    assert any(hit.guarded for hit in hits), (
        "No loader hit detected as guarded — the predicate-call detector is broken, "
        "so the census would not notice a removed guard")


def test_the_census_scans_something():
    """The scanner itself is alive: the repository really holds hundreds of loader
    primitives, so an accidental empty scan cannot pass as a green census."""
    hits = _scan_repo()
    assert len(hits) > 80
    assert any("spec_from_file_location" in hit.primitive for hit in hits)
    assert any(hit.primitive.startswith("sys.path.") for hit in hits)
    assert any(hit.primitive == "runpy.run_path" for hit in hits)
    # Site processing is a code load (.pth files execute import-time code): the census
    # must keep seeing every addsitedir/addpackage call site, or a new unguarded one
    # would pass as a mere path step.
    assert any(hit.primitive == "site.addsitedir" for hit in hits)
    assert any(hit.primitive == "site.addpackage" for hit in hits)


def test_every_exemption_entry_matches_a_real_hit():
    """No dead exemptions: every named entry above still covers at least one real hit,
    so the table cannot rot into covering nothing (or into hiding a removed loader)."""
    hits = _scan_repo()
    seen = {(hit.relpath, hit.qualname) for hit in hits}
    stale = sorted(f"{relpath}:{qualname}" for (relpath, qualname) in EXEMPT_LOADERS
                   if (relpath, qualname) not in seen)
    assert not stale, f"Exemption entries that no longer match any loader hit: {stale}"


def test_the_fence_module_is_never_exempt():
    """The predicate itself lives in the fence module and must keep importing only the
    stdlib (the arm path runs before application imports): no census exemption may ever
    cover hermes_cli/hosted_code_fence.py itself."""
    assert not any(relpath == "hermes_cli/hosted_code_fence.py"
                   for (relpath, _qualname) in EXEMPT_LOADERS)
    assert not any(prefix == "hermes_cli/hosted_code_fence.py" for prefix in EXEMPT_TREES)
