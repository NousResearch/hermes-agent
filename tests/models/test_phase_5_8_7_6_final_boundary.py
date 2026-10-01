"""Final 5.8.7 scoped static/lazy-import and single-owner boundary gates."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCOPED_ROOTS = (
    "plugins/model-providers",
    "plugins/image_gen/deepinfra",
    "plugins/video_gen/deepinfra",
    "plugins/platforms/discord",
    "plugins/platforms/telegram",
    "plugins/platforms/feishu",
)
EXTRA_PLATFORMS = (
    "plugins/platforms/slack/adapter.py",
    "plugins/platforms/matrix/adapter.py",
)
OLD_AUTHORITY = (
    "hermes_cli.models",
    "hermes_cli.model_switch",
    "hermes_cli.model_selection_guards",
    "hermes_cli.provider_groups",
    "hermes_cli.providers",
    "hermes_cli.runtime_provider",
)
PHASE_6_AUTH = {
    "plugins/model-providers/actual/__init__.py": {"resolve_api_key_provider_credentials"},
    "plugins/model-providers/copilot-acp/__init__.py": {"resolve_external_process_provider_credentials"},
    "plugins/model-providers/opencode-zen/__init__.py": {"resolve_api_key_provider_credentials"},
}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                yield node.module, name.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                yield item.name, ""
        elif isinstance(node, ast.Call) and node.args:
            # Static analysis also inspects literal lazy imports, not only imports
            # appearing in an AST Import/ImportFrom node.
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            func = node.func
            if (
                isinstance(func, ast.Name) and func.id == "__import__"
                or isinstance(func, ast.Attribute) and func.attr == "import_module"
            ):
                yield first.value, "*dynamic*"


def _scoped_files() -> list[Path]:
    return [
        *(
            path for base in SCOPED_ROOTS
            for path in (ROOT / base).rglob("*.py")
        ),
        *(ROOT / item for item in EXTRA_PLATFORMS),
    ]


def _is_old(module: str) -> bool:
    return any(module == old or module.startswith(old + ".") for old in OLD_AUTHORITY)


def test_all_scoped_provider_and_platform_plugins_reject_old_semantic_imports():
    files = _scoped_files()
    assert len(files) >= 60  # Explicit scope must not silently shrink.
    bad = []
    for path in files:
        for module, symbol in _imports(path):
            if _is_old(module):
                bad.append((path.relative_to(ROOT).as_posix(), module, symbol))
    assert bad == []


def test_phase_6_auth_exceptions_are_exact_not_wildcard():
    actual = {}
    for p in (ROOT / "plugins/model-providers").rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        for module, symbol in _imports(p):
            if module == "hermes_cli.auth":
                actual.setdefault(rel, set()).add(symbol)
    assert actual == PHASE_6_AUTH


def test_final_catalogue_presentation_and_selection_owners_are_unique():
    for gone in (
        "hermes_cli/model_switch_providers.py",
        "hermes_cli/provider_groups.py",
        "hermes_cli/model_selection_guards.py",
    ):
        assert not (ROOT / gone).exists(), gone
    expected = {
        "hermes_cli/inventory.py": ("application_provider_discovery", "list_authenticated_providers"),
        "hermes_cli/main_provider_setup.py": ("application_provider_groups", "group_providers"),
        "hermes_cli/auth_model_picker.py": ("application_model_selection_guards", "selection_warnings"),
        "hermes_cli/main.py": ("application_model_selection_guards", "selection_warnings"),
        "hermes_cli/cli_model_switch_mixin.py": (
            "application_model_selection_guards", "combined_selection_warning",
        ),
    }
    for rel, target in expected.items():
        assert target in set(_imports(ROOT / rel)), rel
    for rel in (
        "application_provider_discovery.py", "application_model_selection_guards.py",
        "application_provider_groups.py",
    ):
        assert (ROOT / rel).exists()


def test_discovery_is_observation_only_and_setup_is_the_explicit_writer():
    source = (ROOT / "application_provider_discovery.py").read_text(encoding="utf-8")
    assert "save_config" not in source
    assert "_save_discovered_models_to_config" not in source
    setup = set(_imports(ROOT / "hermes_cli/model_setup_flows_custom.py"))
    assert ("application_discovered_catalog_persistence",
            "_save_discovered_models_to_config") in setup
    web = (ROOT / "hermes_cli/web_routers/models.py").read_text(encoding="utf-8")
    assert "_save_discovered_models_to_config" not in web


def test_external_lazy_plugin_entry_uses_one_application_discovery_owner():
    tree = ast.parse((ROOT / "hermes_cli/model_switch.py").read_text(encoding="utf-8"))
    lazy = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_PLUGIN_COMPAT_LAZY"
                for target in node.targets)
    )
    mapping = ast.literal_eval(lazy.value)
    assert mapping["list_picker_providers"] == (
        "application_provider_discovery", "list_picker_providers",
    )


def test_router_key_lookup_cannot_borrow_process_key_after_profile_scope_miss(monkeypatch):
    from agent import secret_scope
    from application_provider_secret_inputs import scoped_key_env
    from providers import get_provider_profile

    # First declared credential absent in this profile. It must NOT fall back
    # to a different user's process-level key before checking this profile's alias.
    monkeypatch.setenv("RAMP_ROUTER_API_KEY", "other-profile-secret")
    monkeypatch.setattr(secret_scope, "current_secret_scope", lambda: object())
    monkeypatch.setattr(secret_scope, "get_secret",
                        lambda name, fallback="": "profile-key" if name == "ROUTER_API_KEY" else "")
    assert scoped_key_env("RAMP_ROUTER_API_KEY") == ""
    profile = get_provider_profile("router")
    assert profile is not None
    module = sys.modules[profile.__class__.__module__]
    assert module._resolve_api_key() == "profile-key"
    # A profile with neither declared key must remain unauthenticated even
    # when the process retains a token belonging to a different profile.
    monkeypatch.setattr(secret_scope, "get_secret", lambda _name, _fallback="": "")
    assert module._resolve_api_key() == ""


def test_feishu_dynamic_env_loader_is_app_config_not_provider_semantics():
    imports = set(_imports(ROOT / "plugins/platforms/feishu/feishu_comment_rules.py"))
    assert ("hermes_cli.env_loader", "*dynamic*") in imports
    assert not any(_is_old(module) for module, _ in imports)
