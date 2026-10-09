"""Architecture guard for Phase 5.3 model-identity ownership."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "models"
IDENTITY_OWNER = MODELS_ROOT / "identity.py"
ALIAS_OWNER = MODELS_ROOT / "aliases.py"
LEGACY_NORMALIZER = ROOT / "hermes_cli" / "model_normalize.py"
ACP_CATALOG = ROOT / "acp_adapter" / "model_catalog.py"

SCAN_ROOTS = (
    "acp_adapter",
    "agent",
    "gateway",
    "hermes_cli",
    "models",
    "runtime",
    "tui_gateway",
)

IDENTITY_DEFINITIONS = {
    "parse_model_ref": IDENTITY_OWNER,
    "parse_configured_provider_ref": IDENTITY_OWNER,
    "format_model_ref": IDENTITY_OWNER,
    "normalize_model_id": IDENTITY_OWNER,
    "normalize_model_ref": IDENTITY_OWNER,
    "resolve_model_alias": ALIAS_OWNER,
}

OBSOLETE_IDENTITY_DEFINITIONS = {
    "parse_model_input",
    "encode_model_choice",
    "_choice_provider",
    "ModelIdentity",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _top_level_definitions(path: Path):
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name


def test_model_identity_owner_does_not_depend_on_upper_layers():
    forbidden = ("hermes_cli", "agent", "gateway", "tui_gateway", "auth")
    violations = []
    for path in (IDENTITY_OWNER, ALIAS_OWNER):
        for module in _imports(path):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append((str(path.relative_to(ROOT)), module))
    assert violations == []


def test_legacy_cli_model_normalizer_is_deleted_and_unreferenced():
    assert not LEGACY_NORMALIZER.exists()

    violations = []
    for path in _sources():
        for module in _imports(path):
            if module == "hermes_cli.model_normalize" or module.startswith(
                "hermes_cli.model_normalize."
            ):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_generic_model_identity_contract_has_one_owner():
    definitions: dict[str, list[Path]] = {
        name: [] for name in IDENTITY_DEFINITIONS
    }
    obsolete: list[tuple[str, str]] = []

    for path in _sources():
        for name in _top_level_definitions(path):
            if name in definitions:
                definitions[name].append(path)
            if name in OBSOLETE_IDENTITY_DEFINITIONS:
                obsolete.append((str(path.relative_to(ROOT)), name))

    assert obsolete == []
    assert {
        name: [str(path.relative_to(ROOT)) for path in paths]
        for name, paths in definitions.items()
    } == {
        name: [str(owner.relative_to(ROOT))]
        for name, owner in IDENTITY_DEFINITIONS.items()
    }


def test_acp_uses_model_identity_codec_instead_of_parsing_choice_ids():
    tree = _tree(ACP_CATALOG)
    model_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "models"
        for alias in node.names
    }
    assert {"ModelRef", "format_model_ref", "parse_model_ref"} <= model_imports

    local_definitions = set(_top_level_definitions(ACP_CATALOG))
    assert {"_choice_provider", "encode_model_choice"}.isdisjoint(local_definitions)

    colon_splits = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"split", "rsplit", "partition", "rpartition"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == ":"
        ):
            continue
        colon_splits.append(getattr(node, "lineno", 0))
    assert colon_splits == []


def test_model_identity_provider_canonicalization_comes_from_providers():
    tree = _tree(IDENTITY_OWNER)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "providers"
        for alias in node.names
    }
    assert "normalize_provider" in imported
    assert "normalize_provider" not in set(_top_level_definitions(IDENTITY_OWNER))
