"""Architecture guards for the Phase 5.5 model metadata ownership boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = ROOT / "models" / "metadata"


def _python_files(root: Path):
    return tuple(sorted(root.rglob("*.py")))


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _definitions(path: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(("function", node.name))
        elif isinstance(node, ast.ClassDef):
            found.add(("class", node.name))
    return found


def test_metadata_domain_has_no_upward_runtime_imports():
    forbidden = ("agent", "hermes_cli", "gateway", "tui_gateway")
    prefixes = tuple(f"{name}." for name in forbidden)
    offenders: list[str] = []
    for path in _python_files(METADATA_ROOT):
        for imported in sorted(_imports(path)):
            if imported in forbidden or imported.startswith(prefixes):
                offenders.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not offenders, "model metadata imports upward:\n" + "\n".join(offenders)


def test_cli_reasoning_metadata_owner_is_deleted():
    assert not (ROOT / "hermes_cli" / "models_reasoning_caps.py").exists()



def test_canonical_metadata_has_no_legacy_capability_projection():
    path = METADATA_ROOT / "types.py"
    assert ("class", "ModelCapabilities") not in _definitions(path)


def test_agent_models_dev_no_longer_owns_capability_projection():
    path = ROOT / "agent" / "models_dev.py"
    defs = _definitions(path)
    assert ("class", "ModelCapabilities") not in defs
    assert ("function", "get_model_capabilities") not in defs


def test_agent_model_metadata_no_longer_owns_context_resolution():
    path = ROOT / "agent" / "model_metadata.py"
    assert ("function", "get_model_context_length") not in _definitions(path)


def test_image_routing_delegates_vision_capability_lookup():
    path = ROOT / "agent" / "image_routing.py"
    defs = _definitions(path)
    assert ("function", "_lookup_supports_vision") in defs
    assert ("function", "_probe_models_dev") not in defs
    assert ("function", "_probe_ollama") not in defs
    assert ("function", "_probe_managed_runtime") not in defs
    assert "models.metadata" in _imports(path)


def test_runtime_consumers_do_not_import_deleted_capability_owners():
    forbidden = {
        "hermes_cli.models_reasoning_caps",
    }
    offenders: list[str] = []
    for root_name in ("agent", "hermes_cli", "gateway", "tui_gateway", "plugins"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in _python_files(root):
            for imported in _imports(path):
                if imported in forbidden:
                    offenders.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not offenders, "obsolete capability imports survive:\n" + "\n".join(offenders)


def test_phase_5_8_4_5_runtime_consumers_use_canonical_semantic_owners():
    auxiliary = ROOT / "agent" / "auxiliary_client.py"
    imports = _imports(auxiliary)
    assert "models.metadata" in imports
    assert "models.selection" in imports
    assert "providers" in imports

    source = auxiliary.read_text(encoding="utf-8")
    for obsolete in (
        "provider_rejects_vision_input",
        "provider_vision_default",
        "select_provider_vision_model",
        "_main_model_supports_vision",
        "from hermes_cli.providers import get_provider",
        "from hermes_cli.models import get_nous_recommended_aux_model",
    ):
        assert obsolete not in source

    adapter = ROOT / "agent" / "auxiliary_model_resolution.py"
    adapter_defs = _definitions(adapter)
    for obsolete in (
        "provider_rejects_vision_input",
        "provider_vision_default",
        "is_declared_vision_default",
        "select_provider_vision_model",
    ):
        assert ("function", obsolete) not in adapter_defs

    nous_profile = ROOT / "plugins" / "model-providers" / "nous" / "__init__.py"
    assert "get_nous_recommended_aux_model" not in nous_profile.read_text(encoding="utf-8")
    assert (ROOT / "providers" / "nous_recommendations.py").exists()
