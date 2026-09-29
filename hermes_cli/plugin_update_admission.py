"""Data-only compatibility gate for enabled plugins during core updates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


CONTRACTS_PATH = "hermes_cli/plugin_host_contracts.json"


class PluginHostContractError(RuntimeError):
    """The selected core candidate cannot keep an enabled plugin working."""


def _enabled_requirements() -> list[tuple[str, str, int]]:
    from pm.plugin_declarations import read_python_declaration
    from pm.package import InstallError
    from pm.plugins_state import read_home_selection
    from pm.workspace import enabled_plugin_entries

    requirements: list[tuple[str, str, int]] = []
    try:
        entries = enabled_plugin_entries(skip_invalid_secondary=False)
    except (InstallError, OSError, UnicodeError, ValueError) as exc:
        raise PluginHostContractError(
            f"Could not inspect enabled plugin declarations: {exc}") from exc
    for plugins_dir, selection_key, plugin_dir in entries:
        try:
            config = read_home_selection(plugins_dir.parent) or {}
        except (OSError, UnicodeError, ValueError) as exc:
            raise PluginHostContractError(
                f"Could not inspect enabled plugin policy for '{selection_key}': {exc}") from exc
        config_entries = (config.get("plugins") or {}).get("entries") or {}
        entry = config_entries.get(selection_key) if isinstance(config_entries, dict) else None
        if not isinstance(entry, dict) or entry.get("update_admission") != "required":
            continue
        try:
            manifest = read_python_declaration(plugin_dir).manifest
        except (OSError, UnicodeError, ValueError) as exc:
            raise PluginHostContractError(
                f"Could not inspect enabled plugin declaration for '{selection_key}': {exc}") from exc
        declared = manifest.get("requires_host_contracts")
        if not isinstance(declared, dict) or not declared:
            raise PluginHostContractError(
                f"Enabled plugin '{selection_key}' has invalid requires_host_contracts; "
                "expected a non-empty mapping.")
        for contract, version in declared.items():
            if not isinstance(contract, str) or not contract.strip() or type(version) is not int or version < 1:
                raise PluginHostContractError(
                    f"Enabled plugin '{selection_key}' has an invalid host contract requirement.")
            requirements.append((selection_key, contract.strip(), version))
    return requirements


def _verify_registry(raw: str, requirements: list[tuple[str, str, int]]) -> None:
    try:
        document = json.loads(raw)
        if document.get("schema") != 1 or not isinstance(document.get("contracts"), dict):
            raise ValueError("unsupported registry schema")
        contracts = document["contracts"]
        for contract, versions in contracts.items():
            if not isinstance(contract, str) or not contract.strip():
                raise ValueError("contract names must be non-empty strings")
            if (
                not isinstance(versions, list)
                or not versions
                or any(type(version) is not int or version < 1 for version in versions)
                or len(versions) != len(set(versions))
            ):
                raise ValueError(f"contract '{contract}' must publish unique positive integer versions")
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PluginHostContractError(f"Candidate host contract registry is invalid: {exc}") from exc

    missing = []
    for plugin, contract, version in requirements:
        supported = contracts.get(contract, [])
        if not isinstance(supported, list) or version not in supported:
            missing.append(f"plugin '{plugin}' requires {contract} API {version}")
    if missing:
        raise PluginHostContractError(
            "Update blocked before checkout activation: " + "; ".join(missing) + ".")


def verify_git_candidate(git_cmd: list[str], root: Path, target_ref: str) -> None:
    """Refuse a fetched Git candidate before it can move the live checkout."""
    requirements = _enabled_requirements()
    if not requirements:
        return
    result = subprocess.run(
        [*git_cmd, "show", f"{target_ref}:{CONTRACTS_PATH}"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        needed = "; ".join(
            f"plugin '{plugin}' requires {contract} API {version}"
            for plugin, contract, version in requirements
        )
        raise PluginHostContractError(
            f"Update blocked before checkout activation: candidate does not publish {CONTRACTS_PATH}, "
            f"but {needed}.")
    _verify_registry(result.stdout, requirements)


def verify_tree_candidate(root: Path) -> None:
    """Refuse an extracted candidate before the ZIP updater swaps live files."""
    requirements = _enabled_requirements()
    if not requirements:
        return
    path = root / CONTRACTS_PATH
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        needed = "; ".join(
            f"plugin '{plugin}' requires {contract} API {version}"
            for plugin, contract, version in requirements
        )
        raise PluginHostContractError(
            f"Update blocked before checkout activation: candidate does not publish {CONTRACTS_PATH}, "
            f"but {needed}.") from None
    _verify_registry(raw, requirements)
