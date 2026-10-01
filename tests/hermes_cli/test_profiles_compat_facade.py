"""Compatibility contract for the extracted hermes_cli.profiles facade."""

import ast
from pathlib import Path
import subprocess

import pytest

import hermes_cli.profiles as facade
from gateway import profile_serving
from profiles import current, metadata, names, paths, registry


REPO_ROOT = Path(__file__).resolve().parents[2]


PUBLIC_COMPAT_EXPORTS = {
    "PROFILE_ROLES": metadata.PROFILE_ROLES,
    "SETUP_ROLE": metadata.SETUP_ROLE,
    "current_profile_name": current.current_profile_name,
    "drop_profile_role": metadata.drop_profile_role,
    "format_profile_label": metadata.format_profile_label,
    "get_active_profile": current.get_active_profile,
    "get_active_profile_name": current.get_active_profile_name,
    "get_profile_dir": paths.get_profile_dir,
    "list_profile_names": registry.list_profile_names,
    "normalize_profile_name": names.normalize_profile_name,
    "parked_marker_path": profile_serving.parked_marker_path,
    "profile_exists": registry.profile_exists,
    "profile_is_parked": profile_serving.profile_is_parked,
    "profile_is_standalone": profile_serving.profile_is_standalone,
    "profile_matches_home": registry.profile_matches_home,
    "profile_root_for_env_home": paths.profile_root_for_env_home,
    "profiles_to_serve": profile_serving.profiles_to_serve,
    "read_profile_meta": metadata.read_profile_meta,
    "resolve_profile_env": paths.resolve_profile_env,
    "set_active_profile": current.set_active_profile,
    "validate_alias_name": names.validate_alias_name,
    "validate_profile_name": names.validate_profile_name,
    "write_profile_meta": metadata.write_profile_meta,
}

PRIVATE_COMPAT_SEAMS = {
    "_get_default_hermes_home": paths._get_default_hermes_home,
    "_get_profiles_root": paths._get_profiles_root,
}

RETIRED_PRIVATE_NAMES = (
    "_HERMES_SUBCOMMANDS",
    "_PROFILE_ID_RE",
    "_PROFILE_NAME_RULE",
    "_RESERVED_NAMES",
    "_STANDALONE_DEFAULT_WARNING",
    "_STANDALONE_MEMO",
    "_STANDALONE_WARNED",
    "_canon_valid",
    "_clean_previous_names",
    "_existing_profile_dir",
    "_get_active_profile_path",
    "_invalid_profile_name_error",
    "_iter_named_profile_dirs",
    "_load_yaml_dict",
    "_missing_profile_error",
    "_parked_default_warned",
    "_standalone_truthy",
    "_suggest_profile_name",
    "_unknown_profile_error",
)


@pytest.mark.parametrize(
    ("name", "owner"),
    [*PUBLIC_COMPAT_EXPORTS.items(), *PRIVATE_COMPAT_SEAMS.items()],
)
def test_extracted_profile_facade_preserves_compatibility_surface(
    name: str, owner: object
) -> None:
    assert getattr(facade, name) is owner


@pytest.mark.parametrize("name", RETIRED_PRIVATE_NAMES)
def test_extracted_profile_facade_does_not_restore_retired_internals(name: str) -> None:
    assert not hasattr(facade, name)

def _retired_regex_facade_references(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: set[str] = set()
    offenders: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli.profiles":
            if any(alias.name == "_PROFILE_ID_RE" for alias in node.names):
                offenders.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "hermes_cli.profiles":
                    aliases.add(alias.asname or "hermes_cli.profiles")
        elif isinstance(node, ast.ImportFrom) and node.module == "hermes_cli":
            for alias in node.names:
                if alias.name == "profiles":
                    aliases.add(alias.asname or "profiles")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "_PROFILE_ID_RE":
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id in aliases:
            offenders.append(node.lineno)
        elif (
            isinstance(value, ast.Attribute)
            and value.attr == "profiles"
            and isinstance(value.value, ast.Name)
            and value.value.id == "hermes_cli"
        ):
            offenders.append(node.lineno)
    return offenders


def test_no_production_consumer_uses_retired_profile_regex_facade() -> None:
    offenders: list[tuple[str, int]] = []
    tracked = subprocess.run(
        ["git", "grep", "-l", "_PROFILE_ID_RE", "--", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative_text in tracked:
        relative = Path(relative_text)
        if relative.parts and relative.parts[0] == "tests":
            continue
        path = REPO_ROOT / relative
        for line in _retired_regex_facade_references(path):
            offenders.append((str(relative), line))
    assert offenders == []
