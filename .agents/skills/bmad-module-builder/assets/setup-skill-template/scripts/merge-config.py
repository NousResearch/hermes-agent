#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Merge module configuration into shared _bmad/config.yaml and config.user.yaml.

Reads a module.yaml definition and a JSON answers file, then writes or updates
the shared config.yaml (core values at root + module section) and config.user.yaml
(user_name, communication_language, plus any module variable with user_setting: true).
Uses an anti-zombie pattern for the module section in config.yaml.

Legacy migration: when --legacy-dir is provided, reads old per-module config files
from {legacy-dir}/{module-code}/config.yaml and {legacy-dir}/core/config.yaml.
Matching values serve as fallback defaults (answers override them). After a
successful merge, the legacy config.yaml files are deleted. Only the current
module and core directories are touched — other module directories are left alone.

Exit codes: 0=success, 1=validation error, 2=runtime error
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required (PEP 723 dependency)", file=sys.stderr)
    sys.exit(2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge module config into shared _bmad/config.yaml with anti-zombie pattern."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the target _bmad/config.yaml file",
    )
    parser.add_argument(
        "--module-yaml",
        required=True,
        help="Path to the module.yaml definition file",
    )
    parser.add_argument(
        "--answers",
        required=True,
        help="Path to JSON file with collected answers",
    )
    parser.add_argument(
        "--user-config-path",
        required=True,
        help="Path to the target _bmad/config.user.yaml file",
    )
    parser.add_argument(
        "--legacy-dir",
        help="Path to _bmad/ directory to check for legacy per-module config files. "
        "Matching values are used as fallback defaults, then legacy files are deleted.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress to stderr",
    )
    return parser.parse_args()


def load_yaml_file(path: str) -> dict:
    """Load a YAML file, returning empty dict if file doesn't exist."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content if content else {}


def load_json_file(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Keys that live at config root (shared across all modules)
_CORE_KEYS = frozenset(
    {"user_name", "communication_language", "document_output_language", "output_folder"}
)


def load_legacy_values(
    legacy_dir: str, module_code: str, module_yaml: dict, verbose: bool = False
) -> tuple[dict, dict, list]:
    """Read legacy per-module config files and return core/module value dicts.

    Reads {legacy_dir}/core/config.yaml and {legacy_dir}/{module_code}/config.yaml.
    Only returns values whose keys match the current schema (core keys or module.yaml
    variable definitions). Other modules' directories are not touched.

    Returns:
        (legacy_core, legacy_module, files_found) where files_found lists paths read.
    """
    legacy_core: dict = {}
    legacy_module: dict = {}
    files_found: list = []

    # Read core legacy config
    core_path = Path(legacy_dir) / "core" / "config.yaml"
    if core_path.exists():
        core_data = load_yaml_file(str(core_path))
        files_found.append(str(core_path))
        for k, v in core_data.items():
            if k in _CORE_KEYS:
                legacy_core[k] = v
        if verbose:
            print(f"Legacy core config: {list(legacy_core.keys())}", file=sys.stderr)

    # Read module legacy config
    mod_path = Path(legacy_dir) / module_code / "config.yaml"
    if mod_path.exists():
        mod_data = load_yaml_file(str(mod_path))
        files_found.append(str(mod_path))
        for k, v in mod_data.items():
            if k in _CORE_KEYS:
                # Core keys duplicated in module config — only use if not already set
                if k not in legacy_core:
                    legacy_core[k] = v
            elif k in module_yaml and isinstance(module_yaml[k], dict):
                # Module-specific key that matches a current variable definition
                legacy_module[k] = v
        if verbose:
            print(
                f"Legacy module config: {list(legacy_module.keys())}", file=sys.stderr
            )

    return legacy_core, legacy_module, files_found


def apply_legacy_defaults(answers: dict, legacy_core: dict, legacy_module: dict) -> dict:
    """Apply legacy values as fallback defaults under the answers.

    Legacy values fill in any key not already present in answers.
    Explicit answers always win.
    """
    merged = dict(answers)

    if legacy_core:
        core = merged.get("core", {})
        filled_core = dict(legacy_core)  # legacy as base
        filled_core.update(core)  # answers override
        merged["core"] = filled_core

    if legacy_module:
        mod = merged.get("module", {})
        filled_mod = dict(legacy_module)  # legacy as base
        filled_mod.update(mod)  # answers override
        merged["module"] = filled_mod

    return merged


def cleanup_legacy_configs(
    legacy_dir: str, module_code: str, verbose: bool = False
) -> list:
    """Delete legacy config.yaml files for this module and core only.

    Returns list of deleted file paths.
    """
    deleted = []
    for subdir in (module_code, "core"):
        legacy_path = Path(legacy_dir) / subdir / "config.yaml"
        if legacy_path.exists():
            if verbose:
                print(f"Deleting legacy config: {legacy_path}", file=sys.stderr)
            legacy_path.unlink()
            deleted.append(str(legacy_path))
    return deleted


def extract_module_metadata(module_yaml: dict) -> dict:
    """Extract non-variable metadata fields from module.yaml."""
    meta = {}
    for k in ("name", "description"):
        if k in module_yaml:
            meta[k] = module_yaml[k]
    meta["version"] = module_yaml.get("module_version")  # null if absent
    if "default_selected" in module_yaml:
        meta["default_selected"] = module_yaml["default_selected"]
    return meta


def apply_result_templates(
    module_yaml: dict, module_answers: dict, verbose: bool = False
) -> dict:
    """Apply result templates from module.yaml to transform raw answer values.

    For each answer, if the corresponding variable definition in module.yaml has
    a 'result' field, replaces {value} in that template with the answer. Skips
    the template if the answer already contains '{project-root}' to prevent
    double-prefixing.
    """
    transformed = {}
    for key, value in module_answers.items():
        var_def = module_yaml.get(key)
        if (
            isinstance(var_def, dict)
            and "result" in var_def
            and "{project-root}" not in str(value)
        ):
            template = var_def["result"]
            transformed[key] = template.replace("{value}", str(value))
            if verbose:
                print(
                    f"Applied result template for '{key}': {value} → {transformed[key]}",
                    file=sys.stderr,
                )
        else:
            transformed[key] = value
    return transformed


def merge_config(
    existing_config: dict,
    module_yaml: dict,
    answers: dict,
    verbose: bool = False,
) -> dict:
    """Merge answers into config, applying anti-zombie pattern.

    Args:
        existing_config: Current config.yaml contents (may be empty)
        module_yaml: The module definition
        answers: JSON with 'core' and/or 'module' keys
        verbose: Print progress to stderr

    Returns:
        Updated config dict ready to write
    """
    config = dict(existing_config)
    module_code = module_yaml.get("code")

    if not module_code:
        print("Error: module.yaml must have a 'code' field", file=sys.stderr)
        sys.exit(1)

    # Migrate legacy core: section to root
    if "core" in config and isinstance(config["core"], dict):
        if verbose:
            print("Migrating legacy 'core' section to root", file=sys.stderr)
        config.update(config.pop("core"))

    # Strip user-only keys from config — they belong exclusively in config.user.yaml
    for key in _CORE_USER_KEYS:
        if key in config:
            if verbose:
                print(f"Removing user-only key '{key}' from config (belongs in config.user.yaml)", file=sys.stderr)
            del config[key]

    # Write core values at root (global properties, not nested under "core")
    # Exclude user-only keys — those belong exclusively in config.user.yaml
    core_answers = answers.get("core")
    if core_answers:
        shared_core = {k: v for k, v in core_answers.items() if k not in _CORE_USER_KEYS}
        if shared_core:
            if verbose:
                print(f"Writing core config at root: {list(shared_core.keys())}", file=sys.stderr)
            config.update(shared_core)

    # Anti-zombie: remove existing module section
    if module_code in config:
        if verbose:
            print(
                f"Removing existing '{module_code}' section (anti-zombie)",
                file=sys.stderr,
            )
        del config[module_code]

    # Build module section: metadata + variable values
    module_section = extract_module_metadata(module_yaml)
    module_answers = apply_result_templates(
        module_yaml, answers.get("module", {}), verbose
    )
    module_section.update(module_answers)

    if verbose:
        print(
            f"Writing '{module_code}' section with keys: {list(module_section.keys())}",
            file=sys.stderr,
        )

    config[module_code] = module_section

    return config


# Core keys that are always written to config.user.yaml
_CORE_USER_KEYS = ("user_name", "communication_language")


def extract_user_settings(module_yaml: dict, answers: dict) -> dict:
    """Collect settings that belong in config.user.yaml.

    Includes user_name and communication_language from core answers, plus any
    module variable whose definition contains user_setting: true.
    """
    user_settings = {}

    core_answers = answers.get("core", {})
    for key in _CORE_USER_KEYS:
        if key in core_answers:
            user_settings[key] = core_answers[key]

    module_answers = answers.get("module", {})
    for var_name, var_def in module_yaml.items():
        if isinstance(var_def, dict) and var_def.get("user_setting") is True:
            if var_name in module_answers:
                user_settings[var_name] = module_answers[var_name]

    return user_settings


def write_config(config: dict, config_path: str, verbose: bool = False) -> None:
    """Write config dict to YAML file, creating parent dirs as needed."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing config to {path}", file=sys.stderr)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def reject_unresolved_paths(named_paths: list[tuple[str, str]]) -> None:
    """Exit with a clear error if any path argument still contains the literal
    ``{project-root}`` token. That token is meaningful only inside config
    values; filesystem path arguments must be resolved by the caller. Failing
    loudly here prevents silently creating a junk ``{project-root}/`` directory.
    """
    for name, value in named_paths:
        if value and "{project-root}" in value:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"Unresolved '{{project-root}}' token in {name} path: {value!r}. "
                            "Resolve '{project-root}' to the actual project root before running "
                            "this script — it is a filesystem path, not a config value."
                        ),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    args = parse_args()

    reject_unresolved_paths(
        [
            ("--config-path", args.config_path),
            ("--user-config-path", args.user_config_path),
            ("--legacy-dir", args.legacy_dir),
        ]
    )

    # Load inputs
    module_yaml = load_yaml_file(args.module_yaml)
    if not module_yaml:
        print(f"Error: Could not load module.yaml from {args.module_yaml}", file=sys.stderr)
        sys.exit(1)

    answers = load_json_file(args.answers)
    existing_config = load_yaml_file(args.config_path)

    if args.verbose:
        exists = Path(args.config_path).exists()
        print(f"Config file exists: {exists}", file=sys.stderr)
        if exists:
            print(f"Existing sections: {list(existing_config.keys())}", file=sys.stderr)

    # Legacy migration: read old per-module configs as fallback defaults
    legacy_files_found = []
    if args.legacy_dir:
        module_code = module_yaml.get("code", "")
        legacy_core, legacy_module, legacy_files_found = load_legacy_values(
            args.legacy_dir, module_code, module_yaml, args.verbose
        )
        if legacy_core or legacy_module:
            answers = apply_legacy_defaults(answers, legacy_core, legacy_module)
            if args.verbose:
                print("Applied legacy values as fallback defaults", file=sys.stderr)

    # Merge and write config.yaml
    updated_config = merge_config(existing_config, module_yaml, answers, args.verbose)
    write_config(updated_config, args.config_path, args.verbose)

    # Merge and write config.user.yaml
    user_settings = extract_user_settings(module_yaml, answers)
    existing_user_config = load_yaml_file(args.user_config_path)
    updated_user_config = dict(existing_user_config)
    updated_user_config.update(user_settings)
    if user_settings:
        write_config(updated_user_config, args.user_config_path, args.verbose)

    # Legacy cleanup: delete old per-module config files
    legacy_deleted = []
    if args.legacy_dir:
        legacy_deleted = cleanup_legacy_configs(
            args.legacy_dir, module_yaml["code"], args.verbose
        )

    # Output result summary as JSON
    module_code = module_yaml["code"]
    result = {
        "status": "success",
        "config_path": str(Path(args.config_path).resolve()),
        "user_config_path": str(Path(args.user_config_path).resolve()),
        "module_code": module_code,
        "core_updated": bool(answers.get("core")),
        "module_keys": list(updated_config.get(module_code, {}).keys()),
        "user_keys": list(user_settings.keys()),
        "legacy_configs_found": legacy_files_found,
        "legacy_configs_deleted": legacy_deleted,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
