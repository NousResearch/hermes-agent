from __future__ import annotations

VALID_MODELS = {
    "claude-sonnet-4.6",
    "gpt-5.4",
    "deepseek",
    "ollama",
    "flash_or_o4_mini",
}

VALID_TASK_TYPES = {"chat", "coding", "writing", "research", "batch", "trivial"}
VALID_MODES = {"draft", "execute", "review"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_PRIVACY = {"normal", "sensitive", "local_only"}
VALID_QUOTA = {"normal", "low", "critical"}
VALID_SPEED = {"normal", "fast"}
VALID_POLICY_FIELDS = {"task_type", "mode", "priority", "privacy", "quota", "speed", "has_code", "has_logs"}
VALID_BOOLEAN_POLICY_FIELDS = {"has_code", "has_logs"}
VALID_POLICY_VALUES = {
    "task_type": VALID_TASK_TYPES,
    "mode": VALID_MODES,
    "priority": VALID_PRIORITIES,
    "privacy": VALID_PRIVACY,
    "quota": VALID_QUOTA,
    "speed": VALID_SPEED,
}


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _warn(warnings: list[str], msg: str) -> None:
    warnings.append(msg)


def _normalize_when_signature(when: dict) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(when.items(), key=lambda item: item[0]))


def _is_subset_when(left: dict, right: dict) -> bool:
    return all(right.get(key) == value for key, value in left.items())


def _validate_policy_value(errors: list[str], path: str, field: str, value: object) -> None:
    if field in VALID_BOOLEAN_POLICY_FIELDS:
        if not isinstance(value, bool):
            _err(errors, f"{path}.{field} חייב להיות boolean")
        return

    valid_values = VALID_POLICY_VALUES.get(field)
    if valid_values is None:
        return

    if value not in valid_values:
        allowed = ", ".join(sorted(valid_values))
        _err(errors, f"{path}.{field} כולל ערך לא חוקי: {value} (מותר: {allowed})")


def validate_router_config(config: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, dict):
        return {"valid": False, "errors": ["קובץ config ראשי לא תקין"], "warnings": []}

    for key in ("router", "base_by_task", "fallbacks"):
        if key not in config:
            _err(errors, f"שדה חובה חסר: {key}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    router = config["router"]
    if not isinstance(router, dict):
        _err(errors, "router חייב להיות mapping")
        router = {}

    version = router.get("version")
    if version is not None and not isinstance(version, str):
        _err(errors, "router.version חייב להיות string")

    default_model = router.get("default_model")
    if default_model not in VALID_MODELS:
        _err(errors, f"router.default_model לא חוקי: {default_model}")

    base_by_task = config.get("base_by_task", {})
    if not isinstance(base_by_task, dict):
        _err(errors, "base_by_task חייב להיות mapping")
        base_by_task = {}
    else:
        for task_type, model in base_by_task.items():
            if task_type not in VALID_TASK_TYPES:
                _err(errors, f"base_by_task כולל task_type לא חוקי: {task_type}")
            if model not in VALID_MODELS:
                _err(errors, f"base_by_task.{task_type} כולל מודל לא חוקי: {model}")

        missing_tasks = sorted(VALID_TASK_TYPES - set(base_by_task.keys()))
        if missing_tasks:
            _warn(warnings, f"base_by_task לא מכסה את כל סוגי המשימות: {', '.join(missing_tasks)}")

    fallbacks = config.get("fallbacks", {})
    if not isinstance(fallbacks, dict):
        _err(errors, "fallbacks חייב להיות mapping")
        fallbacks = {}
    else:
        for model, values in fallbacks.items():
            if model not in VALID_MODELS:
                _err(errors, f"fallbacks key לא חוקי: {model}")
            if not isinstance(values, list):
                _err(errors, f"fallbacks.{model} חייב להיות list")
                continue
            seen_items = set()
            for item in values:
                if item not in VALID_MODELS:
                    _err(errors, f"fallbacks.{model} כולל מודל fallback לא חוקי: {item}")
                if item == model:
                    _warn(warnings, f"fallbacks.{model} כולל את עצמו")
                if item in seen_items:
                    _warn(warnings, f"fallbacks.{model} כולל כפילות של {item}")
                seen_items.add(item)

        referenced_models = {default_model} | {model for model in base_by_task.values() if model in VALID_MODELS}
        for model in sorted(referenced_models):
            if model and model not in fallbacks:
                _warn(warnings, f"fallbacks חסר עבור מודל בשימוש: {model}")

    mode_overrides = config.get("mode_overrides", {})
    if mode_overrides and not isinstance(mode_overrides, dict):
        _err(errors, "mode_overrides חייב להיות mapping")
    elif isinstance(mode_overrides, dict):
        for mode, mapping in mode_overrides.items():
            if mode not in VALID_MODES:
                _err(errors, f"mode_overrides כולל mode לא חוקי: {mode}")
            if not isinstance(mapping, dict):
                _err(errors, f"mode_overrides.{mode} חייב להיות mapping")
                continue
            for task_type, model in mapping.items():
                if task_type not in VALID_TASK_TYPES:
                    _err(errors, f"mode_overrides.{mode} כולל task_type לא חוקי: {task_type}")
                if model not in VALID_MODELS:
                    _err(errors, f"mode_overrides.{mode}.{task_type} כולל מודל לא חוקי: {model}")

    reviewers = config.get("reviewers", {})
    if reviewers and not isinstance(reviewers, dict):
        _err(errors, "reviewers חייב להיות mapping")
    elif isinstance(reviewers, dict):
        for priority, mapping in reviewers.items():
            if priority not in VALID_PRIORITIES:
                _err(errors, f"reviewers כולל priority לא חוקי: {priority}")
            if not isinstance(mapping, dict):
                _err(errors, f"reviewers.{priority} חייב להיות mapping")
                continue
            for task_type, model in mapping.items():
                if task_type not in VALID_TASK_TYPES:
                    _err(errors, f"reviewers.{priority} כולל task_type לא חוקי: {task_type}")
                if model not in VALID_MODELS:
                    _err(errors, f"reviewers.{priority}.{task_type} כולל מודל לא חוקי: {model}")

    policy_overrides = config.get("policy_overrides", [])
    valid_policy_entries: list[dict[str, object]] = []
    if policy_overrides and not isinstance(policy_overrides, list):
        _err(errors, "policy_overrides חייב להיות list")
    elif isinstance(policy_overrides, list):
        seen_names: dict[str, int] = {}
        seen_signatures: dict[tuple[tuple[str, object], ...], int] = {}

        for i, override in enumerate(policy_overrides):
            path = f"policy_overrides[{i}]"
            if not isinstance(override, dict):
                _err(errors, f"{path} חייב להיות mapping")
                continue

            name = override.get("name")
            if not isinstance(name, str) or not name:
                _err(errors, f"{path}.name חסר או לא תקין")
            else:
                previous_index = seen_names.get(name)
                if previous_index is not None:
                    _warn(warnings, f"{path}.name כפול ל-{name} (כבר קיים ב-policy_overrides[{previous_index}])")
                else:
                    seen_names[name] = i

            reason = override.get("reason")
            if reason is not None and not isinstance(reason, str):
                _err(errors, f"{path}.reason חייב להיות string אם הוא קיים")

            when = override.get("when")
            if not isinstance(when, dict) or not when:
                _err(errors, f"{path}.when חייב להיות mapping לא ריק")
                when = None
            else:
                for field, value in when.items():
                    if field not in VALID_POLICY_FIELDS:
                        _err(errors, f"{path}.when כולל שדה לא נתמך: {field}")
                        continue
                    _validate_policy_value(errors, f"{path}.when", field, value)

                if len(when) < 2:
                    _warn(warnings, f"{path} רחב מאוד — מומלץ לפחות שני תנאים ב-when")
                if "task_type" not in when and "quota" not in when:
                    _warn(warnings, f"{path} לא כולל task_type או quota — עלול להיות כלל רחב מדי")

            force = override.get("force")
            if force not in VALID_MODELS:
                _err(errors, f"{path}.force לא חוקי: {force}")

            if isinstance(when, dict) and force in VALID_MODELS:
                if when.get("privacy") == "local_only" and force != "ollama":
                    _warn(warnings, f"{path} מנסה לכפות {force} על privacy=local_only — בפועל hard safety יחזיר ollama")

                signature = _normalize_when_signature(when)
                previous_index = seen_signatures.get(signature)
                if previous_index is not None:
                    _warn(warnings, f"{path}.when זהה ל-policy_overrides[{previous_index}].when")
                else:
                    seen_signatures[signature] = i

                valid_policy_entries.append({
                    "index": i,
                    "path": path,
                    "name": name or f"override-{i}",
                    "when": when,
                    "force": force,
                })

        for entry_index, entry in enumerate(valid_policy_entries):
            for prior in valid_policy_entries[:entry_index]:
                if _is_subset_when(prior["when"], entry["when"]):
                    if prior["force"] == entry["force"]:
                        _warn(
                            warnings,
                            f"{entry['path']} נבלע ע\"י {prior['path']} עם אותו force={entry['force']} — כלל מאוחר מיותר",
                        )
                    else:
                        _warn(
                            warnings,
                            f"{entry['path']} לא יופעל כי {prior['path']} נתפס קודם עבור אותם תנאים או כלל רחב יותר",
                        )
                    break

    return {"valid": not errors, "errors": errors, "warnings": warnings}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Validate router_config.yaml")
    parser.add_argument("config_path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config_path).read_text(encoding="utf-8"))
    result = validate_router_config(config)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"== Router Config Validation ==\nValid: {'yes' if result['valid'] else 'no'}\n")
        print("Errors:")
        if result["errors"]:
            for item in result["errors"]:
                print(f"  - {item}")
        else:
            print("  (none)")
        print("\nWarnings:")
        if result["warnings"]:
            for item in result["warnings"]:
                print(f"  - {item}")
        else:
            print("  (none)")

    raise SystemExit(0 if result["valid"] else 1)



def main():
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, __file__, *sys.argv[1:]]))
