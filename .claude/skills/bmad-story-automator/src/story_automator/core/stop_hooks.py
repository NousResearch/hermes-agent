from __future__ import annotations

import json
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_atomic


CODEX_HOOK_STATUS_MESSAGE = "Checking story automator state"
STOP_HOOK_EVENT = "Stop"


class HookConfigError(Exception):
    def __init__(self, code: str, path: Path, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.path = path
        self.message = message or code


@dataclass(frozen=True)
class HookInstallResult:
    changed: bool
    reason: str
    path: Path
    written: bool = False


@dataclass(frozen=True)
class HookFileUpdate:
    result: HookInstallResult
    data: str | None = None


def ensure_stop_hook(
    *,
    provider: str,
    project_root: Path,
    settings_path: Path | None,
    command: str,
    timeout: int,
) -> dict[str, Any]:
    if provider == "codex":
        return ensure_codex_stop_hook(project_root=project_root, command=command, timeout=timeout)
    if not settings_path:
        raise HookConfigError("missing_settings", project_root / ".claude" / "settings.json")
    return ensure_claude_stop_hook(settings_path=settings_path, command=command, timeout=timeout)


def ensure_claude_stop_hook(*, settings_path: Path, command: str, timeout: int) -> dict[str, Any]:
    result = _ensure_json_stop_hook(settings_path, command=command, timeout=timeout)
    return {
        "changed": result.changed,
        "reason": result.reason,
        "provider": "claude",
        "path": str(result.path),
        "message": _claude_hook_message(result.changed),
    }


def ensure_codex_stop_hook(*, project_root: Path, command: str, timeout: int) -> dict[str, Any]:
    codex_dir = project_root / ".codex"
    hooks_path = codex_dir / "hooks.json"
    config_path = codex_dir / "config.toml"

    config_update = _prepare_codex_hooks_feature(config_path)
    hook_update = _prepare_json_stop_hook(
        hooks_path,
        command=command,
        timeout=timeout,
        status_message=CODEX_HOOK_STATUS_MESSAGE,
    )
    _write_prepared_update(config_update)
    _write_prepared_update(hook_update)

    hook_result = hook_update.result
    config_result = config_update.result
    changed = hook_result.changed or config_result.changed
    trusted = _codex_project_is_trusted(config_path, project_root)

    if changed:
        reason = "codex_hook_configured"
    elif not trusted:
        reason = "pending_trust"
    elif hook_result.written:
        reason = hook_result.reason
    else:
        reason = "already_configured"

    return {
        "changed": changed,
        "reason": reason,
        "provider": "codex",
        "path": str(hooks_path),
        "hooksPath": str(hooks_path),
        "configPath": str(config_path),
        "hooksChanged": hook_result.changed,
        "configChanged": config_result.changed,
        "hooksReason": hook_result.reason,
        "configReason": config_result.reason,
        "trusted": trusted,
        "verificationState": _codex_verification_state(changed, trusted),
        "message": _codex_hook_message(changed, trusted),
    }


def _codex_verification_state(changed: bool, trusted: bool) -> str:
    if changed:
        return "configured"
    if trusted:
        return "verified"
    return "pending_trust"


def _codex_hook_message(changed: bool, trusted: bool) -> str:
    if changed:
        suffix = (
            "Restart Codex from this trusted project session for the hook to load."
            if trusted
            else "Trust this project in Codex, then restart Codex so the hook can load."
        )
        return "Codex Stop hook configured in .codex/hooks.json and hooks enabled in .codex/config.toml. " + suffix
    if trusted:
        return "Codex Stop hook verified."
    return "Codex Stop hook is configured on disk, but this project is not yet trusted in Codex."


def _claude_hook_message(changed: bool) -> str:
    if changed:
        return "Claude Stop hook configured in .claude/settings.json. Restart Claude for the hook to load."
    return "Claude Stop hook verified."


def _ensure_json_stop_hook(
    path: Path,
    *,
    command: str,
    timeout: int,
    status_message: str | None = None,
) -> HookInstallResult:
    update = _prepare_json_stop_hook(path, command=command, timeout=timeout, status_message=status_message)
    _write_prepared_update(update)
    return update.result


def _prepare_json_stop_hook(
    path: Path,
    *,
    command: str,
    timeout: int,
    status_message: str | None = None,
) -> HookFileUpdate:
    payload = _stop_hook_payload(command=command, timeout=timeout, status_message=status_message)
    if not path.exists():
        return HookFileUpdate(
            result=HookInstallResult(changed=True, reason="created", path=path, written=True),
            data=json.dumps(payload, indent=2) + "\n",
        )

    root = _read_json_object(path)
    hooks = _object_child(root, "hooks", path)
    stop_hooks = _list_child(hooks, STOP_HOOK_EVENT, path)

    exists = False
    needs_update = False
    for entry in stop_hooks:
        if not isinstance(entry, dict):
            continue
        handlers = entry.get("hooks", [])
        if not isinstance(handlers, list):
            continue
        for hook in handlers:
            if not isinstance(hook, dict):
                continue
            existing = hook.get("command")
            if not _is_story_automator_stop_hook(existing, command):
                continue
            exists = True
            if hook.get("type") != "command":
                hook["type"] = "command"
                needs_update = True
            if existing != command:
                hook["command"] = command
                needs_update = True
            if hook.get("timeout") != timeout:
                hook["timeout"] = timeout
                needs_update = True
            if status_message is not None and hook.get("statusMessage") != status_message:
                hook["statusMessage"] = status_message
                needs_update = True

    if exists and not needs_update:
        return HookFileUpdate(result=HookInstallResult(changed=False, reason="already_configured", path=path))
    if exists:
        return HookFileUpdate(
            result=HookInstallResult(changed=True, reason="hook_normalized", path=path, written=True),
            data=json.dumps(root, indent=2) + "\n",
        )

    stop_hooks.append(payload["hooks"][STOP_HOOK_EVENT][0])
    return HookFileUpdate(
        result=HookInstallResult(changed=True, reason="added", path=path, written=True),
        data=json.dumps(root, indent=2) + "\n",
    )


def _write_prepared_update(update: HookFileUpdate) -> None:
    if not update.result.written:
        return
    if update.data is None:
        raise HookConfigError("missing_prepared_data", update.result.path)
    ensure_dir(update.result.path.parent)
    write_atomic(update.result.path, update.data)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HookConfigError("invalid_json", path, str(exc)) from exc
    if not isinstance(root, dict):
        raise HookConfigError("invalid_json_object", path)
    return root


def _object_child(root: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = root.setdefault(key, {})
    if not isinstance(value, dict):
        raise HookConfigError(f"invalid_{key}_object", path)
    return value


def _list_child(root: dict[str, Any], key: str, path: Path) -> list[Any]:
    value = root.setdefault(key, [])
    if not isinstance(value, list):
        raise HookConfigError(f"invalid_{key.lower()}_hooks", path)
    return value


def _stop_hook_payload(*, command: str, timeout: int, status_message: str | None = None) -> dict[str, Any]:
    hook: dict[str, Any] = {
        "type": "command",
        "command": command,
        "timeout": timeout,
    }
    if status_message is not None:
        hook["statusMessage"] = status_message
    return {"hooks": {STOP_HOOK_EVENT: [{"hooks": [hook]}]}}


def _is_story_automator_stop_hook(existing: Any, command: str) -> bool:
    if existing == command:
        return True
    return _is_story_automator_stop_hook_command(str(existing))


def _is_story_automator_stop_hook_command(value: str) -> bool:
    try:
        parts = shlex.split(value)
    except ValueError:
        return False
    if not parts:
        return False
    parts = _strip_env_prefix(parts)
    if not parts:
        return False
    command_name = Path(parts[0]).name
    if command_name == "story-automator":
        return len(parts) > 1 and parts[1] == "stop-hook"
    return (
        _is_python_command(command_name)
        and len(parts) > 3
        and parts[1] == "-m"
        and parts[2] == "story_automator"
        and parts[3] == "stop-hook"
    )


def _strip_env_prefix(parts: list[str]) -> list[str]:
    if Path(parts[0]).name != "env":
        return parts
    idx = 1
    while idx < len(parts):
        part = parts[idx]
        if part in {"-i", "-0"}:
            idx += 1
            continue
        if part in {"-u", "--unset"} and idx + 1 < len(parts):
            idx += 2
            continue
        if part.startswith("--unset="):
            idx += 1
            continue
        if "=" in part and not part.startswith("-"):
            idx += 1
            continue
        break
    return parts[idx:]


def _is_python_command(command_name: str) -> bool:
    return bool(re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", command_name))


def _ensure_codex_hooks_feature(path: Path) -> HookInstallResult:
    update = _prepare_codex_hooks_feature(path)
    _write_prepared_update(update)
    return update.result


def _prepare_codex_hooks_feature(path: Path) -> HookFileUpdate:
    if not path.exists():
        return HookFileUpdate(
            result=HookInstallResult(changed=True, reason="created", path=path, written=True),
            data="[features]\nhooks = true\n",
        )

    text = path.read_text(encoding="utf-8")
    parsed = _parse_toml(text, path)
    features = parsed.get("features", {})
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise HookConfigError("invalid_features_table", path)
    # Already correct only when the current key is enabled AND no deprecated
    # `codex_hooks` lingers (a leftover legacy key still triggers Codex warnings).
    if features.get("hooks") is True and "codex_hooks" not in features:
        return HookFileUpdate(result=HookInstallResult(changed=False, reason="already_enabled", path=path))

    updated = _set_features_hooks(text)
    _parse_toml(updated, path)
    return HookFileUpdate(
        result=HookInstallResult(changed=True, reason="hooks_enabled", path=path, written=True),
        data=updated,
    )


def _parse_toml(text: str, path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise HookConfigError("invalid_toml", path, str(exc)) from exc


def _codex_project_is_trusted(config_path: Path, project_root: Path) -> bool:
    resolved_root = project_root.resolve()
    return (
        _config_trusts_project(config_path, resolved_root, ignore_errors=False)
        or _config_trusts_project(_codex_global_config_path(), resolved_root, ignore_errors=True)
    )


def _codex_global_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _config_trusts_project(config_path: Path, resolved_root: Path, *, ignore_errors: bool) -> bool:
    if not config_path.exists():
        return False
    try:
        parsed = _parse_toml(config_path.read_text(encoding="utf-8"), config_path)
    except (HookConfigError, OSError):
        if ignore_errors:
            return False
        raise
    projects = parsed.get("projects", {})
    if not isinstance(projects, dict):
        return False
    for raw_key, raw_config in projects.items():
        if not isinstance(raw_key, str) or not isinstance(raw_config, dict):
            continue
        try:
            if Path(raw_key).expanduser().resolve() != resolved_root:
                continue
        except OSError:
            if raw_key != str(resolved_root):
                continue
        trust_level = str(raw_config.get("trust_level") or "").strip().lower()
        if trust_level == "trusted":
            return True
    return False


def _set_features_hooks(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return "[features]\nhooks = true\n"

    table_start = _find_table_start(lines, "features")
    if table_start is None:
        return _set_top_level_features_hooks(text, lines)

    table_end = _find_table_end(lines, table_start)
    hooks_pattern = re.compile(r"^\s*hooks\s*=.*$")
    legacy_pattern = re.compile(r"^\s*codex_hooks\s*=.*$")
    hooks_index: int | None = None
    legacy_index: int | None = None
    for index in range(table_start + 1, table_end):
        if hooks_index is None and hooks_pattern.match(lines[index]):
            hooks_index = index
        elif legacy_index is None and legacy_pattern.match(lines[index]):
            legacy_index = index

    return _apply_feature_hooks(lines, table_start, hooks_index, legacy_index, key="hooks")


def _set_top_level_features_hooks(text: str, lines: list[str]) -> str:
    root_end = _find_first_table_start(lines)
    hooks_dotted = re.compile(r"^\s*features\.hooks\s*=.*$")
    legacy_dotted = re.compile(r"^\s*features\.codex_hooks\s*=.*$")
    hooks_index: int | None = None
    legacy_index: int | None = None
    for index, line in enumerate(lines[:root_end]):
        if hooks_index is None and hooks_dotted.match(line):
            hooks_index = index
        elif legacy_index is None and legacy_dotted.match(line):
            legacy_index = index
    if hooks_index is not None or legacy_index is not None:
        return _apply_feature_hooks(lines, root_end, hooks_index, legacy_index, key="features.hooks")

    inline_features = re.compile(r"^(\s*)features\s*=\s*\{(.*)\}\s*(#.*)?$")
    for index, line in enumerate(lines[:root_end]):
        match = inline_features.match(line)
        if match:
            lines[index] = _set_inline_features_table_line(match)
            return "\n".join(lines) + "\n"

    last_dotted_index: int | None = None
    dotted_features = re.compile(r"^\s*features\.[A-Za-z0-9_-]+(?:\.[^=]+)?\s*=.*$")
    for index, line in enumerate(lines[:root_end]):
        if dotted_features.match(line):
            last_dotted_index = index
    if last_dotted_index is not None:
        lines.insert(last_dotted_index + 1, "features.hooks = true")
        return "\n".join(lines) + "\n"

    separator = "\n" if text.endswith("\n") else "\n\n"
    return f"{text}{separator}[features]\nhooks = true\n"


def _apply_feature_hooks(
    lines: list[str],
    insert_after: int,
    hooks_index: int | None,
    legacy_index: int | None,
    *,
    key: str,
) -> str:
    if hooks_index is not None:
        # Enable the existing hooks key and drop any leftover deprecated key so
        # the rewritten config never carries both (which would also be a dup key).
        lines[hooks_index] = f"{_leading_whitespace(lines[hooks_index])}{key} = true"
        if legacy_index is not None:
            del lines[legacy_index]
    elif legacy_index is not None:
        # Migrate the deprecated key in place, preserving its indentation.
        lines[legacy_index] = f"{_leading_whitespace(lines[legacy_index])}{key} = true"
    else:
        lines.insert(insert_after + 1, f"{key} = true")
    return "\n".join(lines) + "\n"


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_first_table_start(lines: list[str]) -> int:
    table_pattern = re.compile(r"^\s*\[.+\]\s*(?:#.*)?$")
    for index, line in enumerate(lines):
        if table_pattern.match(line):
            return index
    return len(lines)


def _set_inline_features_table_line(match: re.Match[str]) -> str:
    indent, inner, comment = match.group(1), match.group(2), match.group(3) or ""
    items = [item.strip() for item in _split_inline_table_items(inner) if item.strip()]
    hooks_present = any(re.match(r"^hooks\s*=", item) for item in items)
    updated_items: list[str] = []
    added = False
    for item in items:
        if re.match(r"^hooks\s*=", item):
            updated_items.append("hooks = true")
            added = True
        elif re.match(r"^codex_hooks\s*=", item):
            # Drop the deprecated key if a current one exists; otherwise migrate it.
            if hooks_present:
                continue
            updated_items.append("hooks = true")
            added = True
        else:
            updated_items.append(item)
    if not added:
        updated_items.append("hooks = true")
    return f"{indent}features = {{ {', '.join(updated_items)} }}{comment}"


def _split_inline_table_items(inner: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(inner):
        if quote:
            if quote == '"' and char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                quote = ""
            escaped = False
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "{[(":
            depth += 1
            continue
        if char in "}])" and depth > 0:
            depth -= 1
            continue
        if char == "," and depth == 0:
            items.append(inner[start:index])
            start = index + 1
    items.append(inner[start:])
    return items


def _find_table_start(lines: list[str], table_name: str) -> int | None:
    pattern = re.compile(rf"^\s*\[{re.escape(table_name)}\]\s*(?:#.*)?$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            return index
    return None


def _find_table_end(lines: list[str], table_start: int) -> int:
    table_pattern = re.compile(r"^\s*\[.+\]\s*(?:#.*)?$")
    for index in range(table_start + 1, len(lines)):
        if table_pattern.match(lines[index]):
            return index
    return len(lines)
