"""Guards for writes to the active Hermes config and secret files.

The file tools normally block credential files outright.  Active Hermes
``config.yaml`` / ``.env`` need a narrower path: allow intentional safe edits
while refusing the dangerous class of agent mistakes that wipe real secrets by
replacing them with placeholders, nulls, empties, or by deleting credential
fields entirely.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class LiveConfigGuardResult:
    guarded: bool = False
    redacted_diff: str = ""
    error: Optional[str] = None


_CREDENTIAL_NAME_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|private[_-]?key|"
    r"access[_-]?key|auth[_-]?token|oauth)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"^(?:"
    r"|none|null|nil|undefined|false|0"
    r"|changeme|change[_-]?me|replace[_-]?me|todo|tbd"
    r"|placeholder|dummy|example|sample"
    r"|your(?:[_-]?[a-z0-9]+)*[_-]?(?:key|token|secret|password)(?:[_-]?here)?"
    r"|<[^>]*(?:key|token|secret|password)[^>]*>"
    r"|\[[^\]]*(?:redacted|key|token|secret|password)[^\]]*\]"
    r"|\*+|x{3,}|\.\.\."
    r")$",
    re.IGNORECASE,
)
_ENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=)(?P<value>.*)$"
)
_YAML_SECRET_LINE_RE = re.compile(
    r"^(?P<prefix>\s*[^#\n:]*?(?:api[_-]?key|token|secret|password|credential|"
    r"private[_-]?key|access[_-]?key|auth[_-]?token|oauth)[^:\n]*:\s*)(?P<value>.*)$",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


def classify_live_config_path(path: str | os.PathLike[str]) -> Optional[str]:
    """Return ``config``/``env`` for active Hermes config files, else None."""
    try:
        resolved = Path(path).expanduser().resolve()
        from hermes_constants import get_default_hermes_root, get_hermes_home

        hermes_home = get_hermes_home().expanduser().resolve()
        hermes_root = get_default_hermes_root().expanduser().resolve()
    except Exception:
        return None

    if resolved == hermes_home / "config.yaml":
        return "config"
    # Active profile .env plus top-level root .env are both credential stores.
    if resolved in {hermes_home / ".env", hermes_root / ".env"}:
        return "env"
    return None


def is_live_config_path(path: str | os.PathLike[str]) -> bool:
    return classify_live_config_path(path) is not None


def validate_live_config_write(
    path: str | os.PathLike[str],
    *,
    old_content: Optional[str],
    new_content: str,
) -> LiveConfigGuardResult:
    """Validate a prospective write to a live Hermes config/secrets file.

    Existing real credential values may not be removed or replaced with null,
    empty, redacted, or placeholder-looking values.  The returned diff is always
    redacted when the path is guarded.
    """
    kind = classify_live_config_path(path)
    if kind is None:
        return LiveConfigGuardResult(guarded=False)

    before = old_content or ""
    redacted_diff = _redacted_unified_diff(before, new_content, str(path), kind)
    try:
        old_credentials = _collect_credentials(kind, before, label="existing", path=str(path))
        new_credentials = _collect_credentials(kind, new_content, label="new", path=str(path))
    except ValueError as exc:
        return LiveConfigGuardResult(
            guarded=True,
            redacted_diff=redacted_diff,
            error=f"Refusing to write live Hermes config/secrets file: {exc}",
        )

    violations: list[str] = []
    for name, old_value in sorted(old_credentials.items()):
        if not _is_real_secret(old_value):
            continue
        if name not in new_credentials:
            violations.append(f"would remove existing credential field {name}")
            continue
        new_value = new_credentials[name]
        if _is_placeholder_value(new_value):
            violations.append(f"would replace existing credential field {name} with a placeholder/null/empty value")

    if violations:
        return LiveConfigGuardResult(
            guarded=True,
            redacted_diff=redacted_diff,
            error=(
                "Refusing to write live Hermes config/secrets file: "
                + "; ".join(violations)
                + ". Preserve the existing secret value or update it with a real credential."
            ),
        )

    return LiveConfigGuardResult(guarded=True, redacted_diff=redacted_diff)


def _collect_credentials(kind: str, content: str, *, label: str, path: str) -> dict[str, Any]:
    if kind == "env":
        return _collect_env_credentials(content)
    return _collect_yaml_credentials(content, label=label, path=path)


def _collect_env_credentials(content: str) -> dict[str, str]:
    credentials: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_ASSIGNMENT_RE.match(raw_line)
        if not match:
            continue
        key = match.group("key")
        if not _is_credential_name(key):
            continue
        credentials[key] = _strip_env_value(match.group("value"))
    return credentials


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _collect_yaml_credentials(content: str, *, label: str, path: str) -> dict[str, Any]:
    if not content.strip():
        return {}
    try:
        import yaml
    except ImportError:
        # PyYAML is optional elsewhere, but Hermes config safety needs a
        # conservative fallback: regex line extraction still catches the common
        # ``api_key: value`` shape without pretending nested YAML was parsed.
        return _collect_yaml_credentials_by_line(content)

    try:
        data = yaml.safe_load(content)
    except Exception as exc:
        raise ValueError(
            f"Cannot parse {label} live Hermes config {path!r} for credential safety: {exc}"
        ) from exc

    credentials: dict[str, Any] = {}
    _walk_yaml(data, (), credentials)
    return credentials


def _collect_yaml_credentials_by_line(content: str) -> dict[str, str]:
    credentials: dict[str, str] = {}
    for index, line in enumerate(content.splitlines(), start=1):
        match = _YAML_SECRET_LINE_RE.match(line)
        if match:
            credentials[f"line:{index}"] = match.group("value").strip().strip('"\'')
    return credentials


def _walk_yaml(value: Any, path: tuple[str, ...], out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            if _is_credential_name(str(key)):
                out[".".join(child_path)] = child
            _walk_yaml(child, child_path, out)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_yaml(child, path + (str(index),), out)


def _is_credential_name(name: str) -> bool:
    return bool(_CREDENTIAL_NAME_RE.search(name))


def _is_real_secret(value: Any) -> bool:
    return not _is_placeholder_value(value)


def _is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    text = str(value).strip().strip('"\'')
    if not text:
        return True
    if _PLACEHOLDER_RE.match(text):
        return True
    return False


def _redacted_unified_diff(old: str, new: str, path: str, kind: str) -> str:
    old_redacted = _redact_content(kind, old).splitlines(keepends=True)
    new_redacted = _redact_content(kind, new).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_redacted,
            new_redacted,
            fromfile=f"{path} (before, secrets redacted)",
            tofile=f"{path} (after, secrets redacted)",
        )
    )


def _redact_content(kind: str, content: str) -> str:
    if kind == "env":
        redacted = [_redact_env_line(line) for line in content.splitlines()]
    else:
        redacted = _redact_yaml_lines(content.splitlines())
    return "\n".join(redacted) + ("\n" if content.endswith("\n") else "")


def _redact_env_line(line: str) -> str:
    match = _ENV_ASSIGNMENT_RE.match(line)
    if match and _is_credential_name(match.group("key")):
        return f"{match.group('prefix')}{_REDACTED}"
    return line


def _redact_yaml_lines(lines: list[str]) -> list[str]:
    """Redact YAML credential values, including literal/folded blocks."""
    redacted: list[str] = []
    block_secret_indent: Optional[int] = None
    block_placeholder_emitted = False

    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if block_secret_indent is not None:
            if stripped and indent > block_secret_indent:
                if not block_placeholder_emitted:
                    redacted.append(" " * (block_secret_indent + 2) + _REDACTED)
                    block_placeholder_emitted = True
                continue
            block_secret_indent = None
            block_placeholder_emitted = False

        match = _YAML_SECRET_LINE_RE.match(line)
        if match:
            value = match.group("value").strip()
            redacted.append(f"{match.group('prefix')}{_REDACTED}")
            if value in {"|", ">", "|-", "|+", ">-", ">+"}:
                block_secret_indent = indent
                block_placeholder_emitted = False
            continue

        redacted.append(line)

    return redacted
