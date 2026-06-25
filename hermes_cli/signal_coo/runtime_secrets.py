"""1Password runtime-secret validation for Torben."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SECRET_NAME_RE = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CLIENT_SECRET|MCP_URL|HTTP_URL|ACCOUNT)$")
OP_REF_RE = re.compile(r"^op://[^/\n]+/[^/\n]+/[^/\n]+(?:/[^/\n]+)?$")


@dataclass
class RuntimeSecretReport:
    valid: bool
    missing_required: list[str] = field(default_factory=list)
    plaintext_secret_keys: list[str] = field(default_factory=list)
    invalid_op_refs: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "missing_required": self.missing_required,
            "plaintext_secret_keys": self.plaintext_secret_keys,
            "invalid_op_refs": self.invalid_op_refs,
            "keys": self.keys,
        }


def is_secret_key(key: str) -> bool:
    return bool(SECRET_NAME_RE.search(key.upper()))


def is_op_ref(value: str) -> bool:
    return bool(OP_REF_RE.match(value.strip()))


def parse_env_template(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def validate_runtime_env_template(
    path: str | Path,
    *,
    required_keys: Iterable[str] = (),
) -> RuntimeSecretReport:
    env_path = Path(path)
    values = parse_env_template(env_path.read_text(encoding="utf-8"))
    required = list(required_keys)
    missing = [key for key in required if key not in values or not values[key].strip()]
    plaintext: list[str] = []
    invalid_refs: list[str] = []

    for key, value in values.items():
        if value.startswith("op://") and not is_op_ref(value):
            invalid_refs.append(key)
        if is_secret_key(key) and value and not value.startswith("op://"):
            plaintext.append(key)

    return RuntimeSecretReport(
        valid=not missing and not plaintext and not invalid_refs,
        missing_required=missing,
        plaintext_secret_keys=plaintext,
        invalid_op_refs=invalid_refs,
        keys=sorted(values),
    )


def build_op_run_command(
    *,
    op_bin: str = "/opt/homebrew/bin/op",
    env_file: str | Path | None = None,
    environment: str | None = None,
    command: Iterable[str],
) -> list[str]:
    cmd = [op_bin, "run"]
    if env_file is not None:
        cmd.extend(["--env-file", str(env_file)])
    if environment:
        cmd.extend(["--environment", environment])
    cmd.append("--")
    cmd.extend(command)
    return cmd


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
