from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_COMMAND_TIMEOUT = 600
COMMAND_TIMEOUT_EXIT = 124


@dataclass
class CommandResult:
    output: str
    exit_code: int
    error: Exception | None = None

    def __iter__(self):
        yield self.output
        yield self.exit_code

    def __getitem__(self, index: int):
        if index == 0:
            return self.output
        if index == 1:
            return self.exit_code
        raise IndexError(index)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_z() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(payload: Any) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def file_exists(path: str | Path) -> bool:
    return Path(path).is_file()


def dir_exists(path: str | Path) -> bool:
    return Path(path).is_dir()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_atomic(path: str | Path, data: str | bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        mode = "wb" if isinstance(data, bytes) else "w"
        with os.fdopen(fd, mode) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def atomic_write(path: str | Path, data: str | bytes) -> None:
    write_atomic(path, data)


def run_cmd(
    *args: str,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = DEFAULT_COMMAND_TIMEOUT,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(completed.stdout, completed.returncode)
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        return CommandResult(output, COMMAND_TIMEOUT_EXIT, exc)
    except FileNotFoundError as exc:
        return CommandResult("", 127, exc)


def get_pwd() -> str:
    return os.getcwd()


def get_project_root() -> str:
    return os.environ.get("PROJECT_ROOT", get_pwd())


def get_project_slug(project_root: str | None = None) -> str:
    root = Path(project_root or get_project_root())
    value = re.sub(r"[^a-z0-9]", "", root.name.lower())[:8]
    return value or "project"


def md5_hex8(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]


def get_project_hash(project_root: str | None = None) -> str:
    return md5_hex8(str(Path(project_root or get_project_root()).resolve()))


def project_slug(project_root: str | None = None) -> str:
    return get_project_slug(project_root)


def project_hash(project_root: str | None = None) -> str:
    return get_project_hash(project_root)


def trim_lines(text: str) -> list[str]:
    return [line.rstrip("\r") for line in text.splitlines()]


def filter_input_box(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_box = False
    for line in lines:
        if re.match(r"^\s*[╭┌]", line):
            in_box = True
            continue
        if re.match(r"^\s*[╰└]", line):
            in_box = False
            continue
        if in_box and re.match(r"^\s*[│|]", line):
            continue
        output.append(line)
    return "\n".join(output)


def is_help_flag(value: str) -> bool:
    return value in {"--help", "-h"}


def help_flag(value: str) -> bool:
    return is_help_flag(value)


def strip_inline_yaml_comment(raw: str) -> str:
    text = raw.strip()
    in_quote = ""
    escaped = False
    for idx, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if in_quote == char:
                in_quote = ""
            elif not in_quote:
                in_quote = char
            continue
        if char == "#" and not in_quote and (idx == 0 or text[idx - 1].isspace()):
            return text[:idx].rstrip()
    return text


def unquote_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_string_list_literal(raw: str) -> list[str] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return None


def default_string(value: str | None, default: str) -> str:
    return value if value else default


def count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def extract_json_line(text: str) -> str:
    for line in trim_lines(text):
        for match in re.findall(r"\{.*\}", line):
            try:
                json.loads(match)
            except json.JSONDecodeError:
                continue
            return match
    return ""


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def iso_now() -> str:
    return now_utc_z()


def print_json(payload: Any) -> None:
    write_json(payload)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None
