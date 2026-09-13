"""Shared primitives for the local Hermes continuity pilot."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import platform
import secrets
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ContinuityError(RuntimeError):
    """A safe, user-readable continuity failure."""


@dataclass(frozen=True)
class GitState:
    root: str
    branch: str
    commit: str
    dirty: bool
    changed_files: tuple[str, ...]
    fingerprint: str
    integration_ref: str | None = None
    integration_sha: str | None = None
    merge_base: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "root": self.root,
            "branch": self.branch,
            "commit": self.commit,
            "dirty": self.dirty,
            "changed_files": list(self.changed_files),
            "fingerprint": self.fingerprint,
        }
        if self.integration_ref is not None:
            value.update({
                "integration_ref": self.integration_ref,
                "integration_sha": self.integration_sha,
                "merge_base": self.merge_base,
            })
        return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuityError(f"expected a JSON object in {path}")
    return value


def read_markdown_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContinuityError(f"cannot read authority {path}: {exc}") from exc
    if len(text) > 256_000:
        raise ContinuityError(f"authority is unexpectedly large: {path}")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ContinuityError(f"missing YAML front matter in {path}")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ContinuityError(f"unterminated YAML front matter in {path}") from exc
    try:
        data = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise ContinuityError(f"invalid YAML front matter in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContinuityError(f"expected YAML mapping in {path}")
    return data, "\n".join(lines[end + 1 :]).lstrip("\n")


def render_markdown_frontmatter(data: dict[str, Any], body: str) -> str:
    header = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    suffix = body.rstrip() + "\n" if body else ""
    return f"---\n{header}\n---\n\n{suffix}"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContinuityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def minimal_child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a credential-minimized environment for authority-bearing children."""
    allowed = {
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "WINDIR",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PATH"] = "/usr/bin:/bin"
    env.setdefault("HOME", env.get("TMPDIR", "/tmp"))
    if extra:
        env.update(extra)
    return env


def verify_pinned_executable(
    config: dict[str, Any], repo_root: Path, tool_name: str, executable: Path
) -> str:
    """Bind an authority-bearing executable to the committed platform digest."""
    lock_path = repo_root / config.get(
        "toolchain_lock", ".continuity/toolchain.lock.json"
    )
    lock = load_json(lock_path)
    tool = (lock.get("tools") or {}).get(tool_name)
    if not isinstance(tool, dict):
        raise ContinuityError(f"toolchain lock has no {tool_name!r} entry")
    system = platform.system().lower()
    machine = (
        platform
        .machine()
        .lower()
        .replace("aarch64", "arm64")
        .replace("x86_64", "amd64")
    )
    digest_key = f"{system}-{machine}-sha256"
    expected = tool.get(digest_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContinuityError(
            f"toolchain lock has no valid {digest_key} for {tool_name}"
        )
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(
            f"cannot resolve pinned {tool_name} executable: {exc}"
        ) from exc
    state_root = Path(config["state_root"]).resolve()
    if not resolved.is_relative_to(state_root):
        raise ContinuityError(
            f"{tool_name} executable escapes pilot state root: {resolved}"
        )
    actual = sha256_file(resolved)
    if actual != expected:
        raise ContinuityError(
            f"{tool_name} executable digest mismatch: expected {expected}, got {actual}"
        )
    return actual


def external_input_digests(config: dict[str, Any], repo_root: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for item in config.get("external_instructions", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContinuityError(
                "external instruction pins require exactly path and sha256"
            )
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        expected = item["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ContinuityError(f"external instruction has invalid SHA-256: {path}")
        actual = sha256_file(path)
        if not hmac.compare_digest(actual, expected):
            raise ContinuityError(
                f"external instruction digest mismatch: expected {expected}, got {actual}"
            )
        inputs[str(path)] = actual
    for item in config.get("evidence_executables", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContinuityError(
                "evidence executable pins require exactly path and sha256"
            )
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        expected = item["sha256"]
        actual = sha256_file(path)
        if not isinstance(expected, str) or not hmac.compare_digest(actual, expected):
            raise ContinuityError(
                f"evidence executable digest mismatch: expected {expected}, got {actual}"
            )
        inputs[str(path)] = actual
    beads_path = Path(config["beads"]["binary"])
    inputs[str(beads_path.resolve())] = verify_pinned_executable(
        config, repo_root, "beads", beads_path
    )
    return dict(sorted(inputs.items()))


def _receipt_key_path(config: dict[str, Any]) -> Path:
    return Path(config["state_root"]) / "receipt-signing.key"


def _load_receipt_key(config: dict[str, Any], *, create: bool) -> bytes:
    path = _receipt_key_path(config)
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                key = secrets.token_bytes(32)
                os.write(descriptor, key)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    try:
        key = path.read_bytes()
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ContinuityError(f"cannot read receipt signing key: {exc}") from exc
    if len(key) != 32 or mode & 0o077:
        raise ContinuityError("receipt signing key must be 32 bytes with mode 0600")
    return key


def _receipt_auth_payload(receipt: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in receipt.items() if key != "auth"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def receipt_signing_key(config: dict[str, Any]) -> bytes:
    return _load_receipt_key(config, create=True)


def sign_receipt(
    config: dict[str, Any], receipt: dict[str, Any], *, key: bytes | None = None
) -> dict[str, str]:
    key = key or _load_receipt_key(config, create=True)
    return {
        "algorithm": "hmac-sha256",
        "key_id": hashlib.sha256(key).hexdigest()[:16],
        "digest": hmac.new(
            key, _receipt_auth_payload(receipt), hashlib.sha256
        ).hexdigest(),
    }


def receipt_signature_errors(
    config: dict[str, Any], receipt: dict[str, Any]
) -> list[str]:
    auth = receipt.get("auth")
    if not isinstance(auth, dict) or set(auth) != {"algorithm", "key_id", "digest"}:
        return ["receipt authentication record is absent or malformed"]
    if auth.get("algorithm") != "hmac-sha256":
        return ["receipt authentication algorithm is invalid"]
    try:
        key = _load_receipt_key(config, create=False)
    except ContinuityError as exc:
        return [str(exc)]
    expected_id = hashlib.sha256(key).hexdigest()[:16]
    expected_digest = hmac.new(
        key, _receipt_auth_payload(receipt), hashlib.sha256
    ).hexdigest()
    errors: list[str] = []
    if not hmac.compare_digest(str(auth.get("key_id")), expected_id):
        errors.append("receipt signing key identity does not match")
    if not hmac.compare_digest(str(auth.get("digest")), expected_digest):
        errors.append("receipt signature is invalid")
    return errors


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContinuityError(f"command failed or timed out: {args[0]}: {exc}") from exc


def git_state(
    cwd: Path, timeout: float = 10, integration_ref: str | None = None
) -> GitState:
    def git(*args: str) -> str:
        result = run_command(["git", *args], cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ContinuityError(f"git {' '.join(args)} failed: {detail}")
        return result.stdout.strip()

    root = str(Path(git("rev-parse", "--show-toplevel")).resolve())
    branch = git("branch", "--show-current") or "DETACHED"
    commit = git("rev-parse", "HEAD")
    status_result = run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=cwd,
        timeout=timeout,
    )
    if status_result.returncode != 0:
        detail = (status_result.stderr or status_result.stdout).strip()
        raise ContinuityError(f"git status failed: {detail}")
    status = status_result.stdout.rstrip("\r\n")
    changed = tuple(
        line[3:] if len(line) >= 4 else line for line in status.splitlines()
    )
    digest = hashlib.sha256()
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if diff_result.returncode != 0:
        raise ContinuityError("git diff failed while fingerprinting exact state")
    digest.update(diff_result.stdout)
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if untracked_result.returncode != 0:
        raise ContinuityError("git ls-files failed while fingerprinting exact state")
    for encoded in sorted(
        item for item in untracked_result.stdout.split(b"\0") if item
    ):
        relative = os.fsdecode(encoded)
        candidate = Path(root) / relative
        digest.update(encoded)
        if candidate.is_symlink():
            digest.update(
                os.readlink(candidate).encode("utf-8", errors="surrogateescape")
            )
        elif candidate.is_file():
            digest.update(candidate.read_bytes())
    integration_sha = git("rev-parse", integration_ref) if integration_ref else None
    merge_base = git("merge-base", integration_ref, "HEAD") if integration_ref else None
    return GitState(
        root,
        branch,
        commit,
        bool(status),
        changed,
        digest.hexdigest(),
        integration_ref,
        integration_sha,
        merge_base,
    )


def external_volume_available(path: Path) -> bool:
    """Require a mounted volume, not merely a same-named internal directory."""
    try:
        return path.exists() and path.is_mount()
    except OSError:
        return False


def compact_json(value: dict[str, Any], max_chars: int) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(rendered) <= max_chars:
        return rendered
    minimal = {
        "schema_version": value.get("schema_version", 1),
        "status": "BLOCKED",
        "completion_allowed": False,
        "errors": ["preflight output exceeded configured bound"],
    }
    return json.dumps(minimal, sort_keys=True, separators=(",", ":"))


def receipt_errors(receipt: dict[str, Any], current: GitState) -> list[str]:
    """Return every reason an exact-state receipt cannot authorize promotion."""
    errors: list[str] = []
    allowed_receipt = {
        "schema_version",
        "gate",
        "project",
        "repo_id",
        "lifecycle_target",
        "risk_tier",
        "created_at",
        "git",
        "authority_check",
        "external_inputs",
        "commands",
        "runtime_checks",
        "rollback",
        "result",
        "auth",
    }
    unknown_receipt = sorted(set(receipt) - allowed_receipt)
    if unknown_receipt:
        errors.append("receipt contains unknown keys: " + ", ".join(unknown_receipt))
    if receipt.get("gate") != "hermes-continuity-gate/2":
        errors.append("receipt was not emitted by hermes-continuity-gate/2")
    auth = receipt.get("auth")
    if not isinstance(auth, dict) or set(auth) != {"algorithm", "key_id", "digest"}:
        errors.append("receipt authentication record is absent or malformed")
    external_inputs = receipt.get("external_inputs")
    if not isinstance(external_inputs, dict) or not external_inputs:
        errors.append("receipt has no external input digests")
    target = receipt.get("lifecycle_target")
    if target not in {"TESTED", "ENFORCED"}:
        errors.append(f"invalid lifecycle target: {target!r}")
    expected_git = current.as_dict()
    if receipt.get("git") != expected_git:
        errors.append("receipt repository, branch, commit, or dirty state is stale")
    if current.dirty:
        errors.append("terminal lifecycle receipt requires a clean repository")
    if current.integration_ref and current.merge_base != current.integration_sha:
        errors.append("current branch is not based on the configured integration SHA")
    authority = receipt.get("authority_check")
    if not isinstance(authority, dict) or authority.get("passed") is not True:
        errors.append("strict authority check is absent or failed")

    tier = receipt.get("risk_tier")
    if tier not in {"T0", "T1", "T2", "T3"}:
        errors.append(f"invalid risk tier: {tier!r}")
        tier = "T3"
    commands = receipt.get("commands")
    if not isinstance(commands, list) or (tier != "T0" and not commands):
        errors.append("required command evidence is absent")
        commands = []
    has_positive_test = False
    for command in commands:
        if not isinstance(command, dict):
            errors.append("command evidence is malformed")
            continue
        name = command.get("name", "unnamed")
        allowed_command = {
            "name",
            "kind",
            "command_sha256",
            "cwd",
            "started_at",
            "completed_at",
            "duration_ms",
            "exit_code",
            "output_sha256",
            "scope",
            "test_count",
            "failed_count",
            "skipped",
            "flaky",
        }
        unknown = sorted(set(command) - allowed_command)
        if unknown:
            errors.append(
                f"command contains unknown keys ({name}): {', '.join(unknown)}"
            )
        if command.get("kind") != "command":
            errors.append(f"command has invalid kind: {name}")
        if command.get("exit_code") != 0:
            errors.append(f"command failed: {name}")
        if command.get("failed_count") != 0:
            errors.append(f"command reports failed tests: {name}")
        test_count = command.get("test_count", 0)
        skipped = command.get("skipped", 0)
        if not isinstance(test_count, int) or test_count < 0:
            errors.append(f"command has invalid test count: {name}")
        elif test_count > 0:
            has_positive_test = True
        if not isinstance(skipped, int) or skipped < 0:
            errors.append(f"command has invalid skip count: {name}")
        elif skipped != 0:
            errors.append(f"command contains skips: {name}")
        if command.get("flaky") is True:
            errors.append(f"command is flaky: {name}")
    if tier != "T0" and not has_positive_test:
        errors.append("receipt has no command with a positive test count")
    if target == "ENFORCED" and not any(
        isinstance(command, dict)
        and command.get("scope") == "full"
        and command.get("exit_code") == 0
        for command in commands
    ):
        errors.append("ENFORCED requires a passing full-suite or CI command")

    runtime_checks = receipt.get("runtime_checks")
    if tier in {"T2", "T3"} and (
        not isinstance(runtime_checks, list) or not runtime_checks
    ):
        errors.append("runtime checks are absent for T2/T3")
        runtime_checks = []
    for check in runtime_checks or []:
        if isinstance(check, dict):
            allowed_runtime = {
                "name",
                "kind",
                "command_sha256",
                "cwd",
                "started_at",
                "completed_at",
                "duration_ms",
                "exit_code",
                "output_sha256",
                "surface",
                "passed",
            }
            unknown = sorted(set(check) - allowed_runtime)
            if unknown:
                errors.append(
                    f"runtime check contains unknown keys ({check.get('name', 'unnamed')}): "
                    + ", ".join(unknown)
                )
        if (
            not isinstance(check, dict)
            or check.get("kind") != "runtime"
            or check.get("passed") is not True
            or check.get("exit_code") != 0
        ):
            errors.append(
                f"runtime check failed: {check.get('name', 'unnamed') if isinstance(check, dict) else 'malformed'}"
            )

    rollback = receipt.get("rollback")
    if tier == "T3" and (
        not isinstance(rollback, dict)
        or rollback.get("kind") != "rollback"
        or rollback.get("passed") is not True
        or rollback.get("exit_code") != 0
    ):
        errors.append("rollback proof is absent or failed for T3")
    if isinstance(rollback, dict) and rollback:
        allowed_rollback = {
            "name",
            "kind",
            "command_sha256",
            "cwd",
            "started_at",
            "completed_at",
            "duration_ms",
            "exit_code",
            "output_sha256",
            "mode",
            "passed",
        }
        unknown = sorted(set(rollback) - allowed_rollback)
        if unknown:
            errors.append("rollback contains unknown keys: " + ", ".join(unknown))
    return errors
