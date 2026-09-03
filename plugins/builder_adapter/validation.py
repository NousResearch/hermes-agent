"""Registered validation in OS-owned disposable execution units."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .errors import AdapterError
from .gitops import _run_git


_LAUNCHCTL = "/bin/launchctl"
_SUPERVISOR = Path(__file__).with_name("validation_supervisor.py")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError:
        pass
    return digest.hexdigest()


def _secure_write(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_LAUNCHCTL, *args],
        check=False,
        capture_output=True,
        timeout=10,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def _darwin_user_temp_dir() -> Path:
    """Resolve launchd-loadable owner-only storage without inherited TMPDIR."""
    inherited = os.environ.get("TMPDIR")
    candidates = (
        [Path(inherited)]
        if inherited
        else list(Path("/var/folders").glob("*/*/T"))
    )
    valid = []
    for candidate in candidates:
        try:
            root = candidate.resolve(strict=True)
            stat = root.stat()
        except OSError:
            continue
        if (
            candidate.is_symlink()
            or not root.is_absolute()
            or not root.is_dir()
            or stat.st_uid != os.getuid()
            or stat.st_mode & 0o077
        ):
            continue
        # launchd accepts the OS-advertised /var/folders spelling but may
        # reject its /private/var canonical alias with an opaque EIO.
        valid.append(candidate.absolute())
    if len(set(valid)) != 1:
        raise AdapterError(
            "VALIDATION_CONTAINMENT_UNAVAILABLE",
            "unique owner-only Darwin temporary directory is unavailable",
        )
    return valid[0]


class _UnverifiedLaunchdContainmentProbe:
    """Test-only launchd probe; never selected by the runtime."""

    def __init__(self, *, python: str):
        self.python = str(Path(python).resolve(strict=True))

    def run(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        *,
        scope_id: str,
    ) -> dict:
        if sys.platform != "darwin" or not Path(_LAUNCHCTL).is_file():
            raise AdapterError(
                "VALIDATION_CONTAINMENT_UNAVAILABLE",
                "OS-owned disposable validation unit is unavailable",
            )
        uid = os.getuid()
        token = secrets.token_hex(16)
        safe_scope = "".join(
            character if character.isalnum() else "-"
            for character in scope_id[:40]
        ).strip("-") or "validation"
        label = f"com.hermes.validation.{safe_scope}.{token}"
        domain = f"gui/{uid}"
        service = f"{domain}/{label}"
        with tempfile.TemporaryDirectory(
            prefix="hermes-validation-unit-",
            dir=_darwin_user_temp_dir(),
        ) as raw:
            unit_dir = Path(raw)
            spec_path = unit_dir / "spec.json"
            result_path = unit_dir / "result.json"
            stdout_path = unit_dir / "stdout"
            stderr_path = unit_dir / "stderr"
            plist_path = unit_dir / f"{label}.plist"
            _secure_write(
                spec_path,
                json.dumps(
                    {"argv": argv, "cwd": str(cwd), "env": env},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )
            plist = {
                "Label": label,
                "ProgramArguments": [
                    self.python,
                    str(_SUPERVISOR),
                    str(spec_path),
                    str(result_path),
                ],
                "RunAtLoad": True,
                "KeepAlive": False,
                "AbandonProcessGroup": False,
                "ProcessType": "Background",
                "ExitTimeOut": 1,
                "StandardOutPath": str(stdout_path),
                "StandardErrorPath": str(stderr_path),
            }
            _secure_write(
                plist_path,
                plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=True),
            )
            bootstrap = _launchctl("bootstrap", domain, str(plist_path))
            if bootstrap.returncode != 0:
                detail = bootstrap.stderr.decode("utf-8", errors="replace").strip()
                supervisor_detail = ""
                try:
                    supervisor_detail = stderr_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                except OSError:
                    pass
                raise AdapterError(
                    "VALIDATION_CONTAINMENT_UNAVAILABLE",
                    "transient launchd validation job could not be created"
                    + (f": {detail}" if detail else "")
                    + (
                        f"; supervisor: {supervisor_detail}"
                        if supervisor_detail
                        else ""
                    ),
                )
            result = None
            timed_out = False
            supervisor_error = False
            deadline = time.monotonic() + int(timeout_seconds)
            try:
                while time.monotonic() < deadline:
                    if result_path.is_file():
                        try:
                            result = json.loads(result_path.read_text("utf-8"))
                            break
                        except (OSError, ValueError):
                            supervisor_error = True
                            break
                    state = _launchctl("print", service)
                    if state.returncode != 0:
                        supervisor_error = True
                        break
                    time.sleep(0.05)
                else:
                    timed_out = True
            finally:
                bootout = _launchctl("bootout", service)
                absent = False
                for _attempt in range(100):
                    if _launchctl("print", service).returncode != 0:
                        absent = True
                        break
                    time.sleep(0.05)
            if bootout.returncode != 0 or not absent:
                raise AdapterError(
                    "VALIDATION_CONTAINMENT_UNAVAILABLE",
                    "transient validation job removal was not confirmed",
                )
            if timed_out:
                status = 124
            elif supervisor_error or not isinstance(result, dict):
                raise AdapterError(
                    "VALIDATION_CONTAINMENT_UNAVAILABLE",
                    "validation supervisor terminal state was not confirmed",
                )
            else:
                status = int(result["exit_status"])
            return {
                "exit_status": status,
                "stdout_sha256": _hash_file(stdout_path),
                "stderr_sha256": _hash_file(stderr_path),
                "containment": {
                    "kind": "darwin_transient_user_launchd_job",
                    "job_label": label,
                    "domain": domain,
                    "bootout_exit_status": bootout.returncode,
                    "job_absence_confirmed": absent,
                    "timed_out": timed_out,
                    "supervisor_error": supervisor_error,
                },
            }


class _DockerContainment:
    """Run a sealed Git tree in one disposable, network-disabled container."""

    def __init__(self, *, docker: str, docker_host: str, image_id: str):
        self.docker = str(Path(docker).resolve(strict=True))
        self.docker_host = docker_host
        if not docker_host.startswith("unix://"):
            raise AdapterError(
                "VALIDATION_CONTAINMENT_UNAVAILABLE",
                "validation Docker endpoint must be a local Unix socket",
            )
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise AdapterError("MANIFEST_MISMATCH", "validation image ID is invalid")
        self.image_id = image_id

    def _environment(self, docker_config: Path) -> dict[str, str]:
        return {
            "DOCKER_CONFIG": str(docker_config),
            "DOCKER_HOST": self.docker_host,
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }

    def _docker(
        self,
        *args: str,
        docker_config: Path,
        timeout: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [self.docker, *args],
            check=False,
            capture_output=True,
            timeout=timeout,
            env=self._environment(docker_config),
        )
        if check and result.returncode != 0:
            raise AdapterError(
                "VALIDATION_CONTAINMENT_UNAVAILABLE",
                "isolated Docker operation failed",
            )
        return result

    @staticmethod
    def _verify_regular_tree(worktree: Path, commit: str) -> None:
        raw = _run_git(
            worktree, "ls-tree", "-rz", "--full-tree", "-r", commit
        ).stdout
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            metadata, _path = entry.split(b"\t", 1)
            mode, kind, _object_id = metadata.split(b" ", 2)
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise AdapterError(
                    "MANIFEST_MISMATCH",
                    "validation source contains a non-regular Git object",
                )

    def run(
        self,
        profile_id: str,
        profile: dict,
        worktree: Path,
        commit: str,
        *,
        scope_id: str,
    ) -> dict:
        self._verify_regular_tree(worktree, commit)
        safe_scope = "".join(
            character if character.isalnum() else "-"
            for character in scope_id[:36]
        ).strip("-") or "validation"
        token = secrets.token_hex(8)
        container = f"hermes-validation-{safe_scope}-{token}"
        volume = f"hermes-validation-source-{safe_scope}-{token}"
        commands = []
        overall = "PASSED"
        with tempfile.TemporaryDirectory(prefix="hermes-docker-config-") as raw:
            docker_config = Path(raw)
            (docker_config / "config.json").write_text("{}\n", encoding="utf-8")
            os.chmod(docker_config / "config.json", 0o600)
            inspected = self._docker(
                "image",
                "inspect",
                self.image_id,
                "--format",
                "{{.Id}}",
                docker_config=docker_config,
            )
            if inspected.stdout.decode().strip() != self.image_id:
                raise AdapterError("MANIFEST_MISMATCH", "validation image mismatch")
            created = False
            volume_created = False
            try:
                self._docker(
                    "volume",
                    "create",
                    "--label",
                    f"io.hermes.validation.scope={safe_scope}",
                    volume,
                    docker_config=docker_config,
                )
                volume_created = True
                archive = subprocess.Popen(
                    [
                        "/usr/bin/git",
                        "-C",
                        str(worktree),
                        "archive",
                        "--format=tar",
                        commit,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={
                        "HOME": "/nonexistent",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PATH": "/usr/bin:/bin",
                    },
                )
                assert archive.stdout is not None
                populated = subprocess.run(
                    [
                        self.docker,
                        "run",
                        "--rm",
                        "-i",
                        "--network",
                        "none",
                        "--read-only",
                        "--user",
                        "0:0",
                        "--entrypoint",
                        "/bin/tar",
                        "--mount",
                        f"type=volume,src={volume},dst=/work/source",
                        self.image_id,
                        "-xf",
                        "-",
                        "-C",
                        "/work/source",
                    ],
                    stdin=archive.stdout,
                    capture_output=True,
                    timeout=300,
                    env=self._environment(docker_config),
                )
                archive.stdout.close()
                archive_stderr = archive.communicate(timeout=30)[1]
                if archive.returncode != 0 or populated.returncode != 0:
                    raise AdapterError(
                        "VALIDATION_CONTAINMENT_UNAVAILABLE",
                        "sealed source export could not be materialized",
                    )
                if archive_stderr:
                    raise AdapterError(
                        "VALIDATION_CONTAINMENT_UNAVAILABLE",
                        "sealed source export emitted diagnostics",
                    )
                self._docker(
                    "create",
                    "--name",
                    container,
                    "--network",
                    "none",
                    "--read-only",
                    "--entrypoint",
                    "/bin/sh",
                    "--mount",
                    f"type=volume,src={volume},dst=/work/source,readonly",
                    "--tmpfs",
                    "/work/tmp:rw,nosuid,nodev,size=256m",
                    "--tmpfs",
                    "/work/exec-tmp:rw,exec,nosuid,nodev,size=256m",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=64m",
                    self.image_id,
                    "-c",
                    "while :; do sleep 3600; done",
                    docker_config=docker_config,
                )
                created = True
                self._docker("start", container, docker_config=docker_config)
                self._docker(
                    "exec",
                    "-u",
                    "0:0",
                    container,
                    "/bin/sh",
                    "-ceu",
                    "mkdir -p /work/tmp/home /work/tmp/pytest-cache /work/tmp/ruff-cache /work/tmp/cache /work/exec-tmp/pytest-work; chown -R 501:20 /work/tmp /work/exec-tmp",
                    docker_config=docker_config,
                )
                clean_env = {
                    "HOME": "/work/tmp/home",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONHASHSEED": "0",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": "/work/source",
                    "RUFF_CACHE_DIR": "/work/tmp/ruff-cache",
                    "TMPDIR": "/work/exec-tmp",
                    "TZ": "UTC",
                    "XDG_CACHE_HOME": "/work/tmp/cache",
                }
                for spec in profile.get("commands", []):
                    argv = [
                        "/usr/local/bin/python" if value == "{python}" else str(value)
                        for value in spec["argv"]
                    ]
                    started_at = _utc_now()
                    exec_argv = [self.docker, "exec", "-u", "501:20"]
                    for name, value in clean_env.items():
                        exec_argv.extend(["-e", f"{name}={value}"])
                    exec_argv.extend(
                        ["-w", spec.get("working_directory", "/work/source"), container, *argv]
                    )
                    try:
                        result = subprocess.run(
                            exec_argv,
                            check=False,
                            capture_output=True,
                            timeout=int(spec["timeout_seconds"]),
                            env=self._environment(docker_config),
                        )
                        status = result.returncode
                        stdout = result.stdout
                        stderr = result.stderr
                    except subprocess.TimeoutExpired as exc:
                        status = 124
                        stdout = exc.stdout or b""
                        stderr = exc.stderr or b""
                    finished_at = _utc_now()
                    commands.append(
                        {
                            "command_id": spec["command_id"],
                            "argv": argv,
                            "exit_status": status,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                        }
                    )
                    if status != 0 and spec.get("required", True):
                        overall = "FAILED"
            finally:
                if created:
                    self._docker(
                        "rm", "-f", container, docker_config=docker_config, check=False
                    )
                    absent = self._docker(
                        "container",
                        "inspect",
                        container,
                        docker_config=docker_config,
                        check=False,
                    )
                    if absent.returncode == 0:
                        raise AdapterError(
                            "VALIDATION_CONTAINMENT_UNAVAILABLE",
                            "validation container removal was not confirmed",
                        )
                if volume_created:
                    self._docker(
                        "volume", "rm", volume, docker_config=docker_config, check=False
                    )
                    volume_absent = self._docker(
                        "volume",
                        "inspect",
                        volume,
                        docker_config=docker_config,
                        check=False,
                    )
                    if volume_absent.returncode == 0:
                        raise AdapterError(
                            "VALIDATION_CONTAINMENT_UNAVAILABLE",
                            "validation source volume removal was not confirmed",
                        )
        return {"profile": profile_id, "commands": commands, "overall_status": overall}


class ValidationRunner:
    def __init__(
        self,
        profiles: dict[str, dict],
        *,
        python: str,
        docker: str | None = None,
        docker_host: str | None = None,
        image_id: str | None = None,
    ):
        self._profiles = profiles
        self._docker = (
            _DockerContainment(
                docker=docker, docker_host=docker_host, image_id=image_id
            )
            if docker and docker_host and image_id
            else None
        )

    def run(
        self,
        profile_id: str,
        worktree: Path,
        expected_sha: str,
        *,
        materialized_sha: str | None = None,
        scope_id: str = "validation",
    ) -> dict:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise AdapterError("MANIFEST_MISMATCH", "validation profile unregistered")
        head = (
            materialized_sha
            if materialized_sha is not None
            else _run_git(worktree, "rev-parse", "HEAD").stdout.decode().strip()
        )
        if head != expected_sha:
            raise AdapterError("HEAD_MISMATCH", "validation HEAD mismatch")
        if self._docker is not None:
            return self._docker.run(
                profile_id,
                profile,
                worktree,
                expected_sha,
                scope_id=scope_id,
            )
        raise AdapterError(
            "VALIDATION_CONTAINMENT_UNAVAILABLE",
            "launchd descendant containment is unproven; "
            "registered validation requires a disposable container or VM",
        )
