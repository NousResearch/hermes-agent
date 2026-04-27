"""OpenShell-backed Hermes execution environment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.sandbox_provider import SandboxProvider, SandboxRuntimeInfo, SandboxSpec
from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.file_sync import FileSyncManager, iter_sync_files

logger = logging.getLogger(__name__)


DEFAULT_MIRROR_EXCLUDES = [
    ".git",
    ".hermes",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
]
DEFAULT_GATEWAY_NAME = "hermes-openshell"
DEFAULT_GATEWAY_PORT = 18080


def bundled_source_dir() -> Path:
    return Path(__file__).resolve().parent / "source"


def normalize_openshell_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    source = str(cfg.get("source") or cfg.get("from") or "").strip()
    if not source or source == "bundled-hermes-source":
        source = str(bundled_source_dir())
    mode = str(cfg.get("mode") or "mirror").strip().lower()
    if mode not in ("mirror", "remote"):
        mode = "mirror"
    providers = cfg.get("providers") or []
    if isinstance(providers, str):
        providers = [part.strip() for part in providers.split(",") if part.strip()]
    gateway_port = cfg.get("gateway_port", cfg.get("port", DEFAULT_GATEWAY_PORT))
    try:
        gateway_port = int(gateway_port)
    except (TypeError, ValueError):
        gateway_port = DEFAULT_GATEWAY_PORT
    return {
        "command": str(cfg.get("command") or "openshell").strip() or "openshell",
        "mode": mode,
        "source": source,
        "policy": str(cfg.get("policy") or "").strip(),
        "providers": [str(p) for p in providers if str(p).strip()],
        "auto_providers": bool(cfg.get("auto_providers", True)),
        "gpu": bool(cfg.get("gpu", False)),
        "gateway": str(cfg.get("gateway") or DEFAULT_GATEWAY_NAME).strip() or DEFAULT_GATEWAY_NAME,
        "gateway_endpoint": str(cfg.get("gateway_endpoint") or "").strip(),
        "gateway_port": gateway_port if gateway_port > 0 else 0,
        "gateway_host": str(cfg.get("gateway_host") or "").strip(),
        "remote_workspace_dir": str(cfg.get("remote_workspace_dir") or "/sandbox").rstrip("/") or "/sandbox",
        "remote_agent_workspace_dir": str(cfg.get("remote_agent_workspace_dir") or "/agent").rstrip("/") or "/agent",
        "timeout_seconds": int(cfg.get("timeout_seconds") or 120),
        "mirror_excludes": _normalize_mirror_excludes(cfg.get("mirror_excludes")),
        "extra_syncs": _normalize_extra_syncs(cfg.get("extra_syncs") or cfg.get("extra_dirs") or []),
        "sync_hermes_files": bool(cfg.get("sync_hermes_files", False)),
    }


def build_sandbox_name(scope: str) -> str:
    trimmed = (scope or "session").strip() or "session"
    digest = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:8]
    max_slug_len = 63 - len("hermes") - 1 - len(digest) - 1
    safe = re.sub(r"[^a-z0-9]+", "-", trimmed.lower())
    safe = re.sub(r"-+", "-", safe).strip("-")[:max_slug_len].strip("-") or "session"
    return f"hermes-{safe}-{digest}"


def prepare_openshell_source(
    cfg: dict[str, Any],
    *,
    cwd: str | None = None,
    stream: bool = False,
) -> str:
    """Return a source suitable for OpenShell, importing local builds when possible."""

    source = str(cfg.get("source") or "").strip()
    source_path = _resolve_local_source_path(source, cwd=cwd)
    if source_path is None:
        return source

    docker = shutil.which("docker")
    if not docker or cfg.get("gateway_endpoint"):
        return source

    context_dir, dockerfile = _source_build_paths(source_path)
    if dockerfile is None:
        return source

    container = _resolve_gateway_container(docker, str(cfg.get("gateway") or DEFAULT_GATEWAY_NAME))
    if not container:
        return source

    digest = _hash_local_source(context_dir)
    tag = f"openshell/sandbox-from:hermes-{digest[:16]}"
    if not _host_docker_has_image(docker, tag):
        if stream:
            print(f"Preparing OpenShell source image {tag}...")
        build = subprocess.run(
            [docker, "build", "-t", tag, "-f", str(dockerfile), str(context_dir)],
            capture_output=not stream,
            text=True,
            timeout=max(int(cfg.get("timeout_seconds") or 120), 300),
        )
        if build.returncode != 0:
            return source

    if _gateway_container_has_image(docker, container, tag):
        return tag
    if _import_image_into_gateway(docker, container, tag, timeout=max(int(cfg.get("timeout_seconds") or 120), 300)):
        return tag
    return source


def _resolve_local_source_path(source: str, *, cwd: str | None = None) -> Path | None:
    if not source or "://" in source:
        return None
    raw = Path(source).expanduser()
    candidates = [raw]
    if not raw.is_absolute() and cwd:
        candidates.insert(0, Path(cwd).expanduser() / raw)
    for candidate in candidates:
        if candidate.exists() and (candidate.is_dir() or candidate.is_file()):
            return candidate.resolve()
    return None


def _source_build_paths(source_path: Path) -> tuple[Path, Path | None]:
    if source_path.is_file():
        return source_path.parent, source_path
    dockerfile = source_path / "Dockerfile"
    if dockerfile.is_file():
        return source_path, dockerfile
    return source_path, None


def _hash_local_source(context_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"hermes-openshell-source-v1\0")
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for path in sorted(context_dir.rglob("*")):
        rel = path.relative_to(context_dir).as_posix()
        if any(part in skip_dirs for part in path.relative_to(context_dir).parts):
            continue
        if path.is_dir():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8"))
            digest.update(b"\0")
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _host_docker_has_image(docker: str, tag: str) -> bool:
    result = subprocess.run(
        [docker, "image", "inspect", tag],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


def _resolve_gateway_container(docker: str, gateway: str) -> str | None:
    candidates = [f"openshell-cluster-{gateway}", f"openshell-{gateway}", gateway]
    for candidate in candidates:
        result = subprocess.run(
            [docker, "container", "inspect", candidate],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return candidate
    result = subprocess.run(
        [docker, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    suffix = f"-{gateway}"
    for name in result.stdout.splitlines():
        if name == gateway or name.endswith(suffix):
            return name
    return None


def _gateway_container_has_image(docker: str, container: str, tag: str) -> bool:
    result = subprocess.run(
        [docker, "exec", container, "ctr", "-n", "k8s.io", "images", "list", "-q"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return False
    image_names = set(result.stdout.splitlines())
    return tag in image_names or f"docker.io/{tag}" in image_names


def _import_image_into_gateway(docker: str, container: str, tag: str, *, timeout: int) -> bool:
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag)
    remote_tar = f"/tmp/hermes-openshell-{safe_tag}.tar"
    with tempfile.NamedTemporaryFile(prefix="hermes-openshell-image-", suffix=".tar") as tmp:
        save = subprocess.run(
            [docker, "image", "save", "-o", tmp.name, tag],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if save.returncode != 0:
            return False
        copied = subprocess.run(
            [docker, "cp", tmp.name, f"{container}:{remote_tar}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if copied.returncode != 0:
            return False
        try:
            imported = subprocess.run(
                [docker, "exec", container, "ctr", "-n", "k8s.io", "images", "import", remote_tar],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return imported.returncode == 0
        finally:
            subprocess.run(
                [docker, "exec", container, "rm", "-f", remote_tar],
                capture_output=True,
                text=True,
                timeout=60,
            )


def _openshell_base_argv(cfg: dict[str, Any]) -> list[str]:
    argv = [cfg["command"]]
    if cfg.get("gateway_endpoint"):
        argv.extend(["--gateway-endpoint", cfg["gateway_endpoint"]])
    elif cfg.get("gateway"):
        argv.extend(["--gateway", cfg["gateway"]])
    return argv


def _normalize_mirror_excludes(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    elif not isinstance(raw, list):
        raw = []
    merged: list[str] = []
    for item in [*DEFAULT_MIRROR_EXCLUDES, *raw]:
        value = str(item).strip()
        if value and value not in merged:
            merged.append(value)
    return merged


def _normalize_extra_syncs(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                raw = [stripped]
        else:
            raw = [part.strip() for part in stripped.split(";") if part.strip()]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    syncs: list[dict[str, Any]] = []
    for entry in raw:
        host_path = ""
        remote_path = ""
        mode = "inherit"
        excludes = DEFAULT_MIRROR_EXCLUDES
        if isinstance(entry, str):
            local, sep, remote = entry.partition(":")
            host_path = local.strip()
            remote_path = remote.strip() if sep else ""
        elif isinstance(entry, dict):
            host_path = str(
                entry.get("host")
                or entry.get("local")
                or entry.get("path")
                or entry.get("source")
                or ""
            ).strip()
            remote_path = str(
                entry.get("remote")
                or entry.get("target")
                or entry.get("sandbox_path")
                or ""
            ).strip()
            mode = str(entry.get("mode") or "inherit").strip().lower()
            excludes = entry.get("excludes") or DEFAULT_MIRROR_EXCLUDES
        if not host_path:
            continue
        host_path = os.path.abspath(os.path.expanduser(host_path))
        if not remote_path:
            remote_path = f"/sandbox/{Path(host_path).name}"
        elif not remote_path.startswith("/"):
            remote_path = f"/sandbox/{remote_path}"
        remote_path = remote_path.rstrip("/") or "/sandbox"
        if mode not in ("inherit", "mirror", "remote", "seed"):
            mode = "inherit"
        if isinstance(excludes, str):
            excludes = [part.strip() for part in excludes.split(",") if part.strip()]
        syncs.append(
            {
                "host": host_path,
                "remote": remote_path,
                "mode": mode,
                "excludes": list(excludes or DEFAULT_MIRROR_EXCLUDES),
            }
        )
    return syncs


def ensure_openshell_gateway(cfg: dict[str, Any], *, capture_output: bool = True) -> subprocess.CompletedProcess | None:
    """Start the configured local gateway before sandbox operations.

    OpenShell's sandbox auto-bootstrap uses port 8080. Hermes uses a named
    gateway on a separate default port so unrelated local services do not
    collide with sandbox startup. Set gateway_port to 0 or gateway_endpoint
    to skip local gateway management.
    """

    if cfg.get("gateway_endpoint") or not cfg.get("gateway_port"):
        return None
    args = [
        cfg["command"],
        "gateway",
        "start",
        "--name",
        cfg["gateway"],
        "--port",
        str(cfg["gateway_port"]),
    ]
    if cfg.get("gateway_host"):
        args.extend(["--gateway-host", cfg["gateway_host"]])
    if cfg.get("gpu"):
        args.append("--gpu")
    kwargs: dict[str, Any] = {
        "text": True,
        "timeout": max(cfg.get("timeout_seconds") or 120, 300),
    }
    if capture_output:
        kwargs["capture_output"] = True
    return subprocess.run(args, **kwargs)


def run_openshell_cli(
    cfg: dict[str, Any],
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_openshell_base_argv(cfg), *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout or cfg.get("timeout_seconds") or 120,
    )


class OpenShellSandboxProvider(SandboxProvider):
    name = "openshell"
    description = "OpenShell sandbox provider for Hermes tool execution"

    def is_available(self) -> bool:
        command = os.getenv("HERMES_OPENSHELL_COMMAND", "openshell")
        return shutil.which(command) is not None

    def create_environment(self, spec: SandboxSpec):
        return OpenShellEnvironment(spec)

    def describe_runtime(self, scope: str, config: dict[str, Any] | None = None) -> SandboxRuntimeInfo:
        cfg = normalize_openshell_config(config)
        name = build_sandbox_name(scope)
        result = run_openshell_cli(cfg, ["sandbox", "get", name], timeout=30)
        return SandboxRuntimeInfo(
            provider=self.name,
            runtime_id=name,
            running=result.returncode == 0,
            mode=cfg["mode"],
            source=cfg["source"],
            details={"stdout": result.stdout, "stderr": result.stderr},
        )

    def delete_runtime(self, scope: str, config: dict[str, Any] | None = None) -> None:
        cfg = normalize_openshell_config(config)
        name = build_sandbox_name(scope)
        result = run_openshell_cli(cfg, ["sandbox", "delete", name], timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "openshell sandbox delete failed")


class OpenShellEnvironment(BaseEnvironment):
    """Run Hermes tools inside an OpenShell sandbox over SSH."""

    def __init__(self, spec: SandboxSpec):
        self.spec = spec
        self.cfg = normalize_openshell_config(spec.config)
        self.sandbox_name = build_sandbox_name(spec.scope or spec.task_id)
        self.host_workspace = Path(spec.host_cwd or os.getcwd()).resolve()
        self.remote_workspace = self.cfg["remote_workspace_dir"]
        self.remote_agent_workspace = self.cfg["remote_agent_workspace_dir"]
        self._tmpdir = Path(tempfile.mkdtemp(prefix="hermes-openshell-"))
        self._ssh_config_path = self._tmpdir / "ssh_config"
        self._ssh_host = self.sandbox_name
        self._remote_home = "/root"
        self._workspace_seeded = False
        self._sync_manager: FileSyncManager | None = None
        self._closed = False

        super().__init__(
            cwd=spec.cwd or self.remote_workspace,
            timeout=spec.timeout or self.cfg["timeout_seconds"],
        )
        self._ensure_sandbox_exists()
        self._refresh_ssh_config()
        self._remote_home = self._detect_remote_home()
        self._ensure_remote_dirs()
        if self.cfg.get("sync_hermes_files"):
            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
                upload_fn=self._upload_file,
                delete_fn=self._delete_remote_paths,
                bulk_upload_fn=self._bulk_upload_files,
            )
        self._seed_workspace()
        self.init_session()

    def _run_openshell(self, args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
        return run_openshell_cli(
            self.cfg,
            args,
            cwd=str(self.host_workspace) if self.host_workspace.is_dir() else None,
            timeout=timeout,
        )

    def _ensure_sandbox_exists(self) -> None:
        gateway = ensure_openshell_gateway(self.cfg)
        if gateway is not None and gateway.returncode != 0:
            raise RuntimeError(gateway.stderr.strip() or gateway.stdout.strip() or "openshell gateway start failed")

        result = self._run_openshell(["sandbox", "get", self.sandbox_name], timeout=30)
        if result.returncode == 0:
            return

        source = prepare_openshell_source(self.cfg, cwd=str(self.host_workspace))
        args = [
            "sandbox",
            "create",
            "--name",
            self.sandbox_name,
            "--from",
            source,
        ]
        if self.cfg.get("policy"):
            args.extend(["--policy", self.cfg["policy"]])
        if self.cfg.get("gpu"):
            args.append("--gpu")
        args.append("--auto-providers" if self.cfg.get("auto_providers", True) else "--no-auto-providers")
        for provider in self.cfg.get("providers", []):
            args.extend(["--provider", provider])
        args.extend(["--", "true"])

        create = self._run_openshell(args, timeout=max(self.cfg["timeout_seconds"], 300))
        if create.returncode != 0:
            raise RuntimeError(create.stderr.strip() or create.stdout.strip() or "openshell sandbox create failed")

    def _refresh_ssh_config(self) -> None:
        result = self._run_openshell(["sandbox", "ssh-config", self.sandbox_name], timeout=60)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "openshell sandbox ssh-config failed")
        config_text = result.stdout
        self._ssh_config_path.write_text(config_text, encoding="utf-8")
        for line in config_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("host "):
                parts = stripped.split()
                if len(parts) >= 2 and "*" not in parts[1]:
                    self._ssh_host = parts[1]
                    break

    def _ssh_argv(self, remote_command: str) -> list[str]:
        return ["ssh", "-F", str(self._ssh_config_path), "-T", self._ssh_host, remote_command]

    def _run_ssh_command(self, remote_command: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh_argv(remote_command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _detect_remote_home(self) -> str:
        result = self._run_ssh_command("printf '%s\\n' \"$HOME\"", timeout=30)
        home = result.stdout.strip()
        return home or "/root"

    def _ensure_remote_dirs(self) -> None:
        dirs = [
            f"{self._remote_home}/.hermes",
            f"{self._remote_home}/.hermes/skills",
            f"{self._remote_home}/.hermes/credentials",
            f"{self._remote_home}/.hermes/cache",
            self.remote_workspace,
        ]
        quoted_dirs = " ".join(shlex.quote(path) for path in dirs)
        result = self._run_ssh_command(
            "for d in "
            + quoted_dirs
            + '; do if [ -e "$d" ] && [ ! -d "$d" ]; then rm -f -- "$d"; fi; mkdir -p -- "$d"; done',
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "failed to create OpenShell remote directories")

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        if login:
            remote_command = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            remote_command = f"bash -c {shlex.quote(cmd_string)}"
        return _popen_bash(self._ssh_argv(remote_command), stdin_data)

    def _before_execute(self) -> None:
        if self._sync_manager:
            self._sync_manager.sync()
        if self.cfg["mode"] == "mirror":
            self._sync_workspace_to_remote()
            self._sync_extra_paths_to_remote()
        elif not self._workspace_seeded:
            self._seed_workspace()

    def execute(self, *args, **kwargs) -> dict:
        try:
            return super().execute(*args, **kwargs)
        finally:
            if self.cfg["mode"] == "mirror":
                self._sync_workspace_from_remote()
                self._sync_extra_paths_from_remote()

    def _seed_workspace(self) -> None:
        if self._workspace_seeded:
            return
        self._sync_workspace_to_remote()
        self._sync_extra_paths_to_remote()
        self._workspace_seeded = True

    def _sync_workspace_to_remote(self) -> None:
        self._sync_path_to_remote(
            self.host_workspace,
            self.remote_workspace,
            set(self.cfg.get("mirror_excludes") or DEFAULT_MIRROR_EXCLUDES),
        )

    def _sync_path_to_remote(self, host_path: Path, remote_path: str, exclude_dirs: set[str]) -> None:
        if not host_path.exists():
            logger.warning("OpenShell extra sync path does not exist: %s", host_path)
            return
        if not self.host_workspace.is_dir():
            return
        if self._rsync_path_to_remote(host_path, remote_path, exclude_dirs):
            return
        self._run_ssh_command(
            f"mkdir -p {shlex.quote(remote_path)} && "
            f"find {shlex.quote(remote_path)} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +",
            timeout=60,
        )
        with tempfile.TemporaryDirectory(prefix="hermes-openshell-upload-") as tmp:
            staged = Path(tmp) / "workspace"
            _copy_tree_without_symlinks(
                host_path,
                staged,
                exclude_dirs,
            )
            result = self._run_openshell(
                [
                    "sandbox",
                    "upload",
                    "--no-git-ignore",
                    self.sandbox_name,
                    str(staged),
                    remote_path,
                ],
                timeout=max(self.cfg["timeout_seconds"], 300),
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "openshell sandbox upload failed")

    def _sync_workspace_from_remote(self) -> None:
        if not self.host_workspace.is_dir():
            return
        self._sync_path_from_remote(
            self.remote_workspace,
            self.host_workspace,
            set(self.cfg.get("mirror_excludes") or DEFAULT_MIRROR_EXCLUDES),
        )

    def _sync_path_from_remote(self, remote_path: str, host_path: Path, exclude_dirs: set[str]) -> None:
        if not host_path.is_dir():
            return
        if self._rsync_path_from_remote(remote_path, host_path, exclude_dirs):
            return
        with tempfile.TemporaryDirectory(prefix="hermes-openshell-download-") as tmp:
            result = self._run_openshell(
                ["sandbox", "download", self.sandbox_name, remote_path, tmp],
                timeout=max(self.cfg["timeout_seconds"], 300),
            )
            if result.returncode != 0:
                logger.warning(
                    "OpenShell workspace download failed: %s",
                    result.stderr.strip() or result.stdout.strip(),
                )
                return
            source = Path(tmp)
            maybe_nested = source / Path(remote_path).name
            if maybe_nested.is_dir() and len(list(source.iterdir())) == 1:
                source = maybe_nested
            _replace_directory_contents(
                source,
                host_path,
                exclude_dirs,
            )

    def _rsync_path_to_remote(self, host_path: Path, remote_path: str, exclude_dirs: set[str]) -> bool:
        if not shutil.which("rsync"):
            return False
        mkdir = self._run_ssh_command(f"mkdir -p {shlex.quote(remote_path)}", timeout=30)
        if mkdir.returncode != 0:
            raise RuntimeError(mkdir.stderr.strip() or "failed to create OpenShell remote sync directory")
        result = subprocess.run(
            [
                "rsync",
                "-rt",
                "--delete",
                *self._rsync_exclude_args(exclude_dirs),
                "-e",
                f"ssh -F {self._ssh_config_path} -T",
                f"{str(host_path).rstrip('/')}/",
                f"{self._ssh_host}:{remote_path.rstrip('/')}/",
            ],
            capture_output=True,
            text=True,
            timeout=max(self.cfg["timeout_seconds"], 300),
        )
        if result.returncode == 0:
            return True
        if result.returncode in {12, 127} or "rsync: command not found" in result.stderr:
            logger.warning("OpenShell rsync upload unavailable, falling back to sandbox upload: %s", result.stderr.strip())
            return False
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "OpenShell rsync upload failed")

    def _rsync_path_from_remote(self, remote_path: str, host_path: Path, exclude_dirs: set[str]) -> bool:
        if not shutil.which("rsync"):
            return False
        result = subprocess.run(
            [
                "rsync",
                "-rt",
                "--delete",
                *self._rsync_exclude_args(exclude_dirs),
                "-e",
                f"ssh -F {self._ssh_config_path} -T",
                f"{self._ssh_host}:{remote_path.rstrip('/')}/",
                f"{str(host_path).rstrip('/')}/",
            ],
            capture_output=True,
            text=True,
            timeout=max(self.cfg["timeout_seconds"], 300),
        )
        if result.returncode == 0:
            return True
        if result.returncode in {12, 127} or "rsync: command not found" in result.stderr:
            logger.warning("OpenShell rsync download unavailable, falling back to sandbox download: %s", result.stderr.strip())
            return False
        logger.warning("OpenShell rsync download failed: %s", result.stderr.strip() or result.stdout.strip())
        return True

    def _rsync_exclude_args(self, exclude_dirs: set[str]) -> list[str]:
        args: list[str] = []
        for item in sorted({str(entry).strip() for entry in exclude_dirs if str(entry).strip()}):
            args.extend(["--exclude", item])
        return args

    def _sync_extra_paths_to_remote(self) -> None:
        for sync in self.cfg.get("extra_syncs") or []:
            self._sync_path_to_remote(
                Path(sync["host"]),
                sync["remote"],
                set(sync.get("excludes") or DEFAULT_MIRROR_EXCLUDES),
            )

    def _sync_extra_paths_from_remote(self) -> None:
        for sync in self.cfg.get("extra_syncs") or []:
            mode = str(sync.get("mode") or "inherit").lower()
            should_pull = mode == "mirror" or (mode == "inherit" and self.cfg["mode"] == "mirror")
            if should_pull:
                self._sync_path_from_remote(
                    sync["remote"],
                    Path(sync["host"]),
                    set(sync.get("excludes") or DEFAULT_MIRROR_EXCLUDES),
                )

    def _upload_file(self, host_path: str, remote_path: str) -> None:
        parent = str(Path(remote_path).parent)
        quoted_parent = shlex.quote(parent)
        mkdir = self._run_ssh_command(
            f'if [ -e {quoted_parent} ] && [ ! -d {quoted_parent} ]; then rm -f -- {quoted_parent}; fi; '
            f"mkdir -p -- {quoted_parent}",
            timeout=30,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(mkdir.stderr.strip() or "failed to create remote OpenShell upload directory")
        result = self._run_openshell(
            ["sandbox", "upload", "--no-git-ignore", self.sandbox_name, host_path, remote_path],
            timeout=max(self.cfg["timeout_seconds"], 120),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "openshell sandbox upload failed")

    def _bulk_upload_files(self, files: list[tuple[str, str]]) -> None:
        for host_path, remote_path in files:
            self._upload_file(host_path, remote_path)

    def _delete_remote_paths(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        quoted = " ".join(shlex.quote(path) for path in remote_paths)
        result = self._run_ssh_command(f"rm -f {quoted}", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "failed to delete remote OpenShell paths")

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.cfg["mode"] == "mirror":
                self._sync_workspace_from_remote()
        finally:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def _copy_tree_without_symlinks(source: Path, target: Path, exclude_dirs: set[str]) -> None:
    if source.name.lower() in {item.lower() for item in exclude_dirs}:
        return
    try:
        stats = source.lstat()
    except OSError:
        return
    if stats.st_mode and source.is_symlink():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.name.lower() in {item.lower() for item in exclude_dirs}:
                continue
            _copy_tree_without_symlinks(child, target / child.name, exclude_dirs)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _replace_directory_contents(source: Path, target: Path, exclude_dirs: set[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    excluded = {item.lower() for item in exclude_dirs}
    for child in list(target.iterdir()):
        if child.name.lower() in excluded:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)
    for child in source.iterdir():
        if child.name.lower() in excluded:
            continue
        _copy_tree_without_symlinks(child, target / child.name, exclude_dirs)
