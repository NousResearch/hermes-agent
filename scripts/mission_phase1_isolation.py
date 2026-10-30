#!/usr/bin/env python3
"""Create and enforce a filesystem/network sandbox for Phase 1 Hermes integration runs.

This helper imports no Hermes modules. Runtime commands execute in bubblewrap with the host
filesystem read-only and only the fresh isolation root writable. Network access is disabled by
default; callers may opt into shared networking only for dependency installation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

MANIFEST = "phase1-isolation.json"
STATE_PATH_KEYS = (
    "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HERMES_HOME",
    "HERMES_RUNTIME_DIR", "HERMES_KANBAN_DB", "TMPDIR", "TMP", "TEMP",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "PIP_CACHE_DIR",
    "NPM_CONFIG_CACHE", "VIRTUAL_ENV", "HERMES_KANBAN_HOME", "HERMES_MANAGED_DIR",
    "HERMES_GATEWAY_LOCK_DIR", "HERMES_RPC_DIR", "HERMES_KERNEL_DIR",
    "HERMES_KERNEL_SPILL_DIR", "CLAUDE_CONFIG_DIR", "AGENT_BROWSER_PROFILE",
    "HERMES_ISOLATION_ROOT", "HERMES_PYTHON", "PYTHONPYCACHEPREFIX",
)


class IsolationError(RuntimeError):
    """An unsafe or incomplete isolation configuration was detected."""


def resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def overlaps(path: Path, protected: Path) -> bool:
    return is_within(path, protected) or is_within(protected, path)


def _protected(repo: Path, hermes_home: Path, kanban_db: Path, x_root: Path) -> list[Path]:
    return list(dict.fromkeys(resolved(p) for p in (repo, hermes_home, kanban_db, x_root)))


def _layout(root: Path) -> dict[str, Path]:
    profile = root / "hermes" / "profiles" / "fred"
    return {
        "root": root,
        "home": root / "home",
        "profile": profile,
        "runtime": root / "runtime",
        "database": profile / "phase1-kanban.db",
        "tmp": root / "tmp",
        "config": profile / "config.yaml",
        "xdg_config": root / "xdg" / "config",
        "xdg_cache": root / "xdg" / "cache",
        "xdg_data": root / "xdg" / "data",
        "xdg_state": root / "xdg" / "state",
        "xdg_runtime": root / "xdg" / "runtime",
        "uv_cache": root / "cache" / "uv",
        "uv_python": root / "cache" / "uv-python",
        "pip_cache": root / "cache" / "pip",
        "npm_cache": root / "cache" / "npm",
        "venv": root / "venv",
        "work": root / "work",
        "shm": root / "shm",
        "host_mask": root / "host-mask",
        "pycache": root / "cache" / "python",
        "kanban_home": root / "hermes",
        "managed": root / "managed",
        "locks": root / "locks",
        "rpc": root / "runtime" / "rpc",
        "kernel": root / "runtime" / "kernels",
        "kernel_spill": root / "tmp" / "kernel",
        "claude_config": root / "home" / ".claude",
        "browser_profile": root / "home" / ".browser-profile",
    }


def _validate_root(root: Path, repo: Path, protected: Iterable[Path]) -> Path:
    root = resolved(root)
    repo = resolved(repo)
    if root == Path(root.anchor):
        raise IsolationError("isolation root cannot be a filesystem root")
    if overlaps(root, repo):
        raise IsolationError(f"isolation root overlaps source checkout: {root}")
    for target in protected:
        target = resolved(target)
        if overlaps(root, target):
            raise IsolationError(f"isolation root overlaps protected path: {target}")
    if root.exists():
        raise IsolationError(f"isolation root already exists; refusing to reuse state: {root}")
    return root


def environment(root: Path, repo: Path) -> dict[str, str]:
    paths = _layout(root)
    env = {
        "HOME": str(paths["home"]),
        "USERPROFILE": str(paths["home"]),
        "APPDATA": str(paths["xdg_config"]),
        "LOCALAPPDATA": str(paths["xdg_data"]),
        "HERMES_HOME": str(paths["profile"]),
        "HERMES_ISOLATION_ROOT": str(root),
        "HERMES_PROFILE": "fred",
        "HERMES_PROFILE_NAME": "fred",
        "HERMES_RUNTIME_DIR": str(paths["runtime"]),
        "HERMES_KANBAN_DB": str(paths["database"]),
        "HERMES_KANBAN_HOME": str(paths["kanban_home"]),
        "HERMES_MANAGED_DIR": str(paths["managed"]),
        "HERMES_GATEWAY_LOCK_DIR": str(paths["locks"]),
        "HERMES_RPC_DIR": str(paths["rpc"]),
        "HERMES_KERNEL_DIR": str(paths["kernel"]),
        "HERMES_KERNEL_SPILL_DIR": str(paths["kernel_spill"]),
        "CLAUDE_CONFIG_DIR": str(paths["claude_config"]),
        "AGENT_BROWSER_PROFILE": str(paths["browser_profile"]),
        "TMPDIR": str(paths["tmp"]),
        "TMP": str(paths["tmp"]),
        "TEMP": str(paths["tmp"]),
        "XDG_CONFIG_HOME": str(paths["xdg_config"]),
        "XDG_CACHE_HOME": str(paths["xdg_cache"]),
        "XDG_DATA_HOME": str(paths["xdg_data"]),
        "XDG_STATE_HOME": str(paths["xdg_state"]),
        "XDG_RUNTIME_DIR": str(paths["xdg_runtime"]),
        "UV_CACHE_DIR": str(paths["uv_cache"]),
        "UV_PYTHON_INSTALL_DIR": str(paths["uv_python"]),
        "PIP_CACHE_DIR": str(paths["pip_cache"]),
        "NPM_CONFIG_CACHE": str(paths["npm_cache"]),
        "VIRTUAL_ENV": str(paths["venv"]),
        "HERMES_PYTHON": str(paths["venv"] / "bin" / "python"),
        "PATH": f"{paths['venv'] / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(repo),
        "HERMES_BUNDLED_PLUGINS": str(repo / "plugins"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "GIT_CONFIG_GLOBAL": str(paths["home"] / ".gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    return env


def validate_environment(env: dict[str, str], root: Path, repo: Path,
                         protected: Iterable[Path], config_path: Path | None = None) -> None:
    root, repo = resolved(root), resolved(repo)
    paths = _layout(root)
    for key in STATE_PATH_KEYS:
        value = env.get(key)
        if not value:
            raise IsolationError(f"required runtime state path is unset: {key}")
        path = resolved(value)
        if not is_within(path, root):
            raise IsolationError(f"{key} resolves outside isolation root: {path}")
        for target in protected:
            if overlaps(path, resolved(target)):
                raise IsolationError(f"{key} resolves into protected path {resolved(target)}")
    if resolved(env.get("PYTHONPATH", "")) != repo:
        raise IsolationError("PYTHONPATH must point only to the candidate source checkout")
    if config_path is not None:
        config = resolved(config_path)
        if not is_within(config, root):
            raise IsolationError(f"gateway config resolves outside isolation root: {config}")
        if any(overlaps(config, resolved(p)) for p in protected):
            raise IsolationError("gateway config overlaps a protected path")
        if config != resolved(paths["config"]) or not config.is_file():
            raise IsolationError("gateway config must be the isolated profile's existing config.yaml")
        for line_number, line in enumerate(config.read_text(encoding="utf-8").splitlines(), 1):
            candidate = re.match(r"^\s*[A-Za-z0-9_.-]+:\s*[\"']?([/~][^\s\"']*)", line)
            if candidate:
                referenced = resolved(candidate.group(1))
                if not is_within(referenced, root):
                    raise IsolationError(
                        f"gateway config line {line_number} contains an external path: {referenced}"
                    )
    if resolved(env["HERMES_HOME"]) != paths["profile"]:
        raise IsolationError("Hermes profile home differs from the isolated profile")
    if resolved(env["HERMES_KANBAN_DB"]) != paths["database"]:
        raise IsolationError("Kanban database differs from the isolated board")


def prepare(root_arg: Path, repo: Path, hermes_home: Path, kanban_db: Path, x_root: Path) -> Path:
    protected = _protected(repo, hermes_home, kanban_db, x_root)
    root = _validate_root(root_arg, repo, protected)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    layout = _layout(root)
    for name, path in layout.items():
        if name not in {"root", "database", "config", "venv", "kanban_home"}:
            path.mkdir(parents=True, exist_ok=True)
    _prepare_host_mask(layout["host_mask"])
    env = environment(root, repo)
    validate_environment(env, root, repo, protected, layout["config"])
    # Deliberately minimal profile config. No transport tokens, provider secrets, or profile clone.
    layout["config"].write_text(
        "gateway:\n  standalone: true\nplugins:\n  enabled:\n    - mission-admission\nkanban:\n  dispatch_in_gateway: false\n  max_in_progress: 1\n  dispatch_interval_seconds: 1\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "root": str(root),
        "repo": str(resolved(repo)),
        "protected": [str(p) for p in protected],
        "paths": {key: str(value) for key, value in layout.items()},
        "gateway_config": str(layout["config"]),
        "network": "disabled for runtime; package build may opt into shared network",
        "filesystem": "host root read-only; isolation root writable",
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def load_manifest(root_arg: Path) -> tuple[dict, Path, Path, list[Path]]:
    root = resolved(root_arg)
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise IsolationError(f"isolation manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = resolved(manifest["repo"])
    protected = [resolved(p) for p in manifest["protected"]]
    if resolved(manifest["root"]) != root:
        raise IsolationError("manifest root does not match requested root")
    env = environment(root, repo)
    validate_environment(env, root, repo, protected, Path(manifest["gateway_config"]))
    if set(manifest["paths"]) != set(_layout(root)):
        raise IsolationError("manifest does not enumerate every isolated runtime path")
    for name, value in manifest["paths"].items():
        path = resolved(value)
        if not is_within(path, root):
            raise IsolationError(f"manifest path {name} escapes isolation root: {path}")
        expected = _layout(root).get(name)
        if expected is None or path != resolved(expected):
            raise IsolationError(f"manifest path {name} differs from the fixed isolation layout")
    return manifest, root, repo, protected


def _prepare_host_mask(mask: Path) -> None:
    for name in ("hermes", "codex", "ssh", "aws", "config", "local", "cache",
                 "docker", "kube", "gnupg", "x"):
        (mask / name).mkdir(parents=True, exist_ok=True)
    for name in ("kanban.db", "kanban.db-wal", "kanban.db-shm"):
        (mask / "hermes" / name).touch(exist_ok=True)
    for name in (".netrc", ".npmrc", ".pypirc", ".git-credentials"):
        (mask / name).touch(exist_ok=True)


def bubblewrap(root: Path, env: dict[str, str], command: list[str], *, share_network: bool = False,
               read_only_inputs: Iterable[Path] = (), hide_paths: Iterable[Path] = ()) -> list[str]:
    executable = shutil.which("bwrap")
    if not executable:
        raise IsolationError("bubblewrap is required; refusing an unsandboxed run")
    args = [
        executable, "--die-with-parent", "--new-session",
        "--unshare-net", "--ro-bind", "/", "/",
        "--bind", str(root), str(root), "--bind", str(root / "tmp"), "/tmp",
        "--bind", str(root / "tmp"), "/var/tmp",
        "--proc", "/proc", "--dev", "/dev",
        "--bind", str(root / "shm"), "/dev/shm", "--chdir", str(root / "work"),
        "--clearenv",
    ]
    allowed = {resolved(path) for path in read_only_inputs}
    for source in sorted(allowed):
        if source.name != "kanban.db" or source.parent != (Path.home().resolve() / ".hermes"):
            raise IsolationError(f"unsupported read-only input: {source}")
        target = root / "migration" / ".readonly-kanban.db"
        target.parent.mkdir(parents=True, exist_ok=True)
        for source_path, target_path in (
            (source, target),
            (Path(str(source) + "-wal"), Path(str(target) + "-wal")),
            (Path(str(source) + "-shm"), Path(str(target) + "-shm")),
        ):
            if source_path.exists():
                target_path.touch(exist_ok=True)
                args.extend(("--ro-bind", str(source_path), str(target_path)))
    host_home = Path.home().resolve()
    masked = {
        ".hermes": "hermes", ".codex": "codex", ".ssh": "ssh", ".aws": "aws",
        ".config": "config", ".local": "local", ".cache": "cache",
        ".docker": "docker", ".kube": "kube", ".gnupg": "gnupg",
    }
    for suffix, name in masked.items():
        host_path = host_home / suffix
        if host_path.exists():
            args.extend(("--ro-bind", str(root / "host-mask" / name), str(host_path)))
    for host_path in {resolved(path) for path in hide_paths}:
        args.extend(("--ro-bind", str(root / "host-mask" / "x"), str(host_path)))
    for name in (".netrc", ".npmrc", ".pypirc", ".git-credentials"):
        host_path = host_home / name
        if host_path.exists():
            args.extend(("--ro-bind", str(root / "host-mask" / name), str(host_path)))
    if share_network:
        args.remove("--unshare-net")
    for key, value in sorted(env.items()):
        args.extend(("--setenv", key, value))
    args.extend(("--", *command))
    return args


def validate_command_paths(command: list[str], root: Path, repo: Path,
                           protected: Iterable[Path], read_only_inputs: Iterable[Path]) -> None:
    permitted_inputs = {resolved(path) for path in read_only_inputs}
    protected = [resolved(path) for path in protected]
    for index, value in enumerate(command):
        if not value.startswith("/"):
            continue
        path = resolved(value)
        if is_within(path, repo):
            continue
        if any(overlaps(path, item) for item in protected):
            if path not in permitted_inputs:
                raise IsolationError(f"command references protected path without read-only input grant: {path}")
            if path != protected[2]:
                raise IsolationError(f"only the exact protected Kanban DB may be a read-only input: {path}")
            continue
        if is_within(path, root) or is_within(path, repo):
            continue
        # The executable itself may be a read-only system binary; ordinary arguments may not name
        # arbitrary host state. Other external paths stay denied.
        if index == 0 and is_within(path, Path("/usr")):
            continue
        raise IsolationError(f"command contains an outside path argument: {path}")


def _sandbox_self_test(root: Path, protected: list[Path]) -> None:
    env = environment(root, Path(__file__).resolve().parents[1])
    code = r'''import errno, os, pathlib
root=pathlib.Path(os.environ["HERMES_ISOLATION_ROOT"])
(root/"self-test-writable").write_text("sandbox-ok")
for temp_root in (pathlib.Path("/tmp"), pathlib.Path("/var/tmp")):
    probe=temp_root/"phase1-scratch-proof"
    probe.write_text("sandbox-temp-ok")
    isolated_probe=root/"tmp"/probe.name
    if not isolated_probe.is_file() or os.stat(probe).st_ino != os.stat(isolated_probe).st_ino:
        raise SystemExit(f"temporary directory escaped isolation root: {probe}")
    probe.unlink()
for target in (pathlib.Path(os.environ["HERMES_PROTECTED_HERMES"]), pathlib.Path(os.environ["HERMES_PROTECTED_X"])):
    if not os.statvfs(target).f_flag & getattr(os, "ST_RDONLY", 1):
        raise SystemExit(f"protected mount is writable: {target}")
    try:
        (target/".phase1-isolation-refusal-probe").open("x").close()
    except OSError as exc:
        if exc.errno not in (errno.EROFS, errno.EACCES, errno.ENOENT): raise
    else:
        raise SystemExit(f"sandbox unexpectedly wrote under protected path: {target}")
if os.stat("/proc/self/ns/net").st_ino == int(os.environ["HERMES_HOST_NETNS_INO"]):
    raise SystemExit("runtime network namespace is not isolated")
for target in (pathlib.Path(os.environ["HERMES_PROTECTED_HERMES"]), pathlib.Path(os.environ["HERMES_PROTECTED_X"])):
    visible=list(target.iterdir())
    allowed={"kanban.db", "kanban.db-wal", "kanban.db-shm"} if target.name == ".hermes" else set()
    if any(item.name not in allowed or not item.is_file() or item.stat().st_size != 0 for item in visible):
        raise SystemExit(f"protected host contents remain visible inside sandbox: {target}")
'''
    host_netns = os.stat("/proc/self/ns/net").st_ino
    env.update({
        "HERMES_ISOLATION_ROOT": str(root),
        "HERMES_PROTECTED_HERMES": str(protected[1]),
        "HERMES_PROTECTED_X": str(protected[3]),
        "HERMES_HOST_NETNS_INO": str(host_netns),
    })
    env["HERMES_ISOLATION_ROOT"] = str(root)
    cmd = bubblewrap(root, env, [sys.executable, "-c", code], hide_paths=[protected[3]])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode:
        raise IsolationError(f"bubblewrap isolation proof failed ({result.returncode}): {result.stderr.strip()}")
    if (root / "self-test-writable").read_text(encoding="utf-8") != "sandbox-ok":
        raise IsolationError("sandbox did not write to its isolated root")
    (root / "self-test-writable").unlink()


def self_test(repo: Path, hermes_home: Path, kanban_db: Path, x_root: Path) -> None:
    repo = resolved(repo)
    protected = _protected(repo, hermes_home, kanban_db, x_root)
    with tempfile.TemporaryDirectory(prefix="hermes-phase1-isolation-", dir="/home/ftr3d") as temp:
        root = Path(temp) / "sandbox"
        root.mkdir()
        layout = _layout(root)
        for name, path in layout.items():
            if name not in {"root", "database", "config", "venv", "kanban_home"}:
                path.mkdir(parents=True, exist_ok=True)
        _prepare_host_mask(layout["host_mask"])
        layout["config"].write_text(
            "gateway:\n  standalone: true\nkanban:\n  dispatch_in_gateway: false\n",
            encoding="utf-8",
        )
        base = environment(root, repo)
        validate_environment(base, root, repo, protected, layout["config"])
        for key in STATE_PATH_KEYS:
            attempt = dict(base)
            attempt[key] = str(kanban_db if key == "HERMES_KANBAN_DB" else hermes_home)
            try:
                validate_environment(attempt, root, repo, protected, layout["config"])
            except IsolationError:
                print(f"REFUSED unsafe runtime path env {key}")
            else:
                raise IsolationError(f"negative isolation check unexpectedly allowed {key}")
        attempt = dict(base)
        attempt["XDG_CONFIG_HOME"] = str(x_root)
        try:
            validate_environment(attempt, root, repo, protected, layout["config"])
        except IsolationError:
            print("REFUSED canonical X checkout as runtime path")
        else:
            raise IsolationError("negative isolation check unexpectedly allowed canonical X")
        try:
            validate_environment(base, root, repo, protected, hermes_home / "config.yaml")
        except IsolationError:
            print("REFUSED external gateway config")
        else:
            raise IsolationError("negative isolation check unexpectedly allowed external gateway config")
        escape = root / "escape"
        escape.symlink_to(hermes_home, target_is_directory=True)
        attempt = dict(base)
        attempt["HOME"] = str(escape)
        try:
            validate_environment(attempt, root, repo, protected, layout["config"])
        except IsolationError:
            print("REFUSED symlink escape to live Hermes home")
        else:
            raise IsolationError("negative isolation check unexpectedly allowed symlink escape")
        outside_root = resolved(x_root) / ".phase1-test-root-must-not-be-created"
        try:
            _validate_root(outside_root, repo, protected)
        except IsolationError:
            print("REFUSED isolation root overlapping protected/source path")
        else:
            raise IsolationError("negative isolation check unexpectedly allowed unsafe root")
        for label, command, inputs in (
            ("live default board command argument", ["/usr/bin/python3", str(kanban_db)], []),
            ("live profile command argument", ["/usr/bin/python3", str(hermes_home / "profiles" / "fred")], []),
            ("canonical X command argument", ["/usr/bin/python3", str(x_root)], []),
        ):
            try:
                validate_command_paths(command, root, repo, protected, inputs)
            except IsolationError:
                print(f"REFUSED {label}")
            else:
                raise IsolationError(f"negative isolation check unexpectedly allowed {label}")
        validate_command_paths(["/usr/bin/python3", str(kanban_db)], root, repo,
                               protected, [kanban_db])
        print("ALLOWED exact board only as explicit read-only backup input")
        _sandbox_self_test(root, protected)
        print("PROVED sandbox writes only to isolation root; host Hermes/X mounts are read-only; runtime net namespace is isolated")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protect-hermes-home", type=Path, required=True)
    parser.add_argument("--protect-kanban-db", type=Path, required=True)
    parser.add_argument("--protect-x-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("self-test")
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--share-network", action="store_true", help="only for isolated PM dependency build")
    run.add_argument("--read-only-input", type=Path, action="append", default=[])
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.action == "self-test":
            self_test(args.repo, args.protect_hermes_home, args.protect_kanban_db, args.protect_x_root)
            return 0
        if args.action == "prepare":
            root = prepare(args.root, args.repo, args.protect_hermes_home,
                           args.protect_kanban_db, args.protect_x_root)
            print(root)
            return 0
        manifest, root, repo, protected = load_manifest(args.root)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise IsolationError("a command is required after --")
        env = environment(root, repo)
        validate_environment(env, root, repo, protected, Path(manifest["gateway_config"]))
        validate_command_paths(command, root, repo, protected, args.read_only_input)
        print("isolation preflight passed; launching inside bubblewrap", file=sys.stderr)
        result = subprocess.run(bubblewrap(root, env, command, share_network=args.share_network,
                                            read_only_inputs=args.read_only_input,
                                            hide_paths=[protected[3]]), check=False)
        return result.returncode
    except (IsolationError, OSError, ValueError, KeyError) as exc:
        print(f"isolation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
