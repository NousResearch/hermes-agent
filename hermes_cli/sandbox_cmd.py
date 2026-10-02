"""``hermes sandbox run``: run code from a repo or PR you do not trust in a throwaway container.

The code is copied to a scratch directory on the host without executing anything it carries (no
checkout, no hooks, no repo-configured filters, symlinks kept as links), then run in a container
with no network, no capabilities, a read-only root, an unprivileged user, no environment from the
host and no host mount other than that scratch copy. An optional ``--setup`` step installs
dependencies with network but is otherwise under the same lock. Nothing on the command line can
loosen the lock, and the scratch copy is deleted afterwards.

This is a tool the agent is guided to use, not a security boundary (SECURITY.md §2.2).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home

# The container sees only this tree: the code at src/ and a HOME at home/ that carries what the
# setup step installed (pip --user, npm caches) over to the run step.
_MOUNT = "/sandbox"
_SRC = f"{_MOUNT}/src"
_HOME = f"{_MOUNT}/home"
# nobody:nogroup, not the host uid, so nothing the code writes is owned by the user's account.
_SANDBOX_USER = "65534:65534"
_PIDS_LIMIT = "1024"
_TIMEOUT_EXIT = 124
_PR_RE = re.compile(r"^[1-9][0-9]*$")


class SandboxError(Exception):
    """A refusal or setup failure, reported to the user as one line (exit 2)."""


def sandbox_run_argv(docker: str, image: str, scratch: str, command: list[str], *, network: bool,
                     limits: bool, cpus: float, memory_mb: int, name: str) -> list[str]:
    """The ``docker run`` argv for one step. Pure, so the lock is testable without a runtime.

    Only the setup step gets a network (the runtime's default one); every other flag is the same
    for both steps, and nothing but image, limits and name comes from the caller.
    """
    argv = [docker, "run", "--rm", "--name", name,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=1g",
            "--user", _SANDBOX_USER,
            "-v", f"{scratch}:{_MOUNT}", "-w", _SRC]
    if not network:
        argv += ["--network", "none"]
    if limits:
        argv += ["--pids-limit", _PIDS_LIMIT]
        if cpus > 0:
            argv += ["--cpus", str(cpus)]
        if memory_mb > 0:
            argv += ["--memory", f"{memory_mb}m"]
    # `env` as the entrypoint: an image's init entrypoint (s6 in the default sandbox image) cannot
    # start as an unprivileged user on a read-only root, and HOME must point at the writable copy.
    return argv + ["--entrypoint", "env", image, f"HOME={_HOME}", *command]


def _refuse_path(path: Path) -> None:
    """Refuse trees that hold the user's own secrets: home, the filesystem root, HERMES_HOME."""
    home = Path.home().resolve()
    hermes_home = get_hermes_home().resolve()
    if path == Path(path.anchor) or home.is_relative_to(path) or hermes_home.is_relative_to(path):
        raise SandboxError(f"refusing to copy {path}: it is or contains your home or HERMES_HOME")
    if path.is_relative_to(hermes_home):
        raise SandboxError(f"refusing to copy {path}: it is inside HERMES_HOME")


def _copy_ignore(directory: str, names: list[str]) -> list[str]:
    """Skip ``.git`` (its hooks and config never reach the copy) and anything that is not a
    directory, regular file or symlink: reading a FIFO or a device would block or leak host state."""
    skipped = []
    for name in names:
        mode = os.lstat(os.path.join(directory, name)).st_mode
        if name == ".git" or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            skipped.append(name)
    return skipped


def export_path(src: Path, dest: Path) -> None:
    """Copy a working tree into *dest*; symlinks are copied as links and never followed."""
    src = src.resolve()
    if not src.is_dir():
        raise SandboxError(f"--path {src} is not a directory")
    _refuse_path(src)
    shutil.copytree(src, dest, symlinks=True, ignore=_copy_ignore)


def _hardened_git_env(repo: Path) -> dict[str, str]:
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env

    env = noninteractive_repo_git_env(repo)
    if env is None:
        raise SandboxError(f"cannot neutralise the git filters configured in {repo}; refusing to export")
    return env


def _git(repo: Path, env: dict[str, str], *args: str, timeout: float = 120) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=timeout)
    if proc.returncode != 0:
        raise SandboxError(f"git {args[0]} failed: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def fetch_pr(repo: Path, remote: str, number: str) -> str:
    """Fetch ``pull/<N>/head`` into a private ref (nothing is checked out); return the ref."""
    if not _PR_RE.match(number):
        raise SandboxError(f"--pr expects a pull request number, got {number!r}")
    if remote.startswith("-"):
        raise SandboxError(f"invalid remote {remote!r}")
    ref = f"refs/hermes-sandbox/pr-{number}"
    _git(repo, _hardened_git_env(repo), "fetch", "--no-tags", "--quiet", remote,
         f"+pull/{number}/head:{ref}", timeout=300)
    return ref


def export_ref(repo: Path, ref: str, dest: Path) -> str:
    """Write the tree of *ref* into *dest* with ``git archive`` and return its commit.

    No checkout, so no hook runs; ``git archive`` does apply smudge filters, so the repo's filter
    drivers are neutralised by the hardened git env.
    """
    if ref.startswith("-"):
        raise SandboxError(f"invalid ref {ref!r}")
    env = _hardened_git_env(repo)
    sha = _git(repo, env, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    dest.mkdir(parents=True)
    proc = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", sha], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            # "tar" keeps symlinks wherever they point (they resolve inside the container) but
            # refuses members that would land outside dest.
            archive.extractall(dest, filter="tar")
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0:
        raise SandboxError(f"git archive failed: {stderr.strip()}")
    return sha


def _open_scratch(scratch: Path) -> None:
    """Let the unprivileged container user write the copy. Symlinks are left alone."""
    for root, dirs, files in os.walk(scratch):
        for name in dirs + files:
            path = os.path.join(root, name)
            if not os.path.islink(path):
                extra = 0o777 if os.path.isdir(path) else 0o666
                os.chmod(path, os.stat(path).st_mode | extra)
    os.chmod(scratch, 0o777)


def _run_step(argv: list[str], docker: str, name: str, timeout: float) -> int:
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
        print(f"hermes sandbox: step timed out after {timeout:.0f}s", file=sys.stderr)
        return _TIMEOUT_EXIT


def sandbox_settings(config: dict, image: str | None) -> tuple[str, float, int]:
    """Image and limits: ``--image`` or ``terminal.docker_image``, and the terminal's container
    limits. Nothing else is read, so docker_volumes / docker_forward_env / docker_env /
    docker_extra_args / credential_files never reach a sandbox run."""
    from hermes_cli.config_defaults import DEFAULT_SANDBOX_IMAGE

    terminal = (config or {}).get("terminal") or {}
    chosen = image or terminal.get("docker_image") or DEFAULT_SANDBOX_IMAGE
    return str(chosen), float(terminal.get("container_cpu") or 0), int(terminal.get("container_memory") or 0)


def cmd_sandbox_run(args) -> int:
    command = list(args.run_command or [])
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("hermes sandbox run: give the command to run after --", file=sys.stderr)
        return 2

    from tools.environments.docker import _cgroup_limits_available, docker_runtime_start_hint, find_docker

    docker = find_docker()
    if not docker:
        print("hermes sandbox run needs Docker or Podman; neither was found.", file=sys.stderr)
        return 2
    if subprocess.run([docker, "version"], capture_output=True, stdin=subprocess.DEVNULL,
                      timeout=15).returncode != 0:
        print(f"hermes sandbox run: the container runtime is not reachable; {docker_runtime_start_hint(docker)}.",
              file=sys.stderr)
        return 2

    from hermes_cli.config import load_config

    image, cpus, memory_mb = sandbox_settings(load_config(), args.image)
    scratch = Path(tempfile.mkdtemp(prefix="hermes-sandbox-")).resolve()
    try:
        try:
            if args.path:
                export_path(Path(args.path).expanduser(), scratch / "src")
                label = str(Path(args.path).expanduser().resolve())
            else:
                repo = Path(args.repo).expanduser().resolve()
                ref = fetch_pr(repo, args.remote, args.pr) if args.pr else args.ref
                label = export_ref(repo, ref, scratch / "src")[:12]
        except SandboxError as exc:
            print(f"hermes sandbox run: {exc}", file=sys.stderr)
            return 2
        (scratch / "home").mkdir()
        _open_scratch(scratch)
        limits = _cgroup_limits_available(image)
        print(f"hermes sandbox: {label} in {image} (no network, no credentials, read-only root)", file=sys.stderr)

        def step(step_command: list[str], network: bool) -> int:
            name = f"hermes-sandbox-{uuid.uuid4().hex[:10]}"
            argv = sandbox_run_argv(docker, image, str(scratch), step_command, network=network,
                                    limits=limits, cpus=cpus, memory_mb=memory_mb, name=name)
            return _run_step(argv, docker, name, args.timeout)

        if args.setup and (rc := step(["sh", "-c", args.setup], True)) != 0:
            print(f"hermes sandbox: --setup failed (exit {rc})", file=sys.stderr)
            return rc
        return step(command, False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def cmd_sandbox(args) -> int:
    if getattr(args, "sandbox_action", None) == "run":
        return cmd_sandbox_run(args)
    args.sandbox_parser.print_help()
    return 0
