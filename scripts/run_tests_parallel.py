#!/usr/bin/env python3
"""Per-file parallel test runner.

The minimum-viable replacement for pytest-xdist + a subprocess-isolation
plugin. Discovers test files under ``tests/`` (excluding integration/e2e
unless explicitly requested), then runs one ``python -m pytest <file>``
subprocess per file, with bounded parallelism (default: ``os.cpu_count()``).

Why per-file rather than per-test?
    Per-test spawn overhead (~250ms × 17k tests = 70min CPU minimum)
    swamped the actual work. Per-file spawn (~250ms × ~850 files = ~3.5min)
    fits in the budget while still giving every file a fresh Python
    interpreter — the only isolation boundary that actually matters
    (cross-file module-level state leakage was the original flake source;
    intra-file state is the test author's responsibility).

Why drop xdist entirely?
    xdist's persistent workers accumulate state across files, which is
    exactly the leakage we wanted to fix. xdist also adds complexity
    (loadfile vs loadscope, --max-worker-restart, internal control plane)
    that we don't need when the unit of work is "run pytest on one file".
    A subprocess.Popen pool gated by a semaphore is ~60 lines and does
    the job.

Usage:
    python scripts/run_tests_parallel.py [pytest_args...]

    Common pytest args pass through to each per-file pytest invocation
    (e.g. ``-q``, ``-v``, ``-x``, ``--tb=long``, ``-k 'pattern'``, ``--lf``)
    with no special separator — a bare ``-q`` "just works". Anything after
    a literal ``--`` is also passed through, and stacks with bare flags.

Environment:
    HERMES_TEST_WORKERS  Override worker count (default: os.cpu_count())
    HERMES_TEST_PATHS    Override discovery roots (colon-sep; on Windows
                         ';' also works and drive letters are handled;
                         default: 'tests')

Exit code: 0 if every file's pytest exited 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Tuple


# Default test discovery roots.
_DEFAULT_ROOTS = ["tests"]

# Directories to skip during discovery — these suites require real
# external services (a model gateway, a docker daemon with a prebuilt
# image, etc.) and are run in their own dedicated CI jobs:
#
#   tests/e2e/         — .github/workflows/tests.yml :: e2e job
#   tests/integration/ — historical; legacy --ignore flags
#   tests/docker/      — .github/workflows/docker.yml ::
#                        build-amd64 job (runs against the freshly-loaded
#                        nousresearch/hermes-agent:test image, via
#                        ``HERMES_TEST_IMAGE`` so the fixture skips
#                        rebuild). The full pytest-shard runner can't
#                        host these because the session-scoped
#                        ``built_image`` fixture would do a 3-7min
#                        ``docker build``,
#                        so the build is guaranteed to die in fixture
#                        setup. The dedicated job sidesteps both costs.
_SKIP_PARTS = {"integration", "e2e", "docker"}

# Per-file wall-clock cap. Override
# via --file-timeout or HERMES_TEST_FILE_TIMEOUT.
#
# Set to 300s (5 min) deliberately generous: the per-test subprocess
# isolation plugin spawns a fresh Python process per test, so a
# large-collection file pays N × (interpreter startup + import) of
# overhead before any test logic runs — and that overhead dilates under
# load on shared CI runners, producing false "no tests ran" timeouts on
# files that finish in ~100s on a quiet box. The Docker build matrix jobs
# take 7-10 min anyway, so this headroom costs nothing on total CI wall
# time while keeping a genuinely hung file bounded.
_DEFAULT_FILE_TIMEOUT_SECONDS = 300.0

# One-shot retry of failing test FILES. A file that exits non-zero is re-run
# once in a fresh subprocess; if the re-run passes, the file counts as passed
# but is loudly reported as FLAKY so it gets fixed rather than hidden.
# Deterministic failures fail both attempts — a real regression can never be
# laundered into green by this (it would have to flake in our favor twice in
# a row on the same runner, which is exactly the definition of a flake).
# Set to 0 to disable (env: HERMES_TEST_FILE_RETRIES).
_DEFAULT_FILE_RETRIES = 1

# Duration cache: maps relative file paths to last-observed subprocess
# wall-clock seconds. Used by ``--slice`` to distribute files across
# CI jobs by estimated total time, so no one job gets all the slow files.
_DURATIONS_FILE = "test_durations.json"


def _sandbox_profile_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_real_home(environment: dict[str, str]) -> Path:
    """Resolve the account home independently and reject spoofed declarations."""
    if os.name != "posix":
        raise RuntimeError("account-home attestation is implemented only for POSIX")
    import pwd

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()  # windows-footgun: ok — POSIX-only branch above
    declared_real_raw = environment.get("HERMES_TEST_REAL_HOME", "").strip()
    if declared_real_raw:
        declared_real = Path(declared_real_raw).expanduser().resolve()
        if declared_real != account_home:
            raise RuntimeError("HERMES_TEST_REAL_HOME does not match the OS account home")
        return account_home
    declared_home = Path(environment.get("HOME", "")).expanduser().resolve()
    if declared_home != account_home:
        raise RuntimeError("HOME does not match the OS account home; refusing tests")
    return account_home


def _attest_macos_mach_broker() -> str:
    """Return a broker proven registered before entering the test sandbox."""
    import ctypes

    libc = ctypes.CDLL(None)
    bootstrap_port = ctypes.c_uint32.in_dll(libc, "bootstrap_port").value
    task_self = ctypes.c_uint32.in_dll(libc, "mach_task_self_").value
    libc.bootstrap_look_up.argtypes = [
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    libc.bootstrap_look_up.restype = ctypes.c_int
    libc.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    libc.mach_port_deallocate.restype = ctypes.c_int
    for service in (
        "com.apple.SecurityServer",
        "com.apple.securityd.xpc",
        "com.apple.security.agent",
    ):
        service_port = ctypes.c_uint32()
        status = libc.bootstrap_look_up(
            bootstrap_port,
            service.encode("utf-8"),
            ctypes.byref(service_port),
        )
        if status == 0 and service_port.value:
            libc.mach_port_deallocate(task_self, service_port.value)
            return service
    raise RuntimeError(
        "Hermes macOS tests require a registered credential broker to attest "
        "Seatbelt mach-lookup denial; no baseline broker was reachable"
    )


def _sensitive_host_paths(
    real_home: Path, repo_root: Path
) -> tuple[list[Path], list[Path]]:
    """Return credential/runtime directories and files hidden from tests."""
    directories = [
        real_home / ".hermes",
        real_home / ".credentials",
        real_home / ".ssh",
        real_home / ".aws",
        real_home / ".gnupg",
        real_home / ".claude",
        real_home / ".codex",
        real_home / ".copilot",
        real_home / ".minimax",
        real_home / ".config" / "github-copilot",
        real_home / ".config" / "gh",
        real_home / ".config" / "gcloud",
        real_home / ".config" / "op",
        real_home / ".config" / "openai",
        real_home / ".config" / "anthropic",
        real_home / ".config" / "huggingface",
        real_home / ".cache" / "huggingface",
        real_home / ".azure",
        real_home / ".kube",
        real_home / ".docker",
        real_home / ".local" / "share" / "keyrings",
        real_home / "Library" / "Keychains",
    ]
    files = [
        real_home / ".netrc",
        real_home / ".gitconfig",
        real_home / ".git-credentials",
        real_home / ".npmrc",
        real_home / ".pypirc",
        real_home / ".anthropic_oauth.json",
        real_home / ".cargo" / "credentials",
        real_home / ".cargo" / "credentials.toml",
        real_home / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist",
    ]
    for pattern in (
        ".env*",
        ".npmrc",
        ".pypirc",
        "apps/*/.env*",
        "web/.env*",
        "ui-tui/.env*",
    ):
        files.extend(path for path in repo_root.glob(pattern) if path.is_file())
    return directories, files


def _sandboxed_test_command(
    command: list[str],
    *,
    env: dict[str, str],
    repo_root: Path,
    sandbox_root: Path,
    real_home: Path,
    parent_sandbox_root: Path | None = None,
    ephemeral_docker_socket: Path | None = None,
    writable_repo: bool = False,
    working_directory: Path | None = None,
    additional_writable_paths: tuple[Path, ...] = (),
    additional_readonly_paths: tuple[Path, ...] = (),
) -> list[str]:
    """Wrap one pytest file in the host's fail-closed OS sandbox."""
    directories, files = _sensitive_host_paths(real_home, repo_root)
    if sys.platform.startswith("linux"):
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError(
                "Hermes tests require bubblewrap on Linux; refusing an "
                "unsandboxed test process"
            )
        env["HERMES_TEST_OS_SANDBOX"] = "linux-bwrap"
        live_hermes = real_home / ".hermes"
        if repo_root == live_hermes or repo_root.is_relative_to(live_hermes):
            raise RuntimeError(
                "refusing to run tests from the live ~/.hermes tree; use an "
                "isolated source worktree with its own test venv"
            )
        if Path(sys.executable).absolute().is_relative_to(live_hermes):
            raise RuntimeError(
                "refusing the live ~/.hermes interpreter for tests; create "
                "a test-only virtualenv in the isolated source worktree"
            )
        wrapped = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--ro-bind", "/", "/",
            "--tmpfs", "/run",
            "--tmpfs", "/tmp",
            "--tmpfs", "/var/tmp",
            "--dev", "/dev",
            "--proc", "/proc",
        ]
        # Hide the operator's *entire* home before restoring only the source
        # tree and interpreter needed by this test.  A credential inventory
        # is necessarily incomplete: providers may add another dotfile at
        # any time, and native code must not gain it merely because the
        # Python guard does not know its name yet.
        home_is_masked = real_home.is_dir()
        masked_roots = tuple(
            root
            for root in (real_home, Path("/tmp"), Path("/run"), Path("/var/tmp"))
            if root.is_dir()
        )
        if home_is_masked:
            wrapped.extend(("--tmpfs", str(real_home)))
        restored = tuple(
            required
            for required in (
                repo_root,
                Path(sys.prefix).resolve(),
                Path(sys.base_prefix).resolve(),
                Path(sys._base_executable).absolute().parent.parent,
            )
            if required.is_dir()
        )
        bind_targets = [
            *restored,
            *additional_writable_paths,
            *additional_readonly_paths,
        ]
        if parent_sandbox_root is not None:
            bind_targets.append(parent_sandbox_root)
        bind_targets.append(sandbox_root)
        if ephemeral_docker_socket is not None:
            bind_targets.append(ephemeral_docker_socket.parent)
        parents: set[Path] = set()
        for required in bind_targets:
            enclosing = [root for root in masked_roots if _under(required, root)]
            if not enclosing:
                continue
            mount_root = max(enclosing, key=lambda item: len(item.parts))
            if required == mount_root:
                raise RuntimeError("refusing to restore an entire masked host root")
            parent = required.parent
            while parent != mount_root:
                parents.add(parent)
                parent = parent.parent
        for parent in sorted(parents, key=lambda item: len(item.parts)):
            wrapped.extend(("--dir", str(parent)))
        for required in restored:
            wrapped.extend(("--ro-bind", str(required), str(required)))
        for path in additional_readonly_paths:
            if path.exists():
                wrapped.extend(("--ro-bind", str(path), str(path)))
        for path in additional_writable_paths:
            if path.is_dir():
                wrapped.extend(("--bind", str(path), str(path)))
        if writable_repo:
            wrapped.extend(("--bind", str(repo_root), str(repo_root)))
        masked_directories: list[Path] = [real_home] if home_is_masked else []
        masked_files: list[Path] = []
        for path in directories:
            if path.is_dir() and not home_is_masked:
                wrapped.extend(("--tmpfs", str(path)))
                masked_directories.append(path)
        for path in files:
            if path.is_file() and (
                not home_is_masked or _under(path.resolve(), repo_root.resolve())
            ):
                wrapped.extend(("--ro-bind", "/dev/null", str(path)))
                masked_files.append(path)
        env["HERMES_TEST_MASKED_DIRECTORIES"] = os.pathsep.join(
            str(path) for path in masked_directories
        )
        env["HERMES_TEST_MASKED_FILES"] = os.pathsep.join(
            str(path) for path in masked_files
        )
        if parent_sandbox_root is not None and parent_sandbox_root.is_dir():
            wrapped.extend(
                ("--bind", str(parent_sandbox_root), str(parent_sandbox_root))
            )
        wrapped.extend(("--bind", str(sandbox_root), str(sandbox_root)))
        if ephemeral_docker_socket is not None:
            wrapped.extend(
                (
                    "--bind",
                    str(ephemeral_docker_socket.parent),
                    str(ephemeral_docker_socket.parent),
                )
            )
            env["DOCKER_HOST"] = f"unix://{ephemeral_docker_socket}"
            env["HERMES_TEST_EPHEMERAL_DOCKER"] = "1"
            env["HERMES_TEST_DOCKER_SHIM"] = str(
                repo_root / "scripts" / "hermetic_site" / "docker"
            )
            env["PATH"] = str(repo_root / "scripts" / "hermetic_site") + (
                os.pathsep + env["PATH"] if env.get("PATH") else ""
            )
        wrapped.extend(("--chdir", str(working_directory or repo_root)))
        return [*wrapped, *command]

    if sys.platform == "darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise RuntimeError(
                "Hermes tests require sandbox-exec on macOS; refusing an "
                "unsandboxed test process"
            )
        env["HERMES_TEST_OS_SANDBOX"] = "macos-sandbox-exec"
        env["HERMES_TEST_ATTESTED_MACH_BROKER"] = _attest_macos_mach_broker()
        # Seatbelt intentionally permits a sandboxed process to signal
        # itself, so self-targeted kill(2) cannot attest this boundary. The
        # runner PID is outside the sandbox and remains alive through child
        # collection; a signal-0 probe against it must fail with EPERM.
        env["HERMES_TEST_SANDBOX_PARENT_PID"] = str(os.getpid())
        rules = [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            # macOS has no Linux-style PID namespace. Denying the signal
            # syscall at Seatbelt level prevents ctypes/native-code bypasses
            # from reaching any host process, including the live gateway.
            # Restore signals only within this exact sandbox instance so
            # tests can reap children they created; the Python lineage guard
            # remains a second check before those syscalls.
            "(deny signal)",
            "(allow signal (target self))",
            "(allow signal (target same-sandbox))",
            # Credential and service APIs can bypass command/path guards by
            # calling securityd or launchd over Mach/XPC directly. No Hermes
            # test needs a host broker, so deny lookup broadly and fail closed.
            "(deny mach-lookup)",
        ]
        readable_home_paths = {
            repo_root.resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(sys._base_executable).absolute().parent.parent,
        }
        home_filter_parts = [
            f'(subpath "{_sandbox_profile_escape(str(real_home))}")'
        ]
        for path in sorted(readable_home_paths, key=str):
            if _under(path, real_home):
                escaped = _sandbox_profile_escape(str(path))
                home_filter_parts.append(
                    f'(require-not (subpath "{escaped}"))'
                )
        rules.append(
            "(deny file-read* (require-all "
            + " ".join(home_filter_parts)
            + "))"
        )
        writable_home_parts = [
            f'(subpath "{_sandbox_profile_escape(str(real_home))}")'
        ]
        if writable_repo and _under(repo_root.resolve(), real_home):
            writable_home_parts.append(
                '(require-not (subpath "'
                + _sandbox_profile_escape(str(repo_root.resolve()))
                + '"))'
            )
        rules.append(
            "(deny file-write* (require-all "
            + " ".join(writable_home_parts)
            + "))"
        )
        for path in directories:
            if path.exists():
                escaped = _sandbox_profile_escape(str(path))
                rules.append(f'(deny file-read* (subpath "{escaped}"))')
                rules.append(f'(deny file-write* (subpath "{escaped}"))')
        for path in files:
            if path.exists():
                escaped = _sandbox_profile_escape(str(path))
                # Pytest stats repo-root entries while resolving collection.
                # Deny file contents, not metadata, so an intentionally masked
                # .env file cannot abort collection merely by existing.
                rules.append(f'(deny file-read-data (literal "{escaped}"))')
                rules.append(f'(deny file-write* (literal "{escaped}"))')
        live_roots = (
            real_home / ".hermes",
            real_home / "Library" / "LaunchAgents",
        )
        for path in live_roots:
            if path.exists():
                escaped = _sandbox_profile_escape(str(path))
                rules.append(f'(deny file-write* (subpath "{escaped}"))')
        for executable in (
            "/bin/launchctl", "/usr/bin/security", "/usr/bin/killall",
            "/usr/bin/pkill", "/bin/kill", "/usr/bin/ssh",
            "/usr/bin/curl", "/usr/bin/nc",
        ):
            if Path(executable).exists():
                escaped = _sandbox_profile_escape(executable)
                rules.append(f'(deny process-exec (literal "{escaped}"))')
                rules.append(f'(deny file-read-data (literal "{escaped}"))')
        profile = sandbox_root / "hermes-test.sb"
        profile.write_text("\n".join(rules) + "\n", encoding="utf-8")
        return [sandbox_exec, "-f", str(profile), *command]

    raise RuntimeError(
        f"Hermes has no fail-closed OS test sandbox for {sys.platform!r}; "
        "refusing to run tests on this host"
    )


def _attest_ephemeral_docker(
    socket_path: Path,
    container_id: str,
    expected_image: str,
    real_docker: Path,
    shared_root: Path,
) -> None:
    """Prove a socket belongs to a rootless daemon in an isolated container."""
    if not sys.platform.startswith("linux") or not socket_path.is_socket():
        raise RuntimeError("ephemeral Docker requires a live Linux Unix socket")
    if socket_path.parent.parent != shared_root:
        raise RuntimeError("Docker socket is outside the disposable shared root")
    if not re.fullmatch(r"[a-f0-9]{64}", container_id):
        raise RuntimeError("invalid disposable Docker outer-container identity")
    if not real_docker.is_absolute() or not real_docker.is_file():
        raise RuntimeError("invalid host Docker client for boundary attestation")
    try:
        inspection = subprocess.run(
            [str(real_docker), "inspect", container_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if inspection.returncode != 0:
            raise RuntimeError("cannot inspect disposable Docker outer container")
        record = json.loads(inspection.stdout)[0]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError("invalid Docker outer-container attestation") from exc
    host = record.get("HostConfig", {})
    config = record.get("Config", {})
    mounts = record.get("Mounts", [])
    if not record.get("State", {}).get("Running"):
        raise RuntimeError("disposable Docker outer container is not running")
    if config.get("Image") != expected_image or config.get("User") != "rootless":
        raise RuntimeError("Docker-in-Docker image/user is not the rootless contract")
    if (
        host.get("NetworkMode") != "none"
        or host.get("PidMode") == "host"
        or host.get("IpcMode") == "host"
        or host.get("UTSMode") == "host"
    ):
        raise RuntimeError("Docker-in-Docker shares a host namespace")
    bind_mounts = [mount for mount in mounts if mount.get("Type") == "bind"]
    unexpected_mounts = [
        mount
        for mount in mounts
        if mount.get("Type") not in {"bind", "tmpfs"}
        or (
            mount.get("Type") == "tmpfs"
            and mount.get("Destination") != "/var/lib/docker"
        )
    ]
    if len(bind_mounts) != 1 or unexpected_mounts:
        raise RuntimeError("Docker-in-Docker has unexpected host mounts")
    mount = bind_mounts[0]
    if not (
        mount.get("Type") == "bind"
        and Path(mount.get("Source", "")).resolve() == shared_root
        and mount.get("Destination") == str(shared_root)
        and mount.get("RW") is True
    ):
        raise RuntimeError("Docker-in-Docker exposes a host path outside test storage")
    daemon_info = subprocess.run(
        [
            str(real_docker),
            "--host",
            f"unix://{socket_path}",
            "info",
            "--format",
            "{{json .SecurityOptions}}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if daemon_info.returncode != 0 or "name=rootless" not in daemon_info.stdout:
        raise RuntimeError("inner Docker daemon is not running rootless")


def _split_pathspec(value: str) -> List[str]:
    """Split a separator-joined path list (``--paths``/``--files``/
    ``HERMES_TEST_PATHS``) into individual paths.

    POSIX: ``:``-separated, as documented.

    Windows: ``;`` (``os.pathsep``) and ``:`` are both accepted as
    separators, but a ``:`` that forms a drive letter (``C:\\...`` or
    ``C:/...``) stays glued to its path — a naive ``split(":")`` turns
    ``C:\\repo\\tests`` into ``['C', '\\repo\\tests']``, where the bogus
    ``C`` becomes a phantom discovery root and the rooted remainder only
    resolves by accident of ``Path.__truediv__`` re-anchoring it onto
    ``repo_root``'s drive.
    """
    if sys.platform != "win32":
        return [p for p in value.split(":") if p.strip()]
    parts: List[str] = []
    for chunk in value.split(";"):
        raw = chunk.split(":")
        i = 0
        while i < len(raw):
            part = raw[i]
            if (
                len(part) == 1
                and part.isalpha()
                and i + 1 < len(raw)
                and raw[i + 1][:1] in ("\\", "/")
            ):
                part = f"{part}:{raw[i + 1]}"
                i += 1
            parts.append(part)
            i += 1
    return [p for p in parts if p.strip()]

# Host-OS gating (see the ``_OS_MARKS`` block in tests/conftest.py): tests
# marked for another host are collected and SKIPPED by the conftest hook —
# this runner never executes them, by construction. The summary calls that
# out explicitly so a local run isn't misread as covering macOS/Windows
# behaviour, and names the CI lane where those tests actually execute.
_OS_MARKERS = {
    "linux_only": ("linux", "the main Linux CI lane"),
    "macos_only": ("darwin", "the tests-os CI lane (macos-latest)"),
    "windows_only": ("win32", "the tests-os CI lane (windows-latest)"),
}


def _off_host_marker_files(files: List[Path]) -> dict[str, int]:
    """Count discovered files referencing each marker for an OS we are not on.

    Whole-word text match, same approach as scripts/ci/list_os_marked_tests.py:
    over-counting a prose mention is harmless here (the note is informational);
    what matters is never reporting 0 while gated tests exist.
    """
    off_host = {
        marker: re.compile(rf"\b{marker}\b")
        for marker, (host_prefix, _) in _OS_MARKERS.items()
        if not sys.platform.startswith(host_prefix)
    }
    counts = {marker: 0 for marker in off_host}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker, pattern in off_host.items():
            if pattern.search(text):
                counts[marker] += 1
    return {marker: n for marker, n in counts.items() if n}


def _approximately_count_tests(
    files: List[Path], repo_root: Path
) -> dict[Path, int]:
    """
    Make a decent estimate at individual tests per file.
    Running ``pytest --co -q`` is WAY too slow because it actually imports everything.

    Returns a mapping ``{file_path: test_count}``. Files with zero
    collected tests are omitted from the dict (not an error — e.g. the
    file only defines fixtures / conftest helpers).

    """

    results = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            contents = f.read()
        results[path] = contents.count("def test_")

    return results


def _discover_files(roots: List[Path]) -> List[Path]:
    """Return every ``test_*.py`` under the given roots (sorted).

    Roots may be directories (recursed for ``test_*.py``) or explicit
    ``.py`` files (included as-is, even if they don't match the
    ``test_*`` prefix — caller knows what they want).

    Exclude any file whose path contains a component in ``_SKIP_PARTS``,
    UNLESS the user explicitly named it as a root (in which case the
    user's intent overrides the skip filter). This makes
    ``scripts/run_tests.sh tests/docker/`` work locally the same way
    ``pytest tests/docker/`` does — the CI-level skip exists to keep
    the sharded matrix from blowing up, not to block targeted runs.
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            # Explicit file: include it as-is, skip the _SKIP_PARTS filter
            # since the user named it directly.
            real = root.resolve()
            if real not in seen:
                seen.add(real)
                out.append(root)
            continue
        # If the explicit root itself sits inside a skipped dir (e.g.
        # the user said ``tests/docker``), the user has overridden the
        # skip for that subtree. Compute the set of skip-parts the user
        # opted into, and only filter files whose path crosses a
        # skip-part *outside* that opt-in.
        root_skip_overrides = {
            part for part in root.parts if part in _SKIP_PARTS
        }
        effective_skips = _SKIP_PARTS - root_skip_overrides
        for path in root.rglob("test_*.py"):
            if any(part in effective_skips for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


def _kill_tree(proc: "subprocess.Popen", pgid: int | None = None) -> None:
    """Kill the pytest subprocess and every descendant it spawned.

    A test run can spin up uvicorn servers, async runtimes, or other
    long-running grandchildren that survive the pytest subprocess exit
    if we don't kill the whole tree. ``subprocess.Popen.kill()`` only
    targets the immediate child; grandchildren reparent to PID 1
    (Linux) / get adopted by services.exe (Windows) and leak.

    POSIX: the caller must pass ``pgid`` — the process group id captured
    immediately after Popen (via ``os.getpgid(proc.pid)``). We can't
    look it up here in the happy path because by the time we get
    called the leader process has already been reaped and its pid is
    gone from the kernel's process table, even though descendants in
    the group are still alive. SIGKILL'ing the captured pgid takes out
    everything in that group atomically.

    Windows: ``taskkill /F /T /PID`` walks the recorded ppid chain and
    terminates the whole tree, even when the root has already exited.

    Why not psutil: psutil walks the parent-child tree, but in the
    happy path the root has already been reaped so ``psutil.Process(pid)``
    can't find it; grandchildren reparented to PID 1 are also
    unreachable by tree walk at that point. The platform-native
    primitives (process groups / taskkill) handle both cases correctly
    without an extra abstraction layer.
    """
    if proc.pid is None:
        return

    if sys.platform == "win32":
        try:
            
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )  # windows-footgun: ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        # POSIX: kill the captured pgid. Local-import signal so the
        # SIGKILL attribute is never referenced on Windows.
        if pgid is not None:
            try:
                import signal as _signal
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Belt-and-suspenders: ensure subprocess.communicate() sees the exit.
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_one_file(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
    retries: int = 0,
) -> Tuple[Path, int, str, dict[str, int], float]:
    """Run ``python -m pytest <file> <pytest_args>`` in a fresh subprocess.

    Returns (file, returncode, captured_combined_output, summary_counts, subprocess_wall_seconds).

    ``retries`` > 0 enables the one-shot flake retry: a non-zero exit is
    re-run in a fresh subprocess; if the re-run passes, the file counts as
    passed but the output is prefixed with a FLAKY banner and the file/output
    are recorded in ``_FLAKY_RESULTS`` so the summary can call it out. A
    deterministic failure fails every attempt, so real regressions cannot
    be laundered green.

    ``summary_counts`` is the result of ``_parse_pytest_summary(output)`` —

    pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html):
        0 = all tests passed
        1 = some tests failed
        2 = test execution interrupted
        3 = internal error
        4 = pytest CLI usage error
        5 = no tests collected

    We treat exit 5 as a pass: it just means every test in the file was
    skipped or filtered by a marker (e.g. ``-m 'not integration'`` skips
    files where every test is marked integration). That's intentional and
    not a failure mode.

    On per-file timeout (``file_timeout`` seconds) or any other exception
    during ``communicate()``, we kill the whole process group / process
    tree so grandchildren (uvicorn servers, async runtimes, etc.) do not
    orphan onto PID 1. This outer timeout exists only to
    bound a pathologically slow or hung file as a whole.
    """
    file, rc, output, summary, subproc_wall = _run_one_file_once(
        file, pytest_args, repo_root, file_timeout
    )
    attempt = 0
    while rc != 0 and attempt < retries:
        attempt += 1
        first_output = output
        file, rc, output, summary, subproc_wall2 = _run_one_file_once(
            file, pytest_args, repo_root, file_timeout
        )
        subproc_wall += subproc_wall2
        if rc == 0:
            output = (
                f"⚠ FLAKY: failed on attempt 1, passed on retry "
                f"(attempt {attempt + 1}). Fix the flake — do not ignore this.\n"
                f"--- first-attempt output ---\n{first_output}\n"
                f"--- retry output ---\n{output}"
            )
            with _flaky_lock:
                _FLAKY_RESULTS.append((file, output))
    return file, rc, output, summary, subproc_wall


# Files that failed once and passed on retry, with both attempts' output.
# Keeping the traceback is load-bearing: a self-healed flake without its
# failing assertion is only a filename, which forces another expensive full
# run to rediscover the race.
_FLAKY_RESULTS: List[Tuple[Path, str]] = []
_flaky_lock = threading.Lock()


def _run_one_file_once(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
) -> Tuple[Path, int, str, dict[str, int], float]:
    """Single attempt of a per-file pytest subprocess (see _run_one_file)."""
    cmd = [sys.executable, "-m", "pytest", str(file), *pytest_args]

    # Give this subprocess its own pytest temp root.
    #
    # pytest builds its tmp_path root as <temproot>/pytest-of-<user>/. At the
    # end of a session it walks that directory with cleanup_dead_symlinks().
    # The walk lists the directory. Then it asks whether the `pytest-current`
    # symlink resolves. Then it unlinks the symlink.
    #
    # Every file shared one root. A second process replaced that symlink
    # between the question and the unlink. The first process then died with
    # FileNotFoundError after all of its tests passed.
    #
    # The risk grows with the number of processes that finish together. At 8
    # workers it never occurred. At 144 workers it occurs.
    #
    # One root for each subprocess removes the shared directory that the race
    # needs. The parent deletes the root after the attempt.
    env = os.environ.copy()
    docker_socket_raw = env.pop("HERMES_TEST_EPHEMERAL_DOCKER_SOCKET", "").strip()
    dind_id = env.pop("HERMES_TEST_DIND_CONTAINER_ID", "").strip()
    dind_image = env.pop("HERMES_TEST_DIND_IMAGE", "").strip()
    dind_shared_root_raw = env.pop("HERMES_TEST_DIND_SHARED_ROOT", "").strip()
    real_docker_raw = env.get("HERMES_TEST_REAL_DOCKER", "").strip()
    ephemeral_docker_socket: Path | None = None
    dind_shared_root: Path | None = None
    if (
        docker_socket_raw
        or dind_id
        or dind_image
        or dind_shared_root_raw
        or real_docker_raw
    ):
        candidate = Path(docker_socket_raw).resolve()
        dind_shared_root = Path(dind_shared_root_raw).resolve()
        _attest_ephemeral_docker(
            candidate,
            dind_id,
            dind_image,
            Path(real_docker_raw).resolve(),
            dind_shared_root,
        )
        ephemeral_docker_socket = candidate
        env["HERMES_TEST_DIND_SHARED_ROOT"] = str(dind_shared_root)
    inherited_sandbox_root = env.get("HERMES_TEST_SANDBOX_ROOT", "").strip()
    parent_sandbox_root = (
        Path(inherited_sandbox_root).resolve()
        if inherited_sandbox_root
        else None
    )
    pytest_root = dind_shared_root / "pytest-roots" if dind_shared_root else None
    if pytest_root is not None and not pytest_root.is_dir():
        raise RuntimeError("disposable Docker pytest root does not exist")
    temproot = tempfile.mkdtemp(prefix="hermes-pytest-tmproot-", dir=pytest_root)
    if dind_shared_root is not None:
        # The rootless daemon maps its container root to a distinct host uid.
        # This disposable directory is the only shared filesystem capability;
        # world access here lets Docker bind-mount test fixtures without
        # granting access to any operator or repository path.
        Path(temproot).chmod(0o777)
    sandbox_root = Path(temproot)
    test_home = sandbox_root / "home"
    test_hermes_home = test_home / ".hermes"
    test_home.mkdir()
    test_hermes_home.mkdir()
    real_home = _resolve_real_home(env)
    env["PYTEST_DEBUG_TEMPROOT"] = temproot
    cmd.extend(("-o", f"cache_dir={temproot}/pytest-cache"))
    env["TMPDIR"] = temproot
    env["TEMP"] = temproot
    env["TMP"] = temproot
    env["HOME"] = str(test_home)
    env["USERPROFILE"] = str(test_home)
    env["HOMEDRIVE"] = test_home.drive
    env["HOMEPATH"] = str(test_home)
    env["XDG_CONFIG_HOME"] = str(test_home / ".config")
    env["XDG_CACHE_HOME"] = str(test_home / ".cache")
    env["XDG_DATA_HOME"] = str(test_home / ".local" / "share")
    env["XDG_STATE_HOME"] = str(test_home / ".local" / "state")
    env["HERMES_HOME"] = str(test_hermes_home)
    env["HERMES_TEST_ISOLATION"] = str(test_hermes_home)
    env["HERMES_TEST_SANDBOX_ROOT"] = str(sandbox_root)
    env["HERMES_TEST_GUARD_ACTIVE"] = "1"
    env["HERMES_TEST_REAL_HOME"] = str(real_home)
    env["HERMES_TEST_REPO_ROOT"] = str(repo_root.resolve())
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("HERMES_TEST_GUARD_ROOT_PID", None)
    host_canary_path: Path | None = None
    host_socket_path: Path | None = None
    host_socket: socket.socket | None = None
    if file.name == "test_hermetic_environment_boundary.py" and sys.platform.startswith(
        "linux"
    ):
        canary_fd, canary_name = tempfile.mkstemp(prefix="hermes-host-canary-")
        os.write(canary_fd, b"HERMES_HOST_CANARY")
        os.close(canary_fd)
        host_canary_path = Path(canary_name)
        host_socket_path = host_canary_path.with_suffix(".sock")
        host_socket = socket.socket(socket.AF_UNIX)
        host_socket.bind(str(host_socket_path))
        host_socket.listen(1)
        env["HERMES_TEST_HOST_CANARY"] = str(host_canary_path)
        env["HERMES_TEST_HOST_SOCKET"] = str(host_socket_path)
    guard_site = repo_root / "scripts" / "hermetic_site"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(guard_site) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    cmd = _sandboxed_test_command(
        cmd,
        env=env,
        repo_root=repo_root,
        sandbox_root=sandbox_root,
        real_home=real_home,
        parent_sandbox_root=parent_sandbox_root,
        ephemeral_docker_socket=ephemeral_docker_socket,
    )

    subproc_start = time.monotonic()
    # launch the pytest process
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        # POSIX: place the child at the head of its own process group so
        # _kill_tree can SIGKILL the group atomically.
        # Windows: this maps to CREATE_NEW_PROCESS_GROUP in CPython 3.12+;
        # _kill_tree handles the Windows path via taskkill /F /T.
        start_new_session=True,
    )

    # Capture the pgid NOW, before the leader can exit and be reaped. Once
    # the leader is reaped, os.getpgid(proc.pid) raises ProcessLookupError
    # even though grandchildren in that group are still alive — defeating
    # the whole cleanup. None on Windows where the pgid concept doesn't apply.
    pgid: int | None = None
    if sys.platform != "win32":
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None

    try:
        output, _ = proc.communicate(timeout=file_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, pgid=pgid)
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            output = "(file timeout exceeded; output unavailable)"
        rc = 124  # de facto convention for "killed by timeout".
        output = (
            f"({file_timeout:.0f}s exceeded; "
            f"process tree SIGKILL'd)\n{output}"
        )
    except BaseException:
        # KeyboardInterrupt / runner crash — make sure no zombie
        # grandchildren outlive us.
        _kill_tree(proc, pgid=pgid)
        raise
    else:
        # Happy path: pytest exited on its own. Kill the group anyway in
        # case it left grandchildren behind; already-dead is a no-op.
        _kill_tree(proc, pgid=pgid)

        output +=  "\n"
    finally:
        if host_socket is not None:
            host_socket.close()
        for probe_path in (host_socket_path, host_canary_path):
            if probe_path is not None:
                try:
                    probe_path.unlink()
                except FileNotFoundError:
                    pass
        # Delete the temp root for this attempt. Nothing reads it after the
        # subprocess exits. More than 3000 of them fill the disk of the
        # runner over one suite.
        shutil.rmtree(temproot, ignore_errors=True)

    if rc == 5:
        # No tests collected in THIS file — legitimate per-file: a
        # platform-gated or fully-marker-filtered file (e.g. a win32-only
        # suite on Linux) collects nothing and must not fail the suite.
        # Tolerated here; the RUN-level guard in main() still fails when
        # NOTHING was collected across every file, so a broken invocation
        # (venv without pytest, -k that matches nothing) can't report green.
        rc = 0
    summary = _parse_pytest_summary(output)
    subproc_wall = time.monotonic() - subproc_start
    return file, rc, output, summary, subproc_wall


def _parse_pytest_summary(output: str) -> dict[str, int]:
    """Extract per-file test pass/fail/skip counts from pytest output.

    pytest prints a summary line like ``12 passed, 3 skipped, 1 failed in 2.1s``
    as the last non-empty line before the short test summary.  We scrape that
    line for the individual counts so the progress display can show test-level
    granularity instead of just file-level pass/fail.

    Returns a dict with keys ``passed``, ``failed``, ``skipped``, ``errors``,
    ``xfailed``, ``xpassed`` (only keys found in the output are present).
    """
    result: dict[str, int] = {}
    # Walk backwards from the end — the summary line is always near the tail.
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        # Match "N passed", "N failed", "N skipped", "N errors", "N xfailed", "N xpassed"
        for m in re.finditer(r"(\d+)\s+(passed|failed|skipped|errors|xfailed|xpassed)", line):
            result[m.group(2)] = int(m.group(1))
        # Also match "N error" (singular — pytest uses this sometimes).
        for m in re.finditer(r"(\d+)\s+error\b", line):
            result.setdefault("errors", result.get("errors", 0) + int(m.group(1)))
        if result:
            # Found the counts line — done.
            break
        # Stop at the short test summary header (if any) — everything above
        # that is individual failure details, not the counts line.
        if line.startswith("FAILED") or line.startswith("SHORT TEST SUMMARY"):
            break
    return result


def _format_file(file: Path, repo_root: Path) -> str:
    """Render a test-file path for display: strip the repo-root prefix
    when possible so output reads ``tests/acp/test_auth.py`` instead of
    ``/home/runner/work/hermes-agent/hermes-agent/tests/acp/test_auth.py``.

    Falls back to the absolute path for anything outside the repo root.
    """
    try:
        return str(file.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(file)


def _print_progress(
    tests_done: int,
    approx_total_tests: int,
    file: Path,
    rc: int,
    dur: float,
    repo_root: Path,
    tests_passed: int,
    tests_failed: int,
    test_counts: dict[Path, int],
    file_summary: dict[str, int] | None = None,
    subproc_wall: float | None = None,
) -> None:
    """Single-line live progress.

    When ``file_summary`` is provided (parsed from pytest output), the
    per-file parenthetical shows individual test pass/fail counts instead
    of just the total test count.

    ``subproc_wall`` is the actual subprocess wall-clock time (excluding
    queue-wait). When available, the display shows both the subprocess
    time and the queue-inclusive elapsed time.
    """
    status = "✓" if rc == 0 else "✗"
    pct = min((tests_done / approx_total_tests * 100), 100) if approx_total_tests else 0
    # Digit width for left-side counter padding (derived from total file count).
    fw = len(str(tests_passed + tests_failed))
    # Build per-file test count string.
    if file_summary:
        parts = []
        p = file_summary.get("passed", 0)
        f = file_summary.get("failed", 0)
        s = file_summary.get("skipped", 0)
        e = file_summary.get("errors", 0)
        if p:
            parts.append(f"{p}✓")
        if f:
            parts.append(f"{f}✗")
        if s:
            parts.append(f"{s}s")
        if e:
            parts.append(f"{e}e")
        # xfailed/xpassed are rare; include if present.
        xf = file_summary.get("xfailed", 0)
        xp = file_summary.get("xpassed", 0)
        if xf:
            parts.append(f"{xf}xf")
        if xp:
            parts.append(f"{xp}xp")
        test_str = " ".join(parts) + ", " if parts else ""
    else:
        n_tests = test_counts.get(file, 0)
        test_str = f"{n_tests} tests, " if n_tests else ""
    # Show subprocess time when available; fall back to queue-inclusive dur.
    if subproc_wall is not None:
        time_str = f"{subproc_wall:.1f}s"
    else:
        time_str = f"{dur:.1f}s"
    msg = (
        f"[{pct:5.1f}% | {tests_done:>5}/~{approx_total_tests}"
        f" | ✓{tests_passed:>{fw}} | ✗{tests_failed:>{fw}}] "
        f"{status} {_format_file(file, repo_root)} ({test_str}{time_str})"
    )
    # Truncate to terminal width if available (no clobbering ANSI lines).
    try:
        cols = os.get_terminal_size().columns
        if len(msg) > cols:
            msg = msg[: cols - 1] + "…"
    except OSError:
        pass
    print(msg, flush=True)


def _print_inline_failure(
    file: Path, output: str, repo_root: Path, pytest_passthrough: List[str]
) -> None:
    """Print a compact failure summary immediately when a file fails.

    Shows the tail of the pytest output (the failure section with stack
    traces) and a ready-to-run repro command, so the developer doesn't
    have to wait for the full run to finish before seeing what broke.
    """
    rel = _format_file(file, repo_root)
    # Build a repro command the developer can copy-paste.
    passthrough_str = " ".join(pytest_passthrough) if pytest_passthrough else ""
    repro = f"python -m pytest {rel}"
    if passthrough_str:
        repro += f" {passthrough_str}"

    # Grab just the failure lines (last ~30 lines of pytest output —
    # typically the FAILED summary + short test info).
    lines = output.rstrip().splitlines()
    tail = "\n".join(lines[-30:])

    print(flush=True)
    print(f"  ╔╍ Failed: {rel} ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    for line in tail.splitlines():
        print(f"  ║ {line}", flush=True)
    print("  ║", flush=True)
    print(f"  ║  Repro: {repro}", flush=True)
    print("  ╚╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    print(flush=True)


def _load_durations(repo_root: Path) -> dict[str, float]:
    """Read the duration cache from the repo root.

    Returns a dict mapping relative file paths (e.g.
    ``tests/tools/test_code_execution.py``) to wall-clock seconds from
    the last run. Missing or corrupt file → empty dict (safe fallback).
    """
    path = repo_root / _DURATIONS_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print("[ERROR] Failed to load json durations file! {e}")
        return {}


def _save_durations(
    file_times: List[Tuple[Path, float]],
    repo_root: Path,
) -> None:
    """Write the duration cache so future ``--slice`` runs can use it.

    Merges with any existing cache so entries from files not in the
    current run (e.g. from a different slice) are preserved. Keys are
    repo-relative paths so the cache is portable across checkouts
    and CI runners.
    """
    if os.environ.get("HERMES_TEST_OS_SANDBOX"):
        # A runner invoked from a test is inside a read-only repository mount.
        # Its timing data is ephemeral and must not punch a write hole in the
        # outer sandbox merely to update a performance cache.
        return
    data: dict[str, float] = _load_durations(repo_root)
    for f, t in file_times:
        key = _format_file(f, repo_root)
        data[key] = round(t, 3)
    path = repo_root / _DURATIONS_FILE
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_lpt_slices(
    files: List[Path],
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[List[Path]]:
    """Distribute files across N slices using LPT (Longest Processing Time first).

    Sorts files by estimated duration descending, then greedily assigns each
    file to the slice with the smallest accumulated time so far. This
    minimizes the makespan (max slice duration) and keeps CI jobs balanced.

    Files with no cached duration get a default estimate of 2.0s (roughly
    the P50 from profiling). This means first-time runs (no cache) still
    get reasonable distribution, and new files don't all land in one slice.

    Returns a list of N file-lists, one per slice (0-indexed).
    """
    if slice_count < 2:
        return [files]

    default_dur = 2.0
    file_durs: List[Tuple[Path, float]] = []
    for f in files:
        rel = _format_file(f, repo_root)
        dur = durations.get(rel, default_dur)
        file_durs.append((f, dur))

    # Sort longest first (LPT).
    file_durs.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment: for each file, add it to the slice with the
    # smallest current total.
    bucket_files: List[List[Path]] = [[] for _ in range(slice_count)]
    bucket_totals: List[float] = [0.0] * slice_count

    for f, dur in file_durs:
        min_idx = min(range(slice_count), key=lambda i: bucket_totals[i])
        bucket_files[min_idx].append(f)
        bucket_totals[min_idx] += dur

    return bucket_files


def _slice_files(
    files: List[Path],
    slice_index: int,
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[Path]:
    """Return the subset of *files* belonging to slice *slice_index*.

    Uses :func:`_compute_lpt_slices` for LPT distribution.

    ``slice_index`` is 1-indexed (1..slice_count) for ergonomics —
    ``--slice 1/4`` reads more naturally than ``--slice 0/4``.
    """
    if slice_count < 2:
        return files
    if not (1 <= slice_index <= slice_count):
        print(
            f"error: --slice index must be 1..{slice_count}, got {slice_index}",
            file=sys.stderr,
        )
        sys.exit(2)

    bucket_files = _compute_lpt_slices(files, slice_count, durations, repo_root)

    target = bucket_files[slice_index - 1]
    target_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0) for f in target
    )
    total_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0)
        for bucket in bucket_files
        for f in bucket
    )
    print(
        f"Slice {slice_index}/{slice_count}: {len(target)} files "
        f"(~{target_dur:.0f}s estimated of {total_dur:.0f}s total)",
        flush=True,
    )

    return target


def _make_stdio_glyph_safe() -> None:
    """Keep status glyphs from killing the runner on narrow console encodings.

    On native Windows, piped or legacy-console stdio defaults to a locale
    codec (usually cp1252) that cannot encode the ✓/✗ progress glyphs — the
    first per-file status line then dies with UnicodeEncodeError before a
    single test result is reported. Declare the runner's own output UTF-8
    (what CI and every modern terminal already are), with errors="replace"
    as the can't-crash backstop; where the encoding can't be changed, fall
    back to errors="replace" alone so glyphs degrade to "?" instead of
    killing the run. On already-UTF-8 stdio this is a no-op.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def main() -> int:
    _make_stdio_glyph_safe()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("HERMES_TEST_WORKERS") or (os.cpu_count() or 4) * 2),
        help="Parallel worker count (default: $HERMES_TEST_WORKERS or cpu_count*2)",
    )
    parser.add_argument(
        "--paths",
        default=os.environ.get("HERMES_TEST_PATHS", ":".join(_DEFAULT_ROOTS)),
        help=(
            "Colon-separated discovery roots (default: 'tests'). On "
            "Windows, ';' also separates and drive letters (C:\\...) are "
            "kept intact."
        ),
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Don't skip integration/ e2e/ during discovery",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(
            os.environ.get("HERMES_TEST_FILE_TIMEOUT", _DEFAULT_FILE_TIMEOUT_SECONDS)
        ),
        help=(
            "Per-file wall-clock cap in seconds. On timeout, the pytest "
            "subprocess and its full process tree are SIGKILL'd. "
            f"Default: {_DEFAULT_FILE_TIMEOUT_SECONDS}s ({round(_DEFAULT_FILE_TIMEOUT_SECONDS/60)} min), env: HERMES_TEST_FILE_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--file-retries",
        type=int,
        default=int(
            os.environ.get("HERMES_TEST_FILE_RETRIES", _DEFAULT_FILE_RETRIES)
        ),
        help=(
            "Re-run a failing test FILE this many times in a fresh subprocess "
            "before declaring it failed. A pass-on-retry counts as passed but "
            "is reported as FLAKY in the summary. 0 disables. "
            f"Default: {_DEFAULT_FILE_RETRIES}, env: HERMES_TEST_FILE_RETRIES."
        ),
    )
    parser.add_argument(
        "--slice",
        metavar="I/N",
        help=(
            "Run only slice I of N (e.g. --slice 1/4). "
            "Files are distributed across slices using cached durations "
            "so each slice takes roughly equal wall time. "
            "Without a duration cache, files are distributed by count. "
            "Env: HERMES_TEST_SLICE (format: I/N)."
        ),
    )
    parser.add_argument(
        "--generate-slices",
        metavar="N",
        type=int,
        help=(
            "Discover test files, distribute them across N slices using "
            "LPT on cached durations, and print a JSON matrix to stdout "
            "then exit (no tests run). The JSON has the shape "
            "'{\"slices\": [{\"index\": 1, \"files\": [\"tests/foo.py\", ...]}, ...]}' "
            "so the CI generate job can feed it directly into a matrix."
        ),
    )
    parser.add_argument(
        "--files",
        metavar="LIST",
        help=(
            "Explicit colon-separated list of test files to run (on "
            "Windows, ';' also separates and drive letters are kept "
            "intact). Bypasses discovery entirely — used by CI matrix "
            "jobs that receive their file list from the generate job."
        ),
    )
    parser.add_argument(
        "paths_positional",
        nargs="*",
        metavar="PATH",
        help=(
            "Restrict discovery to these paths (directories or .py files). "
            "Mutually exclusive with --paths. Anything after a literal '--' "
            "separator is passed through to each per-file pytest invocation."
        ),
    )
    # Split argv into "our flags + positional paths" vs "pytest passthrough".
    #
    # Two ways to pass args through to the per-file pytest invocation:
    #   1. Explicit ``--`` separator: everything after it goes to pytest.
    #   2. Bare pytest flags anywhere before ``--``: any token starting with
    #      ``-`` that isn't one of OUR options is routed to pytest, so a bare
    #      ``-q`` / ``-v`` / ``-x`` / ``--tb=long`` / ``-k expr`` "just works"
    #      without the developer remembering the ``--``. This matches the
    #      docstring's promise and pytest muscle-memory.
    #
    # The subtlety bare-flag routing must handle: value-taking pytest flags
    # given in space-separated form (``-k expr``, ``-m mark``, ``-p plugin``,
    # ``-o name=val``). Naively, ``expr`` would look like a positional path and
    # clobber discovery. We peel the following token along with such flags so
    # it never reaches our positional ``paths``. ``=``-joined forms
    # (``-k=expr``, ``--tb=long``) are self-contained and need no lookahead.
    OUR_FLAGS = {
        "-j", "--jobs", "--paths", "--include-integration",
        "--file-timeout", "--file-retries", "--slice", "--generate-slices", "--files",
    }
    # pytest short flags that consume the NEXT token as their value.
    PYTEST_VALUE_FLAGS = {"-k", "-m", "-p", "-o", "-c", "-r", "-W"}

    def _is_our_flag(tok: str) -> bool:
        # Match exact (``-j``, ``--paths``), ``=``-joined (``--paths=x``),
        # and attached short-value (``-j4``) forms of our own options.
        if tok in OUR_FLAGS:
            return True
        head = tok.split("=", 1)[0]
        if head in OUR_FLAGS:
            return True
        # Attached short value, e.g. ``-j4`` → ``-j``.
        if len(tok) > 2 and tok[:2] in OUR_FLAGS and not tok[1] == "-":
            return True
        return False

    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        before, explicit_passthrough = argv[:sep], argv[sep + 1 :]
    else:
        before, explicit_passthrough = argv, []

    our_args: List[str] = []
    bare_passthrough: List[str] = []
    i = 0
    while i < len(before):
        tok = before[i]
        if tok.startswith("-") and not _is_our_flag(tok):
            bare_passthrough.append(tok)
            # Pull the value token for space-separated value flags.
            if tok in PYTEST_VALUE_FLAGS and i + 1 < len(before):
                bare_passthrough.append(before[i + 1])
                i += 2
                continue
        else:
            our_args.append(tok)
        i += 1

    args = parser.parse_args(our_args)

    # ── Node-id selectors → file + ``-k`` filter ────────────────────────────
    # This runner is FILE-granular: it spawns one ``pytest <file>`` per test
    # file. A pytest node id (``tests/foo.py::TestBar::test_baz``) is not an
    # existing path, so discovery silently dropped it and the run exited with
    # "No test files to run" — the selector looked accepted but nothing ran.
    # Translate instead: run the FILE and narrow with ``-k`` on the last
    # segment, which is what the caller meant.
    node_id_selectors: List[Tuple[str, str]] = []
    if args.paths_positional:
        translated: List[str] = []
        for raw in args.paths_positional:
            if "::" not in raw:
                translated.append(raw)
                continue
            file_part, _, selector = raw.partition("::")
            leaf = selector.rsplit("::", 1)[-1]
            # Strip a parametrized id (``test_x[case]``) down to the function
            # name; ``-k`` matches substrings, and brackets are -k syntax.
            leaf = leaf.split("[", 1)[0]
            node_id_selectors.append((raw, leaf))
            translated.append(file_part)
        if node_id_selectors:
            args.paths_positional = translated
            keys = [leaf for _, leaf in node_id_selectors]
            expr = " or ".join(dict.fromkeys(keys))
            for raw, leaf in node_id_selectors:
                print(
                    f"note: '{raw}' is a pytest node id; this runner is "
                    f"file-granular. Running the file with -k {leaf!r}.",
                    file=sys.stderr,
                )
            # Only inject -k when the caller didn't pass one themselves; their
            # explicit filter wins over our inferred one.
            if not any(
                t == "-k" or t.startswith("-k=") or (t.startswith("-k") and len(t) > 2)
                for t in bare_passthrough + explicit_passthrough
            ):
                bare_passthrough = bare_passthrough + ["-k", expr]

    # Bare flags run before any explicit ``--`` passthrough so ordering is
    # intuitive (``run_tests.sh tests/foo.py -q -- --tb=long`` → ``-q --tb=long``).
    pytest_passthrough = bare_passthrough + explicit_passthrough

    # Parse --slice (or HERMES_TEST_SLICE) early so we can exit on bad input
    # before doing any expensive discovery.
    slice_raw = args.slice or os.environ.get("HERMES_TEST_SLICE")
    slice_index: int | None = None
    slice_count: int = 1
    if slice_raw:
        try:
            idx_s, count_s = slice_raw.split("/", 1)
            slice_index = int(idx_s)
            slice_count = int(count_s)
        except (ValueError, AttributeError):
            print(f"error: --slice must be I/N (e.g. 1/4), got: {slice_raw!r}", file=sys.stderr)
            sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent

    # --files: explicit file list from the CI generate job — skip discovery.
    if args.files:
        files = [repo_root / f for f in _split_pathspec(args.files)]
        roots = []
    else:
        # Resolve discovery roots: positional path args override --paths if any
        # were supplied, otherwise --paths (which itself defaults to 'tests').
        if args.paths_positional:
            roots = [repo_root / p for p in args.paths_positional]
        else:
            roots = [repo_root / p for p in _split_pathspec(args.paths)]

        if args.include_integration:
            # Caller takes responsibility — typically used via explicit -k filter.
            global _SKIP_PARTS  # noqa: PLW0603 — config knob
            _SKIP_PARTS = set()

        files = _discover_files(roots)

    if not files:
        print("No test files to run", file=sys.stderr)
        return 1

    # --generate-slices: compute LPT distribution and emit JSON, then exit.
    if args.generate_slices is not None:
        durations = _load_durations(repo_root)
        slices = _compute_lpt_slices(
            files, args.generate_slices, durations, repo_root
        )
        matrix = {
            "slice": [
                {
                    "index": i + 1,
                    "files": ":".join(_format_file(f, repo_root) for f in bucket),
                }
                for i, bucket in enumerate(slices)
            ]
        }
        # Print to stdout so the CI step can capture it with $().
        print(json.dumps(matrix))
        return 0

    # Count individual tests per file
    test_counts = _approximately_count_tests(files, repo_root)
    approx_total_tests = sum(test_counts.values())

    # Apply slicing if requested — distribute files across CI jobs by
    # estimated duration so no one job gets all the slow files.
    if slice_index is not None:
        durations = _load_durations(repo_root)
        files = _slice_files(files, slice_index, slice_count, durations, repo_root)
        # Recount after slicing.
        test_counts = {f: test_counts[f] for f in files if f in test_counts}
        approx_total_tests = sum(test_counts.values())

    if roots:
        roots_str = [str(r.relative_to(repo_root)) if r.is_relative_to(repo_root) else str(r) for r in roots]
        print(
            f"Discovered {len(files)} test files (~{approx_total_tests} tests) under "
            f"{roots_str}; running with -j {args.jobs}",
            flush=True,
        )
    else:
        print(
            f"Running {len(files)} test files (~{approx_total_tests} tests) "
            f"with -j {args.jobs}",
            flush=True,
        )

    # Capture and print on completion (out-of-order is fine — keeps the
    # terminal clean rather than interleaving N parallel pytest outputs).
    failures: List[Tuple[Path, str, Dict[str, int]]] = []
    file_times: List[Tuple[Path, float]] = []  # (file, subprocess_wall) for distribution
    started = time.monotonic()
    files_done = 0
    tests_done = 0
    pass_count = 0
    fail_count = 0
    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0
    # Every collected outcome, not just pass/fail: a legitimately all-skipped
    # (platform-gated) file reports "2 skipped" and must NOT trip the
    # nothing-ran guard, whereas a file that died before collection reports
    # nothing at all and must.
    tests_collected = 0
    lock = threading.Lock()

    def _on_done(file: Path, started_at: float, fut: "Future[Tuple[Path, int, str, Dict[str, int], float]]") -> None:
        nonlocal files_done, tests_done, pass_count, fail_count, tests_passed, tests_failed, tests_skipped
        nonlocal tests_collected
        n_tests = test_counts.get(file, 0)
        try:
            fpath, rc, output, summary, subproc_wall = fut.result()
        except Exception as exc:  # noqa: BLE001 — must always advance counter
            with lock:
                files_done += 1
                tests_done += n_tests
                fail_count += 1
                failures.append((file, f"runner crashed: {exc!r}", {}))
                _print_progress(
                    tests_done, approx_total_tests, file, 1,
                    time.monotonic() - started_at,
                    repo_root, tests_passed, tests_failed,
                    test_counts,
                    subproc_wall=0.0,
                )
            return
        with lock:
            files_done += 1
            tests_done += n_tests
            # Accumulate test-level counts from parsed summary.
            tests_passed += summary.get("passed", 0)
            tests_failed += summary.get("failed", 0)
            tests_skipped += summary.get("skipped", 0)
            tests_collected += sum(
                summary.get(k, 0)
                for k in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
            )
            file_times.append((fpath, subproc_wall))
            if rc == 0:
                pass_count += 1
            else:
                fail_count += 1
                failures.append((fpath, output, summary))
            _print_progress(
                tests_done, approx_total_tests, fpath, rc,
                time.monotonic() - started_at,
                repo_root, tests_passed, tests_failed,
                test_counts,
                file_summary=summary,
                subproc_wall=subproc_wall,
            )
            if rc != 0:
                _print_inline_failure(fpath, output, repo_root, pytest_passthrough)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures: List[Future] = []
        for file in files:
            t0 = time.monotonic()
            fut = pool.submit(
                _run_one_file, file, pytest_passthrough, repo_root,
                args.file_timeout, args.file_retries,
            )
            fut.add_done_callback(lambda f, file=file, t0=t0: _on_done(file, t0, f))
            futures.append(fut)
        # Block until everything's done. ThreadPoolExecutor.__exit__ waits
        # for all submitted work, but doing it explicitly here makes the
        # control flow obvious.
        for fut in futures:
            fut.result() if fut.exception() is None else None

    elapsed = time.monotonic() - started
    print()
    pct = min(100, (tests_done / approx_total_tests * 100)) if approx_total_tests else 0
    skipped_note = f", {tests_skipped} skipped" if tests_skipped else ""
    print(f"=== Summary: {len(files)} files, {tests_passed} tests passed, {tests_failed} failed{skipped_note} ({pct:.0f}% complete) in {elapsed:.1f}s ({args.jobs} workers) ===")

    # Host-OS gating note: tests marked for another OS were skipped by the
    # conftest hook, not run. Say so explicitly — a green local run on Linux
    # proves nothing about the macos_only/windows_only tests, and the reader
    # should know where they DO run rather than misreading skips as coverage.
    off_host = _off_host_marker_files(files)
    if off_host:
        print()
        for marker, n in sorted(off_host.items()):
            _, lane = _OS_MARKERS[marker]
            print(
                f"  note: {marker} tests (in {n} file{'s' if n != 1 else ''}) were "
                f"SKIPPED on this host ({sys.platform}); they run on {lane}."
            )

    # Zero tests collected across the WHOLE run is NOT a pass. Per-file rc=5
    # is deliberately tolerated above (platform-gated files), but if NOTHING
    # ran anywhere the invocation itself was broken — a venv without pytest, a
    # -k/-m filter that matched nothing, or collection erroring everywhere.
    # The summary line above reads green at a glance ("0 failed ... 100%
    # complete"), which has been misread as a successful verification, so say
    # it plainly AND fail the exit code.
    no_tests_ran_at_all = bool(files) and tests_collected == 0
    if no_tests_ran_at_all:
        print()
        print(
            "=== ✗ NO TESTS RAN — 0 collected across "
            f"{len(files)} file{'s' if len(files) != 1 else ''}. "
            "This is NOT a pass. ==="
        )
        print(
            "  Common causes: the selected venv has no pytest; a -k/-m filter "
            "matched nothing; or collection errored in every file."
        )
        print("  Check the per-file output above for the real error.")

    # Flaky files: failed once, passed on the automatic retry. Green, but
    # loudly reported so they get fixed instead of silently re-flaking.
    if _FLAKY_RESULTS:
        print()
        print(f"=== ⚠ {len(_FLAKY_RESULTS)} FLAKY file{'s' if len(_FLAKY_RESULTS) != 1 else ''} (failed once, passed on retry — fix these) ===")
        for f, output in _FLAKY_RESULTS:
            print(f"  {_format_file(f, repo_root)}")
            print(output.rstrip())

    # Save durations for future --slice runs. Each slice writes its own
    # partial test_durations.json; a CI merge step joins them later.
    # Locally, _save_durations merges with any existing cache so entries
    # from previous runs aren't lost.
    if file_times:
        _save_durations(file_times, repo_root)
        print(f"  Durations cached to {_DURATIONS_FILE} ({len(file_times)} files)")

    # Per-file time distribution (throwaway diagnostic — shows how
    # subprocess time is distributed so we can see if startup dominates).
    if file_times:
        times = sorted([t for _, t in file_times])
        total_subproc = sum(times)
        median_t = times[len(times) // 2]
        p50 = median_t
        p90 = times[int(len(times) * 0.90)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[min(int(len(times) * 0.99), len(times) - 1)]
        max_t = times[-1]
        # How many files finish in <1s? That's roughly "just startup".
        fast = sum(1 for t in times if t < 1.0)
        fast_2s = sum(1 for t in times if t < 2.0)
        print()
        print("=== Per-file subprocess time distribution ===")
        print(f"  Files:   {len(times)}")
        print(f"  Total subprocess CPU-wall: {total_subproc:.1f}s  (runner wall: {elapsed:.1f}s, parallelism: {args.jobs}x)")
        print(f"  P50: {p50:.2f}s  P90: {p90:.2f}s  P95: {p95:.2f}s  P99: {p99:.2f}s  Max: {max_t:.2f}s")
        print(f"  <1s: {fast} files ({fast/len(times)*100:.0f}%)  <2s: {fast_2s} files ({fast_2s/len(times)*100:.0f}%)")
        # Top 10 slowest files — likely the ones dragging the run.
        slowest = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
        print("  Top 10 slowest:")
        for f, t in slowest:
            print(f"    {t:>6.2f}s  {_format_file(f, repo_root)}")

    if failures:
        print()
        print("=== Failure output ===")
        for file, output, _summary in failures:
            print()
            print(f"--- {_format_file(file, repo_root)} ---")
            print(output.rstrip())
        print()
        # Split: files with actual test failures vs non-zero exit for other reasons
        test_fail_files = [(f, s) for f, _o, s in failures if s.get("failed", 0) > 0]
        all_passed_but_nonzero = [(f, s) for f, _o, s in failures
                                  if s.get("failed", 0) == 0 and s.get("passed", 0) > 0]
        no_tests_ran = [(f, s) for f, _o, s in failures
                        if s.get("failed", 0) == 0 and s.get("passed", 0) == 0]
        if test_fail_files:
            total_tf = sum(s.get("failed", 0) for _, s in test_fail_files)
            print(f"=== {len(test_fail_files)} file{'s' if len(test_fail_files) != 1 else ''} with test failures ({total_tf} test{'s' if total_tf != 1 else ''} failed) ===")
            for file, s in test_fail_files:
                nf = s.get("failed", 0)
                print(f"  {_format_file(file, repo_root)}  ({nf} test{'s' if nf != 1 else ''} failed)")
        if all_passed_but_nonzero:
            print(f"=== {len(all_passed_but_nonzero)} file{'s' if len(all_passed_but_nonzero) != 1 else ''} where all tests passed but pytest exited non-zero (warnings-as-errors, hook failures, etc.) ===")
            for file, s in all_passed_but_nonzero:
                print(f"  {_format_file(file, repo_root)}  ({s.get('passed', 0)} passed)")
        if no_tests_ran:
            print(f"=== {len(no_tests_ran)} file{'s' if len(no_tests_ran) != 1 else ''} where no tests ran (collection/import error, timeout before collection, etc.) ===")
            for file, s in no_tests_ran:
                print(f"  {_format_file(file, repo_root)}")
        return 1

    if no_tests_ran_at_all:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
