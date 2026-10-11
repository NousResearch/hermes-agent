# Container-Aware CLI Review Fixes Spec

**PR:** NousResearch/hermes-agent#7543
**Review:** cursor[bot] bugbot review (4094049442) + two prior rounds
**Date:** 2026-04-12
**Branch:** `feat/container-aware-cli-clean`

## Review Issues Summary

Six issues were raised across three bugbot review rounds. Three were fixed in intermediate commits (38277a6a, 726cf90f). This spec addresses remaining design concerns surfaced by those reviews and simplifies the implementation based on interview decisions.

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `os.execvp` retry loop unreachable | Medium | Fixed in 79e8cd12 (switched to subprocess.run) |
| 2 | Redundant `shutil.which("sudo")` | Medium | Fixed in 38277a6a (reuses `sudo` var) |
| 3 | Missing `chown -h` on symlink update | Low | Fixed in 38277a6a |
| 4 | Container routing after `parse_args()` | High | Fixed in 726cf90f |
| 5 | Hardcoded `/home/${user}` | Medium | Fixed in 726cf90f |
| 6 | Group membership not gated on `container.enable` | Low | Fixed in 726cf90f |

The mechanical fixes are in place but the overall design needs revision. The retry loop, error swallowing, and process model have deeper issues than what the bugbot flagged.

---

## Spec: Revised `_exec_in_container`

### Design Principles

1. **Let it crash.** No silent fallbacks. If `.container-mode` exists but something goes wrong, the error propagates naturally (Python traceback). The only case where container routing is skipped is when `.container-mode` doesn't exist or `HERMES_DEV=1`.
2. **No retries.** Probe once for sudo, exec once. If it fails, docker/podman's stderr reaches the user verbatim.
3. **Completely transparent.** No error wrapping, no prefixes, no spinners. Docker's output goes straight through.
4. **`os.execvp` on the happy path.** Replace the Python process entirely so there's no idle parent during interactive sessions. Note: `execvp` never returns on success (process is replaced) and raises `OSError` on failure (it does not return a value). The container process's exit code becomes the process exit code by definition — no explicit propagation needed.
5. **One human-readable exception to "let it crash".** `subprocess.TimeoutExpired` from the sudo probe gets a specific catch with a readable message, since a raw traceback for "your Docker daemon is slow" is confusing. All other exceptions propagate naturally.

### Execution Flow

```
1. get_container_exec_info()
   - HERMES_DEV=1 → return None (skip routing)
   - Inside container → return None (skip routing)
   - .container-mode doesn't exist → return None (skip routing)
   - .container-mode exists → parse and return dict
   - .container-mode exists but malformed/unreadable → LET IT CRASH (no try/except)

2. _exec_in_container(container_info, sys.argv[1:])
   a. shutil.which(backend) → if None, print "{backend} not found on PATH" and sys.exit(1)
   b. Sudo probe: subprocess.run([runtime, "inspect", "--format", "ok", container_name], timeout=15)
      - If succeeds → needs_sudo = False
      - If fails → try subprocess.run([sudo, "-n", runtime, "inspect", ...], timeout=15)
        - If succeeds → needs_sudo = True
        - If fails → print error with sudoers hint (including why -n is required) and sys.exit(1)
      - If TimeoutExpired → catch specifically, print human-readable message about slow daemon
   c. Build exec_cmd: [sudo? + runtime, "exec", tty_flags, "-u", exec_user, env_flags, container, hermes_bin, *cli_args]
   d. os.execvp(exec_cmd[0], exec_cmd)
      - On success: process is replaced — Python is gone, container exit code IS the process exit code
      - On OSError: let it crash (natural traceback)
```

### Changes to `hermes_cli/main.py`

#### `_exec_in_container` — rewrite

Remove:
- The entire retry loop (`max_retries`, `for attempt in range(...)`)
- Spinner logic (`"Waiting for container..."`, dots)
- Exit code classification (125/126/127 handling)
- `subprocess.run` for the exec call (keep it only for the sudo probe)
- Special TTY vs non-TTY retry counts
- The `time` import (no longer needed)

Change:
- Use `os.execvp(exec_cmd[0], exec_cmd)` as the final call
- Keep the `subprocess` import only for the sudo probe
- Keep TTY detection for the `-it` vs `-i` flag
- Keep env var forwarding (TERM, COLORTERM, LANG, LC_ALL)
- Keep the sudo probe as-is (it's the one "smart" part)
- Bump probe `timeout` from 5s to 15s — cold podman on a loaded machine needs headroom
- Catch `subprocess.TimeoutExpired` specifically on both probe calls — print a readable message about the daemon being unresponsive instead of a raw traceback
- Expand the sudoers hint error message to explain *why* `-n` (non-interactive) is required: a password prompt would hang the CLI or break piped commands

The function becomes roughly:

```python
def _exec_in_container(container_info: dict, cli_args: list):
    """Replace the current process with a command inside the managed container.

    Probes whether sudo is needed (rootful containers), then os.execvp
    into the container. If exec fails, the OS error propagates naturally.
    """
    import shutil
    import subprocess

    backend = container_info["backend"]
    container_name = container_info["container_name"]
    exec_user = container_info["exec_user"]
    hermes_bin = container_info["hermes_bin"]

    runtime = shutil.which(backend)
    if not runtime:
        print(f"Error: {backend} not found on PATH. Cannot route to container.",
              file=sys.stderr)
        sys.exit(1)

    # Probe whether we need sudo to see the rootful container.
    # Timeout is 15s — cold podman on a loaded machine can take a while.
    # TimeoutExpired is caught specifically for a human-readable message;
    # all other exceptions propagate naturally.
    needs_sudo = False
    sudo = None
    try:
        probe = subprocess.run(
            [runtime, "inspect", "--format", "ok", container_name],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        print(
            f"Error: timed out waiting for {backend} to respond.\n"
            f"The {backend} daemon may be unresponsive or starting up.",
            file=sys.stderr,
        )
        sys.exit(1)

    if probe.returncode != 0:
        sudo = shutil.which("sudo")
        if sudo:
            try:
                probe2 = subprocess.run(
                    [sudo, "-n", runtime, "inspect", "--format", "ok", container_name],
                    capture_output=True, text=True, timeout=15,
                )
            except subprocess.TimeoutExpired:
                print(
                    f"Error: timed out waiting for sudo {backend} to respond.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if probe2.returncode == 0:
                needs_sudo = True
            else:
                print(
                    f"Error: container '{container_name}' not found via {backend}.\n"
                    f"\n"
                    f"The NixOS service runs the container as root. Your user cannot\n"
                    f"see it because {backend} uses per-user namespaces.\n"
                    f"\n"
                    f"Fix: grant passwordless sudo for {backend}. The -n (non-interactive)\n"
                    f"flag is required because the CLI calls sudo non-interactively —\n"
                    f"a password prompt would hang or break piped commands:\n"
                    f"\n"
                    f'  security.sudo.extraRules = [{{\n'
                    f'    users = [ "{os.getenv("USER", "your-user")}" ];\n'
                    f'    commands = [{{ command = "{runtime}"; options = [ "NOPASSWD" ]; }}];\n'
                    f'  }}];\n'
                    f"\n"
                    f"Or run: sudo hermes {' '.join(cli_args)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"The container may be running under root. Try: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

    is_tty = sys.stdin.isatty()
    tty_flags = ["-it"] if is_tty else ["-i"]

    env_flags = []
    for var in ("TERM", "COLORTERM", "LANG", "LC_ALL"):
        val = os.environ.get(var)
        if val:
            env_flags.extend(["-e", f"{var}={val}"])

    cmd_prefix = [sudo, "-n", runtime] if needs_sudo else [runtime]
    exec_cmd = (
        cmd_prefix + ["exec"]
        + tty_flags
        + ["-u", exec_user]
        + env_flags
        + [container_name, hermes_bin]
        + cli_args
    )

    # execvp replaces this process entirely — it never returns on success.
    # On failure it raises OSError, which propagates naturally.
    os.execvp(exec_cmd[0], exec_cmd)
```

#### Container routing call site in `main()` — remove try/except

Current:
```python
try:
    from hermes_cli.config import get_container_exec_info
    container_info = get_container_exec_info()
    if container_info:
        _exec_in_container(container_info, sys.argv[1:])
        sys.exit(1)  # exec failed if we reach here
except SystemExit:
    raise
except Exception:
    pass  # Container routing unavailable, proceed locally
```

Revised:
```python
from hermes_cli.config import get_container_exec_info
container_info = get_container_exec_info()
if container_info:
    _exec_in_container(container_info, sys.argv[1:])
    # Unreachable: os.execvp never returns on success (process is replaced)
    # and raises OSError on failure (which propagates as a traceback).
    # This line exists only as a defensive assertion.
    sys.exit(1)
```

No try/except. If `.container-mode` doesn't exist, `get_container_exec_info()` returns `None` and we skip routing. If it exists but is broken, the exception propagates with a natural traceback.

Note: `sys.exit(1)` after `_exec_in_container` is dead code in all paths — `os.execvp` either replaces the process or raises. It's kept as a belt-and-suspenders assertion with a comment marking it unreachable, not as actual error handling.

### Changes to `hermes_cli/config.py`

#### `get_container_exec_info` — remove inner try/except

Current code catches `(OSError, IOError)` and returns `None`. This silently hides permission errors, corrupt files, etc.

Change: Remove the try/except around file reading. Keep the early returns for `HERMES_DEV=1` and `_is_inside_container()`. The `FileNotFoundError` from `open()` when `.container-mode` doesn't exist should still return `None` (this is the "container mode not enabled" case). All other exceptions propagate.

```python
def get_container_exec_info() -> Optional[dict]:
    if os.environ.get("HERMES_DEV") == "1":
        return None
    if _is_inside_container():
        return None

    container_mode_file = get_hermes_home() / ".container-mode"

    try:
        with open(container_mode_file, "r") as f:
            # ... parse key=value lines ...
    except FileNotFoundError:
        return None
    # All other exceptions (PermissionError, malformed data, etc.) propagate

    return { ... }
```

---

## Spec: NixOS Module Changes

### Symlink creation — simplify to two branches

Current: 4 branches (symlink exists, directory exists, other file, doesn't exist).

Revised: 2 branches.

```bash
if [ -d "${symlinkPath}" ] && [ ! -L "${symlinkPath}" ]; then
  # Real directory — back it up, then create symlink
  _backup="${symlinkPath}.bak.$(date +%s)"
  echo "hermes-agent: backing up existing ${symlinkPath} to $_backup"
  mv "${symlinkPath}" "$_backup"
fi
# For everything else (symlink, doesn't exist, etc.) — just force-create
ln -sfn "${target}" "${symlinkPath}"
chown -h ${user}:${cfg.group} "${symlinkPath}"
```

`ln -sfn` handles: existing symlink (replaces), doesn't exist (creates), and after the `mv` above (creates). The only case that needs special handling is a real directory, because `ln -sfn` cannot atomically replace a directory.

Note: there is a theoretical race between the `[ -d ... ]` check and the `mv` (something could create/remove the directory in between). In practice this is a NixOS activation script running as root during `nixos-rebuild switch` — no other process should be touching `~/.hermes` at that moment. Not worth adding locking for.

### Sudoers — document, don't auto-configure

Do NOT add `security.sudo.extraRules` to the module. Document the sudoers requirement in the module's description/comments and in the error message the CLI prints when sudo probe fails.

### Group membership gating — keep as-is

The fix in 726cf90f (`cfg.container.enable && cfg.container.hostUsers != []`) is correct. Leftover group membership when container mode is disabled is harmless. No cleanup needed.

---

## Spec: Test Rewrite

The existing test file (`tests/hermes_cli/test_container_aware_cli.py`) has 16 tests. With the simplified exec model, several are obsolete.

### Tests to keep (update as needed)

- `test_is_inside_container_dockerenv` — unchanged
- `test_is_inside_container_containerenv` — unchanged
- `test_is_inside_container_cgroup_docker` — unchanged
- `test_is_inside_container_false_on_host` — unchanged
- `test_get_container_exec_info_returns_metadata` — unchanged
- `test_get_container_exec_info_none_inside_container` — unchanged
- `test_get_container_exec_info_none_without_file` — unchanged
- `test_get_container_exec_info_skipped_when_hermes_dev` — unchanged
- `test_get_container_exec_info_not_skipped_when_hermes_dev_zero` — unchanged
- `test_get_container_exec_info_defaults` — unchanged
- `test_get_container_exec_info_docker_backend` — unchanged

### Tests to add

- `test_get_container_exec_info_crashes_on_permission_error` — verify that `PermissionError` propagates (no silent `None` return)
- `test_exec_in_container_calls_execvp` — verify `os.execvp` is called with correct args (runtime, tty flags, user, env, container, binary, cli args)
- `test_exec_in_container_sudo_probe_sets_prefix` — verify that when first probe fails and sudo probe succeeds, `os.execvp` is called with `sudo -n` prefix
- `test_exec_in_container_no_runtime_hard_fails` — keep existing, verify `sys.exit(1)` when `shutil.which` returns None
- `test_exec_in_container_non_tty_uses_i_only` — update to check `os.execvp` args instead of `subprocess.run` args
- `test_exec_in_container_probe_timeout_prints_message` — verify that `subprocess.TimeoutExpired` from the probe produces a human-readable error and `sys.exit(1)`, not a raw traceback
- `test_exec_in_container_container_not_running_no_sudo` — verify the path where runtime exists (`shutil.which` returns a path) but probe returns non-zero and no sudo is available. Should print the "container may be running under root" error. This is distinct from `no_runtime_hard_fails` which covers `shutil.which` returning None.

### Tests to delete

- `test_exec_in_container_tty_retries_on_container_failure` — retry loop removed
- `test_exec_in_container_non_tty_retries_silently_exits_126` — retry loop removed
- `test_exec_in_container_propagates_hermes_exit_code` — no subprocess.run to check exit codes; execvp replaces the process. Note: exit code propagation still works correctly — when `os.execvp` succeeds, the container's process *becomes* this process, so its exit code is the process exit code by OS semantics. No application code needed, no test needed. A comment in the function docstring documents this intent for future readers.

---

## Out of Scope

- Auto-configuring sudoers rules in the NixOS module
- Any changes to `get_container_exec_info` parsing logic beyond the try/except narrowing
- Changes to `.container-mode` file format
- Changes to the `HERMES_DEV=1` bypass
- Changes to container detection logic (`_is_inside_container`)
