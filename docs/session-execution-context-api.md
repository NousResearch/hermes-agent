# Session execution context API

Public defining module: `hermes_cli.session_execution`. Trusted plugins register in the serving Python process; no monkeypatching, no process-global environment mutation.

```python
from hermes_cli.session_execution import (
    SessionExecutionContext, ComputerUseLaunchContext,
    register_session_execution_context, remove_session_execution_context,
    get_session_execution_context,
)

context = SessionExecutionContext(
    env_set={
        "XDG_RUNTIME_DIR": private_runtime, "WAYLAND_DISPLAY": "wayland-1",
        "DISPLAY": private_x_display, "DBUS_SESSION_BUS_ADDRESS": private_bus,
        "CUA_DRIVER_RS_ENABLE_WAYLAND": "1",
    },
    env_unset=frozenset({
        "HYPRLAND_INSTANCE_SIGNATURE", "YDOTOOL_SOCKET", "CUA_INJECT_SOCKET",
        "SWAYSOCK", "AT_SPI_BUS_ADDRESS", "SESSION_MANAGER", "XAUTHORITY",
    }),
    command_prefix=("/absolute/owner-launcher", "fixed-owner-argument", "--"),
    computer_use=ComputerUseLaunchContext(
        driver_command="/absolute/path/to/cua-driver", private_daemon=True,
        command_prefix=("/absolute/driver-containment-wrapper", "--"),
        runtime_dir=private_runtime,
        no_overlay=False, session_name="Work desktop", theme="cua.default",
        desktop_only=True, allow_input=check_input_is_allowed,
    ),
    validate=check_context_is_live,
)
register_session_execution_context(session_id, context, task_ids=(task_id,))
get_session_execution_context(session_id=session_id, task_id=task_id)
remove_session_execution_context(session_id)  # idempotent bool; revokes old snapshots
```

## Identity, validation, environment

Registration is profile-home scoped and immutable (defensive copies). Replacement invalidates old execution leases. Explicit task aliases are required when task ID differs from session ID; conflicting registered owners raise `SessionExecutionError`. There is no inferred delegate inheritance. Register only when attachment/generation changes, not on every tool call.

`validate` is an optional zero-argument callable that must return literal `True`. False/exception fails closed at resolution and before child launches/driver calls. Register only live contexts. Removal means detachment: future fresh lookups are unbound, while old cached leases cannot launch again. A plugin must block tool dispatch if reattachment fails; do not remove a dead attachment and then let execution proceed unbound.

Local foreground/background/PTY and shell-snapshot children receive exact set/unset after ambient merge and before the existing secret sanitizer. Snapshot restoration and login-shell startup cannot override routing keys. Cua receives the same overlay through manifest, daemon, status, MCP, CLI fallback, and teardown launches. MCP's implicit default-env inheritance is scrubbed before executing the configured wrappers when it would reintroduce unset keys. Provider secret sanitization and Cua permission selection remain authoritative; this API cannot pass provider secrets around sanitization or request approval bypass. Reserved driver permission/bypass env is stripped and rebuilt from Hermes' existing mode resolver.

Nonlocal terminal backends and non-Cua computer-use backends refuse registered contexts. Host-control-plane terminal calls explicitly marked `_host_local` remain outside this routing. File-tool-created local environments use the same generation ownership to avoid seeding the terminal cache with an unbound shell.

## Child ownership and wrappers

Both `command_prefix` fields default to `()`. They accept argv sequences copied to tuples, not shell text; the first argument must be an absolute available executable. The final logical invocation is:

- Terminal: `[*context.command_prefix, shell, *shell_args]`
- Cua: `[*context.command_prefix, *context.computer_use.command_prefix, driver, *driver_args]`

The external launcher must preserve stdin/stdout/stderr, exit status, signals, and PTY behavior, and attach executed children to its owned lifetime scope. Explicit session prefixes replace the background executor's generic systemd scope wrapper, preventing a later scope hop out of the owner's containment. No core code infers an external scope or knows plugin-specific ownership names. The private daemon's stop command retains its original prefixes/env after lease revocation, solely to stop that daemon.

This is cooperative routing, **not an OS security sandbox**. Same-user host files/sockets remain reachable unless the external manager/wrapper establishes isolation. Core registration alone does not terminate already-running terminal jobs; their lifetime belongs to the external owner.

## Private driver runtime and desktop scope

`runtime_dir` is optional and requires `private_daemon=True`. It must already exist as an owned private directory; replacement or permission changes fail validation. Its path must be short enough for Unix sockets. The daemon socket is created there without global TMPDIR mutation. Any applicable already-configured capability manifest is copied byte-for-byte to an exclusive mode-0600 file there, then removed on stop. The wrapper must expose that runtime at the same path to every driver subprocess. Manifest contents and existing approval semantics are not changed.

`desktop_only=True` requires a private daemon and explicit owned private Wayland runtime/display socket, checks endpoint identity, and refuses the host Wayland endpoint. Native-Wayland enablement must be explicit (`CUA_DRIVER_RS_ENABLE_WAYLAND=1`). Host display/control environment must be explicitly replaced or unset. Existing Hermes permission configuration still determines standard/bounded/unrestricted mode; private standard-mode daemons are supported.

Desktop-only permits `app="screen"` capture and coordinate/key/text input after a successful desktop capture. It refuses explicit foreign PIDs, window IDs, app names, and AX selectors rather than silently retargeting. App/window listing returns empty lists without host process discovery; per-app foreground attestation and AX operations are unavailable. Native desktop input may require `delivery_mode="foreground"`; existing approval scope still applies. `theme` is a driver theme ID, not a CSS color.

`allow_input` is an optional live zero-argument callable checked before every input RPC, including CLI fallback. It must return literal `True`; false/exception pauses input while allowing capture. This lets owners implement viewer takeover without granting new permissions.

## Teardown and verification

Removal/replacement retires prior Cua transports by exact lease identity without modifying approvals. Remove before tearing down owner-managed jobs/compositor. Keep attachment across ordinary turn ends; the plugin owns durable session/alias mapping and removal on finalization/unload.

Regression command (isolated temporary `HERMES_HOME`, real imports):

```sh
scripts/run_tests.sh tests/tools/test_computer_use*.py tests/tools/test_cua_session_execution.py tests/tools/test_terminal*.py tests/tools/test_process_registry*.py tests/tools/test_file*.py tests/tools/test_local*.py tests/hermes_cli/test_session_execution.py -q
```

The new tests execute real local shells/PTYs and Cua subprocess transports against an explicitly labeled protocol fixture, including wrapper ordering, secret removal, exact env unsets, private standard mode, scoped socket path, manifest copying, takeover, rejection of foreign targets, and revocation. They do not claim real desktop pixel/input validation; run the external compositor/driver containment integration separately.
