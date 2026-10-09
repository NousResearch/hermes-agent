"""The public ``hermes update`` command boundary, split from the CLI facade.

Names owned by ``hermes_cli.main`` are resolved there at call time. This keeps the historical
patch seams used by update tests and avoids a module import cycle.
"""

import os
import subprocess
import sys

from hermes_cli.update_receipt import update_receipt_scope


def _print_update_action_terminal_marker(exit_code: int) -> None:
    """Persist a terminal Desktop action result after the updater has finalized its receipt."""
    action_id = os.environ.get("HERMES_ACTION_ID", "")
    if len(action_id) != 32 or any(char not in "0123456789abcdef" for char in action_id):
        return

    from hermes_cli.update_cmd import _log_only_write
    from hermes_cli.update_receipt import committed_success

    if exit_code == 0:
        if not committed_success():
            return
        line = f"=== hermes-update completed {action_id} ==="
    else:
        line = f"=== hermes-update failed {action_id} exit={exit_code} ==="

    # The update body restores stdio before this boundary runs. Persist the result in the root log
    # so a Desktop backend restart cannot erase it.
    _log_only_write(line + "\n")
    try:
        print(line)
    except BrokenPipeError:
        pass


def _cmd_update_body(args):
    """Update Hermes Agent: hangup protection + update lock around ``_cmd_update_impl``."""
    from hermes_cli import main as _main

    # Marks this frame as the CURRENT updater for _old_updater.in_historical_update().
    _hermes_current_updater_frame = True
    from hermes_cli.update_owning_install import retarget_to_owning_install

    retarget_to_owning_install(_main.PROJECT_ROOT)
    if _main._update_preflight_handled(args):
        return
    gateway_mode = getattr(args, "gateway", False)

    _update_io_state = _main._install_hangup_protection(gateway_mode=gateway_mode)
    from hermes_cli.update_lock import UPDATE_EXIT_CONCURRENT, UpdateLock, describe_holder

    _update_lock = UpdateLock(install_root=_main.PROJECT_ROOT)
    if not _update_lock.acquire():
        print(describe_holder(_update_lock.holder))
        _main._finalize_update_output(_update_io_state)
        from hermes_cli.update_cmd_common import _record_stop

        _record_stop("lock_held", without_receipt="refused")
        sys.exit(UPDATE_EXIT_CONCURRENT)

    from hermes_cli.update_cmd import _cmd_update_impl
    from pm import InstallError

    def _custody_refusal() -> str | None:
        custody = sys.modules.get("hermes_cli.update_custody")
        return custody.refusal_notice() if custody is not None else None

    try:
        _cmd_update_impl(args, gateway_mode=gateway_mode)
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        refusal = _custody_refusal()
        print(refusal or f"✗ Update failed: {exc}")
        _main._finalize_update_receipt(1, f"{type(exc).__name__}: {exc}")
        if gateway_mode:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code

            _write_gateway_update_exit_code(False)
        raise SystemExit(1) from exc
    except SystemExit as update_exit:
        code = update_exit.code if isinstance(update_exit.code, int) else 1
        if code and (refusal := _custody_refusal()):
            print(refusal)
        _main._finalize_update_receipt(code, f"sys.exit({code})")
        if gateway_mode and code:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code

            _write_gateway_update_exit_code(False)
        raise
    except BaseException as update_exc:
        if gateway_mode:
            from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code

            _write_gateway_update_exit_code(False)
        _main._finalize_update_receipt(1, f"{type(update_exc).__name__}: {update_exc}")
        raise
    else:
        from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON

        _main._finalize_update_receipt(0, COMMAND_BOUNDARY_STOP_REASON)
    finally:
        _update_lock.release()
        _main._finalize_update_output(_update_io_state)


@update_receipt_scope()
def cmd_update(args):
    """Run the updater and persist a terminal Desktop action result after receipt finalization."""
    _hermes_current_updater_frame = True
    exit_code = 0
    try:
        return _cmd_update_body(args)
    except SystemExit as exc:
        exit_code = exc.code if type(exc.code) is int else 1
        raise
    except KeyboardInterrupt:
        exit_code = 130
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        _print_update_action_terminal_marker(exit_code)
