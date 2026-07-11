"""Child process used by the NLS-184 external-approval integration test.

This intentionally talks only to the proposed concrete FD ABI. Its stdout is
one small, non-secret completion marker; approval records belong on the
supplied record FD.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

from tools import approval as approval_module


COMMAND = "printf '%s\\n' nls-184-injected-execution"
MARKER = "nls-184-external-approval-process"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("request", "consume", "replay", "fd-probe"))
    parser.add_argument("--grant-fd", type=int)
    parser.add_argument("--record-fd", type=int)
    parser.add_argument("--verification-key")
    parser.add_argument("--session-id")
    parser.add_argument("--hermes-home")
    parser.add_argument("--probe-fds", nargs=2, type=int)
    return parser.parse_args()


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def main() -> int:
    args = _parse_args()
    if args.role == "fd-probe":
        grant_fd, record_fd = args.probe_fds
        print(json.dumps({"grant": _fd_is_open(grant_fd), "records": _fd_is_open(record_fd)}))
        return 0

    os.environ["HERMES_EXEC_ASK"] = "1"
    os.environ["HERMES_EXTERNAL_APPROVAL_MODE"] = "exact-once"
    os.environ["HERMES_HOME"] = args.hermes_home
    os.environ.pop("HERMES_INTERACTIVE", None)
    os.environ.pop("HERMES_GATEWAY_SESSION", None)
    os.environ.pop("HERMES_YOLO_MODE", None)

    token = approval_module.set_current_session_key(args.session_id)
    approval_module.configure_external_approval_fd_protocol(
        grant_input_fd=args.grant_fd,
        record_output_fd=args.record_fd,
        verification_key=base64.b64decode(args.verification_key),
    )
    try:
        result = approval_module.check_all_command_guards(COMMAND, "local")
        print(json.dumps({
            "approved": bool(result["approved"]),
            "executions": int(bool(result["approved"])),
            "marker": MARKER,
            "role": args.role,
        }, sort_keys=True))
        return 0
    finally:
        approval_module.clear_external_approval_fd_protocol()
        approval_module.reset_current_session_key(token)


if __name__ == "__main__":
    raise SystemExit(main())
