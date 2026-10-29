"""Session-bound tmux supervision with explicit application adapters."""

import argparse
import json
import sys

from . import omp, supervision


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tmux-supervise", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--workspace", required=True)
    prepare.add_argument("--tmux-session", required=True)
    prepare.add_argument("--state-root")
    prepare.add_argument("--run-id")
    prepare.add_argument("--adapter", choices=("command", "omp"), default="command")
    launch = commands.add_parser(
        "launch", help="Monitor any command's process lifecycle"
    )
    launch.add_argument("--run-dir", required=True)
    launch.add_argument(
        "--command-file", required=True, help="JSON argv array, not shell text"
    )
    launch.add_argument("--tmux-executable", default="tmux")
    launch_omp = commands.add_parser("launch-omp", help="OMP lifecycle-hook adapter")
    launch_omp.add_argument("--run-dir", required=True)
    launch_omp.add_argument("--prompt-file", required=True)
    launch_omp.add_argument("--omp-executable", default="omp")
    launch_omp.add_argument("--tmux-executable", default="tmux")
    launch_omp.add_argument("--model")
    launch_omp.add_argument("--thinking", choices=omp.THINKING_LEVELS)
    launch_omp.add_argument("--append-system-prompt", metavar="FILE")
    launch_omp.add_argument("--canary", action="store_true")
    watch = commands.add_parser("watch")
    watch.add_argument("--run-dir", required=True)
    watch.add_argument("--timeout", type=float, default=300)
    status = commands.add_parser("status")
    status.add_argument("--run-dir", required=True)
    internal = commands.add_parser("_run-command", help=argparse.SUPPRESS)
    internal.add_argument("--run-dir", required=True)
    internal.add_argument("--command-file", required=True)
    internal.add_argument("--executable")
    internal.add_argument("--command-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in {"launch", "_run-command"}:
            from . import process_adapter

            if command == "_run-command":
                arguments = process_adapter.read_command(
                    args["command_file"],
                    expected_sha256=args["command_sha256"],
                    executable=args["executable"],
                )
                return process_adapter.run_command(args["run_dir"], arguments)
            result = process_adapter.launch(**args)
        else:
            result = {
                "prepare": supervision.prepare,
                "launch-omp": omp.launch,
                "watch": supervision.watch,
                "status": supervision.status,
            }[command](**args)
    except (supervision.TUIError, OSError) as exc:
        print(
            json.dumps({
                "error": str(exc)
                if isinstance(exc, supervision.TUIError)
                else "filesystem_or_process_error"
            }),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
