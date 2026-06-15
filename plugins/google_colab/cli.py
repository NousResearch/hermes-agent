from __future__ import annotations

from typing import Any

from . import core


def _print(payload: dict[str, Any]) -> None:
    print(core.to_json(payload))


def register_cli(subparser) -> None:
    actions = subparser.add_subparsers(dest="google_colab_action")

    status_parser = actions.add_parser("status", help="Show Colab CLI status.")
    status_parser.add_argument("--probe-sessions", action="store_true")
    status_parser.add_argument("--auth", choices=["adc", "oauth2"], default="adc")
    status_parser.add_argument("--config-path")
    status_parser.add_argument("--allow-windows-native", action="store_true")

    sessions_parser = actions.add_parser("sessions", help="List active Colab sessions.")
    sessions_parser.add_argument("--auth", choices=["adc", "oauth2"], default="adc")
    sessions_parser.add_argument("--config-path")
    sessions_parser.add_argument("--allow-windows-native", action="store_true")

    run_parser = actions.add_parser("run", help="Run a script via colab run.")
    run_parser.add_argument("script_path")
    run_parser.add_argument("script_args", nargs="*")
    run_parser.add_argument("--gpu", choices=sorted(core.ALLOWED_GPUS))
    run_parser.add_argument("--tpu", choices=sorted(core.ALLOWED_TPUS))
    run_parser.add_argument("-s", "--session-name")
    run_parser.add_argument("--keep", action="store_true")
    run_parser.add_argument("--confirm", action="store_true")
    run_parser.add_argument("--auth", choices=["adc", "oauth2"], default="adc")
    run_parser.add_argument("--config-path")
    run_parser.add_argument("--timeout-seconds", type=int, default=core.DEFAULT_TIMEOUT_SECONDS)
    run_parser.add_argument("--allow-windows-native", action="store_true")

    sft_parser = actions.add_parser("sft-template", help="Write a TRL SFT Colab job template.")
    sft_parser.add_argument("--output-path", required=True)
    sft_parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    sft_parser.add_argument("--dataset-name", default="trl-lib/Capybara")
    sft_parser.add_argument("--dataset-split", default="train")
    sft_parser.add_argument("--output-dir", default="./hermes-sft-adapter")
    sft_parser.add_argument("--max-steps", type=int, default=120)
    sft_parser.add_argument("--push-to-hub-repo", default="")

    subparser.set_defaults(func=google_colab_command)


def google_colab_command(args: Any) -> int:
    action = getattr(args, "google_colab_action", None)
    if action == "status":
        _print(
            core.status_payload(
                {
                    "probe_sessions": args.probe_sessions,
                    "auth": args.auth,
                    "config_path": args.config_path,
                    "allow_windows_native": args.allow_windows_native,
                }
            )
        )
        return 0
    if action == "sessions":
        _print(
            core.sessions_payload(
                {
                    "auth": args.auth,
                    "config_path": args.config_path,
                    "allow_windows_native": args.allow_windows_native,
                }
            )
        )
        return 0
    if action == "run":
        _print(
            core.run_job(
                {
                    "script_path": args.script_path,
                    "args": args.script_args,
                    "gpu": args.gpu,
                    "tpu": args.tpu,
                    "keep": args.keep,
                    "session_name": args.session_name,
                    "auth": args.auth,
                    "config_path": args.config_path,
                    "timeout_seconds": args.timeout_seconds,
                    "confirmed": args.confirm,
                    "allow_windows_native": args.allow_windows_native,
                }
            )
        )
        return 0
    if action == "sft-template":
        _print(
            core.write_sft_template(
                {
                    "output_path": args.output_path,
                    "model_id": args.model_id,
                    "dataset_name": args.dataset_name,
                    "dataset_split": args.dataset_split,
                    "output_dir": args.output_dir,
                    "max_steps": args.max_steps,
                    "push_to_hub_repo": args.push_to_hub_repo,
                }
            )
        )
        return 0
    print("usage: hermes google-colab {status,sessions,run,sft-template}")
    return 2
