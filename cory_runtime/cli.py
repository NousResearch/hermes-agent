from __future__ import annotations

import argparse
import logging

from hermes_logging import setup_logging

from .config import CoryWorkerConfig
from .control_plane import ControlPlaneClient
from .executor import HermesCoryExecutor
from .worker import CoryControlPlaneWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-cory-worker",
        description="Run Cory request-interpretation jobs against a Cory control plane using Hermes as the execution runtime.",
    )
    parser.add_argument("--control-plane-base-url")
    parser.add_argument("--internal-api-token")
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--poll-interval-seconds", type=float)
    parser.add_argument("--max-backoff-seconds", type=float)
    parser.add_argument("--max-completion-attempts", type=int)
    parser.add_argument("--request-timeout-seconds", type=float)
    parser.add_argument("--once", action="store_true")
    return parser


def load_config(args: argparse.Namespace) -> CoryWorkerConfig:
    try:
        config = CoryWorkerConfig.from_env()
    except ValueError:
        base_url = args.control_plane_base_url
        token = args.internal_api_token
        if not base_url or not token:
            raise
        config = CoryWorkerConfig(
            control_plane_base_url=base_url.rstrip("/"),
            internal_api_token=token,
        )

    if args.control_plane_base_url:
        config.control_plane_base_url = args.control_plane_base_url.rstrip("/")
    if args.internal_api_token:
        config.internal_api_token = args.internal_api_token
    if args.model:
        config.model = args.model
    if args.provider:
        config.provider = args.provider
    if args.poll_interval_seconds is not None:
        config.poll_interval_seconds = args.poll_interval_seconds
    if args.max_backoff_seconds is not None:
        config.max_backoff_seconds = args.max_backoff_seconds
    if args.max_completion_attempts is not None:
        config.max_completion_attempts = args.max_completion_attempts
    if args.request_timeout_seconds is not None:
        config.request_timeout_seconds = args.request_timeout_seconds
    if args.once:
        config.once = True

    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args)

    setup_logging(mode="cli")
    logging.getLogger(__name__).info(
        "Starting hermes-cory-worker against %s",
        config.control_plane_base_url,
    )

    with ControlPlaneClient(
        base_url=config.control_plane_base_url,
        token=config.internal_api_token,
        timeout_seconds=config.request_timeout_seconds,
    ) as client:
        executor = HermesCoryExecutor(
            model=config.model,
            provider=config.provider,
            max_completion_attempts=config.max_completion_attempts,
        )
        worker = CoryControlPlaneWorker(
            client=client,
            executor=executor,
            poll_interval_seconds=config.poll_interval_seconds,
            max_backoff_seconds=config.max_backoff_seconds,
        )

        if config.once:
            worker.run_once()
            return

        worker.run_forever()


if __name__ == "__main__":
    main()
