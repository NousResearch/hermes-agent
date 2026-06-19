"""CLI for Pydantic-only investment-assistant HITL workflows."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from .pydantic_resume import (
    CandidateTriageHitlState,
    create_candidate_triage_hitl_state_from_files,
    load_candidate_triage_hitl_state,
    resume_candidate_triage_hitl_from_file,
    save_candidate_triage_hitl_state,
    start_candidate_triage_hitl_from_files,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "create-triage-state":
        return _create_triage_state(args)
    if args.command == "resume-triage":
        return _resume_triage(args)
    if args.command == "show":
        return _show(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-pydantic-hitl",
        description="Run investment-assistant Pydantic HITL flows without Hermes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-triage-state",
        help="Create a waiting candidate-triage HITL state.",
    )
    create.add_argument("--discovery", required=True, help="Theme discovery artifact JSON path.")
    create.add_argument("--lightweight", required=True, help="Futu lightweight enrichment artifact JSON path.")
    create.add_argument(
        "--plan",
        help=(
            "Existing candidate_triage_plan JSON path. If omitted, the CLI runs "
            "the Pydantic triage planning agent first."
        ),
    )
    create.add_argument("--output", required=True, help="Output HITL state JSON path.")
    create.add_argument("--json", action="store_true", help="Print the full state JSON to stdout.")
    create.add_argument(
        "--timeout-seconds",
        type=int,
        default=_env_int("IA_HITL_TIMEOUT_SECONDS", 900),
        help="Wall-clock timeout for agent work. Use 0 to disable.",
    )
    create.add_argument("--trace", action="store_true", help="Enable PydanticAI trace logging to stderr.")

    resume = subparsers.add_parser(
        "resume-triage",
        help="Resume a waiting candidate-triage HITL state.",
    )
    resume.add_argument("--state", required=True, help="Waiting HITL state JSON path.")
    resume.add_argument("--output", required=True, help="Completed HITL state JSON path.")
    resume.add_argument("--option-id", default="", help="Strategy option id to use.")
    resume.add_argument("--answer", default="", help="Natural-language user answer, e.g. '选 2'.")
    resume.add_argument("--modifications", default="", help="Free-form strategy modifications.")
    resume.add_argument(
        "--must-include",
        nargs="*",
        default=[],
        help="Symbols that must be considered for deep research.",
    )
    resume.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Symbols to exclude when evidence boundary allows.",
    )
    resume.add_argument("--json", action="store_true", help="Print the full completed state JSON to stdout.")
    resume.add_argument(
        "--timeout-seconds",
        type=int,
        default=_env_int("IA_HITL_TIMEOUT_SECONDS", 900),
        help="Wall-clock timeout for agent work. Use 0 to disable.",
    )
    resume.add_argument("--trace", action="store_true", help="Enable PydanticAI trace logging to stderr.")
    resume.add_argument("--deep-min", type=int, help="Override IA_TRIAGE_DEEP_MIN for this run.")
    resume.add_argument("--deep-max", type=int, help="Override IA_TRIAGE_DEEP_MAX for this run.")

    show = subparsers.add_parser("show", help="Show a compact HITL state summary.")
    show.add_argument("--state", required=True, help="HITL state JSON path.")
    show.add_argument("--json", action="store_true", help="Print the full state JSON to stdout.")

    return parser


def _create_triage_state(args: argparse.Namespace) -> int:
    _configure_run(args)
    _progress("create-triage-state: loading artifacts")
    try:
        with _operation_timeout(args.timeout_seconds, "create-triage-state"):
            if args.plan:
                state = create_candidate_triage_hitl_state_from_files(
                    discovery_path=args.discovery,
                    lightweight_path=args.lightweight,
                    plan_path=args.plan,
                    output_path=args.output,
                )
            else:
                _progress("create-triage-state: running triage planning agent")
                state = start_candidate_triage_hitl_from_files(
                    discovery_path=args.discovery,
                    lightweight_path=args.lightweight,
                    output_path=args.output,
                )
    except TimeoutError as exc:
        _progress(f"create-triage-state: {exc}")
        return 124
    _progress("create-triage-state: completed")
    _print_state(state, output_path=args.output, emit_json=args.json)
    return 0


def _resume_triage(args: argparse.Namespace) -> int:
    _configure_run(args)
    _progress("resume-triage: loading state and running candidate triage agent")
    try:
        with _operation_timeout(args.timeout_seconds, "resume-triage"):
            state = resume_candidate_triage_hitl_from_file(
                state_path=args.state,
                output_path=args.output,
                option_id=args.option_id,
                answer=args.answer,
                modifications=args.modifications,
                must_include_symbols=args.must_include,
                exclude_symbols=args.exclude,
            )
    except TimeoutError as exc:
        _progress(f"resume-triage: {exc}")
        return 124
    _progress("resume-triage: completed")
    _print_state(state, output_path=args.output, emit_json=args.json)
    return 0


def _show(args: argparse.Namespace) -> int:
    state = load_candidate_triage_hitl_state(args.state)
    _print_state(state, output_path=args.state, emit_json=args.json)
    return 0


def _print_state(state: CandidateTriageHitlState, *, output_path: str | Path, emit_json: bool = False) -> None:
    if emit_json:
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"status: {state.status}")
    print(f"state: {state.state}")
    print(f"session_id: {state.session_id}")
    print(f"theme: {state.theme}")
    print(f"market: {state.market}")
    print(f"output: {output_path}")
    if state.status == "waiting_for_human":
        print(f"prompt: {state.prompt_to_user}")
        print("options:")
        for index, option in enumerate(state.candidate_triage_plan.strategy_options, start=1):
            print(
                f"  {index}. {option.option_id} | {option.name} | "
                f"deep={option.deep_research_total} watch={option.expected_watchlist_count}"
            )
        return

    if state.selection:
        print(f"selected_option_id: {state.selection.selected_option_id}")
        if state.selection.must_include_symbols:
            print("must_include: " + ", ".join(state.selection.must_include_symbols))
        if state.selection.exclude_symbols:
            print("exclude: " + ", ".join(state.selection.exclude_symbols))
    if state.candidate_triage:
        print(f"deep_count: {len(state.candidate_triage.deep_enrichment_queue)}")
        print(f"watchlist_count: {len(state.candidate_triage.watchlist)}")
        print(f"deferred_count: {len(state.candidate_triage.deferred)}")
        print(f"rejected_count: {len(state.candidate_triage.rejected)}")


def _configure_run(args: argparse.Namespace) -> None:
    if getattr(args, "trace", False):
        os.environ["IA_PYDANTIC_TRACE"] = "1"
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    deep_min = getattr(args, "deep_min", None)
    deep_max = getattr(args, "deep_max", None)
    if deep_min is not None:
        os.environ["IA_TRIAGE_DEEP_MIN"] = str(deep_min)
    if deep_max is not None:
        os.environ["IA_TRIAGE_DEEP_MAX"] = str(deep_max)


@contextlib.contextmanager
def _operation_timeout(seconds: int, label: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
