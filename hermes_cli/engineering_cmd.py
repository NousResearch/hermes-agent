"""Operator-facing CLI for the sequential engineering workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.i18n import t


def _read_checks(path: Path) -> tuple:
    from agent.engineering_execution import CheckSpec

    if path.stat().st_size > 64 * 1024:
        raise ValueError("checks file is too large")
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, list) or not 1 <= len(document) <= 32:
        raise ValueError("checks must be a nonempty list of at most 32 entries")
    checks = []
    for row in document:
        if not isinstance(row, dict) or row.keys() != {"id", "argv", "timeout"}:
            raise ValueError("invalid check schema")
        check_id, argv, timeout = row["id"], row["argv"], row["timeout"]
        if (
            not isinstance(check_id, str)
            or not check_id
            or len(check_id) > 128
            or not isinstance(argv, list)
            or not 1 <= len(argv) <= 32
            or any(
                not isinstance(arg, str) or not arg or len(arg) > 2048 for arg in argv
            )
            or sum(len(arg) for arg in argv) > 16_384
            or type(timeout) not in (int, float)
            or not 0 < timeout <= 600
        ):
            raise ValueError("invalid check entry")
        checks.append(CheckSpec(check_id, tuple(argv), timeout))
    if len({check.check_id for check in checks}) != len(checks):
        raise ValueError("duplicate check id")
    return tuple(checks)


def _catalogue() -> dict:
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    return build_model_options_payload(
        load_picker_context(),
        explicit_only=True,
        refresh=True,
    )


def _run_workflow(**kwargs):
    from agent.engineering_runner import run_project_workflow

    return run_project_workflow(**kwargs)


def _pick_assignments() -> dict[str, dict[str, str]] | None:
    """Use the same provider/model catalogue and chooser as Hermes auxiliary pickers."""
    from hermes_cli.main_provider_setup import _prompt_provider_choice

    if not sys.stdin.isatty():
        return None
    catalogue = _catalogue()
    providers = [
        row
        for row in catalogue.get("providers", [])
        if row.get("authenticated") is True
        and row.get("slug") not in ("auto", "moa")
        and row.get("models")
    ]
    if not providers:
        return None
    assignments = {}
    for stage in ("planner", "worker", "reviewer"):
        stage_label = t(f"engineering.{stage}")
        index = _prompt_provider_choice(
            [row.get("name") or row["slug"] for row in providers],
            title=t("engineering.select_provider", stage=stage_label),
        )
        if index is None:
            return None
        row = providers[index]
        models = list(row["models"])
        index = _prompt_provider_choice(
            models,
            title=t("engineering.select_model", stage=stage_label),
        )
        if index is None:
            return None
        assignments[stage] = {"provider": row["slug"], "model": models[index]}
    return assignments


def run_cli(args: argparse.Namespace) -> int:
    """Render stable status/reason codes; avoid logging task or provider secrets."""
    try:
        checks = _read_checks(Path(args.checks_file))
        assignments = _pick_assignments()
        if assignments is None:
            print(
                json.dumps({
                    "status": "BLOCKED",
                    "reason": "model_selection_unavailable",
                })
            )
            return 2
        result = _run_workflow(
            objective=args.objective,
            assignments=assignments,
            workspace=Path(args.workspace),
            checks=checks,
            backend=args.backend,
            image=args.image or "",
        )
    except Exception:
        print(
            json.dumps({
                "status": "BLOCKED",
                "reason": "invalid_configuration_or_backend",
            })
        )
        return 2
    print(json.dumps(result.__dict__, ensure_ascii=False))
    return 0 if result.status == "DONE" else 2


def build_parser(subparsers) -> None:
    parser = subparsers.add_parser("engineering", help=t("engineering.command_help"))
    parser.add_argument(
        "--objective", required=True, help=t("engineering.objective_help")
    )
    parser.add_argument(
        "--workspace", required=True, help=t("engineering.workspace_help")
    )
    parser.add_argument(
        "--checks-file", required=True, help=t("engineering.checks_help")
    )
    parser.add_argument(
        "--backend",
        choices=("docker", "native"),
        default="docker",
        help=t("engineering.backend_help"),
    )
    parser.add_argument("--image", help=t("engineering.image_help"))
    parser.set_defaults(func=run_cli)
