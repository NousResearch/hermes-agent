"""One-tick, opt-in CLI edge for governed conductor campaigns."""

from __future__ import annotations

import json
from pathlib import Path

from conductor.config import resolve_conductor_config
from conductor.engine import Conductor
from conductor.launcher import TmuxLauncher
from conductor.runtime import plan_from_config
from conductor.store import ConductorStore
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home


_DEFINITION_FIELDS = {"campaign_id", "cwd", "mutable_manifest", "steps"}
_MAX_DEFINITION_BYTES = 1_048_576


def build_conductor_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "conductor",
        help="Run exactly one governed campaign tick",
        description="Run one deterministic tick from a bounded JSON campaign definition.",
    )
    parser.add_argument("definition", help="Path to a bounded campaign JSON definition")
    parser.set_defaults(func=cmd_conductor)


def _load_definition(path_value: str) -> dict:
    path = Path(path_value).expanduser()
    try:
        if path.stat().st_size > _MAX_DEFINITION_BYTES:
            raise SystemExit("conductor definition exceeds 1 MiB")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load conductor definition: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("conductor definition must be a JSON object")
    unknown = set(value) - _DEFINITION_FIELDS
    if unknown:
        raise SystemExit(
            f"unsupported field in conductor definition: {sorted(unknown)[0]}"
        )
    return value


def cmd_conductor(args) -> int:
    config = load_config()
    resolved = resolve_conductor_config(config)
    if not resolved["enabled"]:
        raise SystemExit(
            "conductor is disabled; set conductor.enabled: true in config.yaml"
        )
    definition = _load_definition(args.definition)
    plan = plan_from_config(definition, config)
    state_path = Path(str(resolved["state_path"])).expanduser()
    if not state_path.is_absolute():
        state_path = get_hermes_home() / state_path
    store = ConductorStore(state_path)
    store.create_campaign(plan)
    result = Conductor(
        store,
        TmuxLauncher(),
        tick_lease_seconds=resolved["tick_lease_seconds"],
    ).tick(plan.campaign_id)
    campaign = store.get_campaign(plan.campaign_id)
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "conductor_turns": campaign.conductor_turns,
                "result": result.value,
                "state": campaign.state,
                "step_index": campaign.step_index,
            },
            sort_keys=True,
        )
    )
    return 0
