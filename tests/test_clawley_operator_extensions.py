from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clawley_daily_brief import build_daily_brief
from scripts.maintainer_sweep_to_kanban_proposals import build_kanban_proposals


def test_maintainer_sweep_to_kanban_proposals_is_read_only() -> None:
    summary = {
        "repo": "acme/widgets",
        "records": 3,
        "mutation_allowed": False,
        "action_state": "proposal",
        "items": [
            {"kind": "issue", "number": 1, "title": "Fix docs typo", "recommendation": "safe_candidate", "labels": ["docs"]},
            {"kind": "pr", "number": 2, "title": "Dangerous auth change", "recommendation": "needs_human", "labels": ["security"]},
        ],
    }

    proposals = build_kanban_proposals(summary)

    assert proposals["write_performed"] is False
    assert proposals["mutation_allowed"] is False
    assert proposals["repo"] == "acme/widgets"
    assert proposals["proposals"][0]["status"] == "proposal"
    assert proposals["proposals"][0]["suggested_priority"] == "normal"
    assert proposals["proposals"][1]["requires_human_gate"] is True
    assert proposals["next_safe_action"] == "review_proposals_before_kanban_import"


def test_daily_brief_summarises_sources_without_actions() -> None:
    payload = build_daily_brief(
        {
            "quantos": {"pat_fx": {"setups_total": 114, "resolved_expectancy_r": 0.117}},
            "kanban": {"blocked": 2, "ready": 4},
            "cron": {"failed_last_24h": 1},
            "gateway": {"errors_last_24h": 0},
        }
    )

    assert payload["schema"] == "clawley_daily_brief.v1"
    assert payload["write_performed"] is False
    assert payload["safety_flags"]["read_only"] is True
    assert payload["sections"]["quantos"]["pat_fx"]["setups_total"] == 114
    assert payload["recommendations"][0] == "review_blocked_kanban_items"
    assert payload["recommendations"][1] == "inspect_failed_cron_jobs"
    assert "Geen automatische acties uitgevoerd" in payload["markdown"]
