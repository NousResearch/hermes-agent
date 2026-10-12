"""Contract tests for the ``hermes-gateway-ops`` optional skill.

The skill is prose: it teaches operators a diagnosis ladder whose *facts*
(drain-marker staleness rules, state-file field names, log markers, lifecycle
verbs) must keep matching the code they describe. Nothing else catches that
drift, so this module pins the load-bearing claims:

* the skill ships with resolvable frontmatter and its referenced files exist
* every ``related_skills`` entry resolves to a real skill in this repo
* the drain-marker rules the skill states match ``gateway.drain_control``
* the state-file fields and lifecycle states it names exist in the gateway
* the log markers it quotes are still emitted by the gateway

No network, no gateway process: pure source inspection (stdlib + pytest).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "devops" / "hermes-gateway-ops"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE = SKILL_DIR / "references" / "platform-lifecycle-and-dashboard-access.md"

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8-sig")


def _frontmatter() -> dict[str, str]:
    match = FRONTMATTER.match(_skill_text())
    assert match, "SKILL.md must start with YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "#")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _repo_text(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8-sig")


def test_skill_files_present():
    assert SKILL_MD.is_file()
    assert REFERENCE.is_file(), "the skill links this reference by name"


def test_frontmatter_is_complete_and_conventional():
    fields = _frontmatter()
    assert fields["name"] == SKILL_DIR.name
    assert len(fields["description"]) <= 60
    assert fields["description"].endswith(".")
    for key in ("version", "author", "license", "platforms"):
        assert fields.get(key), f"frontmatter must declare {key}"


def test_related_skills_resolve():
    text = _skill_text()
    match = re.search(r"related_skills:\s*\[(.*?)\]", text, re.DOTALL)
    assert match, "frontmatter must declare related_skills"
    for name in (n.strip() for n in match.group(1).split(",")):
        if not name:
            continue
        found = list(REPO.glob(f"skills/**/{name}/SKILL.md")) + list(
            REPO.glob(f"optional-skills/**/{name}/SKILL.md")
        )
        assert found, f"related skill {name!r} does not exist in this repo"


def test_referenced_files_exist():
    for rel in re.findall(r"`(references/[\w.-]+\.md)`", _skill_text()):
        assert (SKILL_DIR / rel).is_file(), f"{rel} is linked but missing"


def test_drain_marker_facts_match_drain_control():
    """The skill quotes the marker filename, epoch shape and the one-hour expiry."""
    skill = _skill_text() + REFERENCE.read_text(encoding="utf-8-sig")
    drain = _repo_text("gateway/drain_control.py")

    assert '".drain_request.json"' in drain
    assert ".drain_request.json" in skill

    # Same-epoch markers expire after DRAIN_REQUEST_MAX_AGE_SECONDS; the skill
    # tells the operator "an hour", so pin the constant to that claim.
    const = re.search(r"DRAIN_REQUEST_MAX_AGE_SECONDS\s*=\s*([\d.]+)", drain)
    assert const, "drain_control must define the staleness bound"
    assert float(const.group(1)) == 3600.0, "skill documents a one-hour expiry"
    assert re.search(r"expire\s+after\s+an\s+hour", skill) or "an hour" in skill

    # Both recovery helpers named by the skill must exist.
    for helper in ("clear_drain_request", "drain_requested"):
        assert f"def {helper}(" in drain, f"drain_control lost {helper}()"
        assert helper in skill


def test_gateway_state_fields_and_lifecycle_states_exist():
    """Every state-file field the skill tells operators to read is real."""
    skill = REFERENCE.read_text(encoding="utf-8-sig") + _skill_text()
    status = _repo_text("gateway/status.py")
    status_and_cli = status + _repo_text("hermes_cli/gateway.py")

    for field in (
        "gateway_state", "served_profiles", "restart_requested",
        "active_agents", "platforms", "updated_at",
    ):
        assert f'"{field}"' in status, f"gateway_state lost the {field!r} field"
        assert field in skill, f"skill claims {field!r} but never mentions it"

    # The lifecycle ladder lives in the CLI's status reader as well as the writer.
    for state in ("running", "degraded", "starting", "draining", "startup_failed"):
        assert f'"{state}"' in status_and_cli, f"{state!r} is not a gateway state"
        assert state in skill


def test_log_markers_are_still_emitted():
    """The skill teaches operators to search for these exact strings."""
    run = _repo_text("gateway/run.py")
    assert "as a planned --replace takeover" in run
    assert "did not exit after SIGTERM, sending SIGKILL" in run

    turn_recovery = _repo_text("agent/turn_recovery.py")
    assert "API failed after" in turn_recovery
    # The attempts variable is renamed between releases; the operator-visible
    # prefix is the contract, not the identifier.
    assert re.search(r"API failed after \{\w+\} retries", turn_recovery)

    watchers = _repo_text("gateway/kanban_watchers.py")
    assert "another gateway already holds the dispatcher" in watchers


def test_lifecycle_verbs_named_by_skill_exist():
    """`gateway restart|stop|start`, `--replace`, and `migrate --multiplex`."""
    cli = _repo_text("hermes_cli/gateway.py")
    for verb in ('"restart"', '"stop"', '"start"'):
        assert verb in cli, f"hermes gateway lost the {verb} verb"
    assert "--replace" in cli
    assert "migrate" in cli

    # The multiplexer's cross-profile attach/refusal wording the skill quotes.
    assert "already owns this host" in _repo_text("gateway/run.py")


def test_heartbeat_interval_claim_matches_shutdown_watchdog():
    """The skill says the heartbeat is rewritten every 30s on a healthy loop."""
    watchdog = _repo_text("gateway/shutdown_watchdog.py")
    assert "state/gateway.heartbeat" in watchdog
    skill = _skill_text()
    assert re.search(r"every 30s", skill)
    interval = re.search(r"HEARTBEAT[A-Z_]*\s*=\s*([\d.]+)", watchdog)
    if interval:
        assert float(interval.group(1)) == 30.0


def test_no_machine_local_paths_or_secrets():
    for path in (SKILL_MD, REFERENCE):
        text = path.read_text(encoding="utf-8-sig")
        assert not re.search(r"/home/(?!runner\b)[a-z0-9_-]+/", text)
        assert "/opt/data" not in text, "host-local paths must not leak upstream"
        assert "PRELVAS" not in text.replace("PRELVAS94", "")


@pytest.mark.parametrize("field", ["telegram", "discord", "slack"])
def test_platform_examples_are_real_platforms(field):
    """Example platform names in the skill must exist as platform adapters."""
    assert (REPO / "plugins" / "platforms" / field / "adapter.py").is_file()
