"""Evidence harness for the review-skill injection fix.

The defect: a review run was force-loaded a review skill the assignee's profile could
not resolve, and the worker died at INIT rather than continuing without it.

Runs REAL code from the tree named on the command line, against REAL profile homes built in a temp
farm — no monkeypatching of the behaviour under test.

    python verify_injected_skill.py <tree> preload <home> <injected 0|1>
    python verify_injected_skill.py <tree> readiness
    python verify_injected_skill.py <tree> resolve <profile> <name> [<name>...]
    python verify_injected_skill.py <tree> gate <lane-home> <present|absent>
    python verify_injected_skill.py <tree> cardskill <lane-home>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
SUB = sys.argv[2] if len(sys.argv) > 2 else ""
REAL_HOME = Path(os.environ.get("REAL_HERMES_HOME", str(Path.home() / ".hermes")))
sys.path.insert(0, str(TREE))


def make_lane_home(root: Path, lane: str, donor: Path, *, with_skill: bool) -> Path:
    """A real profile home for *lane*: its donor skills as symlinks, minus sdlc-review."""
    home = root / lane
    skills = home / "skills"
    skills.mkdir(parents=True)
    (home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    donor_skills = donor / lane / "skills"
    if donor_skills.is_dir():
        for entry in sorted(donor_skills.iterdir()):
            if entry.name == "sdlc-review":
                continue
            try:
                (skills / entry.name).symlink_to(entry.resolve(), target_is_directory=True)
            except OSError:
                pass
    if with_skill:
        target = skills / "sdlc-review"
        target.mkdir()
        (target / "SKILL.md").write_text(
            "---\nname: sdlc-review\ndescription: review doctrine.\n---\n\n# review\n",
            encoding="utf-8",
        )
    return home


def section_preload(home: str, injected: bool) -> None:
    os.environ["HERMES_HOME"] = home
    os.environ.pop("HERMES_KANBAN_ADVISORY_SKILLS", None)
    if injected:
        os.environ["HERMES_KANBAN_ADVISORY_SKILLS"] = "sdlc-review"
    from hermes_cli.oneshot import _build_preloaded_skills_prompt

    try:
        prompt = _build_preloaded_skills_prompt(["sdlc-review"])
        result = {"outcome": "continued", "prompt": bool(prompt)}
    except Exception as exc:  # noqa: BLE001 - the crash IS the observation
        result = {"outcome": "raised", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({
        "section": "preload", "tree": str(TREE), "home": home,
        "advisory_marker": injected, "only_skill": "sdlc-review", **result,
    }))


def section_readiness() -> None:
    os.environ.pop("HERMES_HOME", None)
    os.environ.pop("HERMES_KANBAN_ADVISORY_SKILLS", None)
    from hermes_cli import kanban_db_dispatch as kbd

    report = kbd.review_skill_readiness()
    print(json.dumps({
        "section": "readiness", "tree": str(TREE),
        "injected_skills": list(kbd.review_injected_skills()),
        "profiles": report,
        "unresolved": [row["profile"] for row in report if row["missing"]],
    }, indent=2))


def section_gate(lane_home: str, present: bool) -> None:
    """Real dispatcher tick: real home, real config load, real card, recording spawn."""
    tmp = Path(tempfile.mkdtemp(prefix="gate-"))
    fake_home = tmp / "fakehome"
    farm = fake_home / ".hermes"
    (farm / "profiles").mkdir(parents=True)
    shutil.copytree(lane_home, farm / "profiles" / "reviewer", symlinks=True)
    os.environ["HOME"] = str(fake_home)
    os.environ["HERMES_HOME"] = str(farm)
    # Point every board lookup at the throwaway farm below: the inherited HERMES_KANBAN_DB/HERMES_KANBAN_BOARD
    # name THIS worker's live board, which the delegated-child fence rightly refuses to mutate.
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        os.environ.pop(key, None)
    os.environ.pop("HERMES_KANBAN_ADVISORY_SKILLS", None)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    captured: list[dict] = []

    def spawn(task, workspace):
        captured.append({
            "skills": list(task.skills or []),
            "advisory_skills": list(getattr(task, "advisory_skills", ()) or ()),
        })
        return None

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="gate probe", assignee="reviewer")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.request_review(
            conn, task_id, summary="ready", expected_run_id=claimed.current_run_id,
        )
        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        events = [row[0] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]
        comments = [row[0] for row in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]
    print(json.dumps({
        "section": "gate", "tree": str(TREE), "lane_owns_skill": present,
        "spawned": [[t[0], t[1]] for t in result.spawned],
        "worker_args": captured,
        "events": events,
        "card_comment": next((c for c in comments if "does not resolve" in c), None),
    }, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)


def section_resolve(profile: str, names: list[str]) -> None:
    """Whether *profile*'s lane resolves each *name* — the dispatcher's own gate predicate."""
    os.environ.pop("HERMES_KANBAN_ADVISORY_SKILLS", None)
    from hermes_cli.kanban_db_dispatch import _profile_skill_resolvable, lane_profile_home

    home = lane_profile_home(profile)
    print(json.dumps({
        "section": "resolve", "tree": str(TREE), "profile": profile, "profile_home": home,
        "resolved": {name: bool(_profile_skill_resolvable(home, name)) for name in names},
    }, indent=2))


def section_cardskill(lane_home: str) -> None:
    """Real dispatcher tick for a READY card naming a skill its lane does not own (class fix).

    The card's own skill list used to be equally fatal: the name reached the worker's preload
    loader unchecked and a run died at INIT with ``Unknown skill(s): <name>``, parking the card.
    """
    tmp = Path(tempfile.mkdtemp(prefix="cardskill-"))
    fake_home = tmp / "fakehome"
    farm = fake_home / ".hermes"
    (farm / "profiles").mkdir(parents=True)
    shutil.copytree(lane_home, farm / "profiles" / "worker", symlinks=True)
    os.environ["HOME"] = str(fake_home)
    os.environ["HERMES_HOME"] = str(farm)
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        os.environ.pop(key, None)
    os.environ.pop("HERMES_KANBAN_ADVISORY_SKILLS", None)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    captured: list[dict] = []

    def spawn(task, workspace):
        captured.append({
            "skills": list(task.skills or []),
            "advisory_skills": list(getattr(task, "advisory_skills", ()) or ()),
        })
        return None

    wanted = ["lane-absent-skill", "another-absent-skill"]
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="card skill probe", assignee="worker", skills=wanted)
        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        events = [row[0] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]
        comments = [row[0] for row in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()]
    print(json.dumps({
        "section": "cardskill", "tree": str(TREE), "card_requested": wanted,
        "spawned": [[t[0], t[1]] for t in result.spawned],
        "worker_args": captured,
        "held": captured == [{"skills": wanted, "advisory_skills": wanted}],
        "events": events,
        "card_comment": next((c for c in comments if "do not resolve" in c), None),
    }, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    if SUB == "preload":
        section_preload(sys.argv[3], sys.argv[4] == "1")
    elif SUB == "readiness":
        section_readiness()
    elif SUB == "resolve":
        section_resolve(sys.argv[3], sys.argv[4:])
    elif SUB == "cardskill":
        section_cardskill(sys.argv[3])
    elif SUB == "gate":
        section_gate(sys.argv[3], sys.argv[4] == "present")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
