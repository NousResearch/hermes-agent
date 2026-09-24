"""Write-approval pending store and skill-delete guards hold against path-shaped names.

Two operator/agent-facing name lookups must stay inside the thing they name:

* ``/memory`` and ``/skills`` ``approve|reject|diff <id>`` (#119997): the id names a record staged under
  ``<HERMES_HOME>/pending/<subsystem>/``. A ``../`` or absolute id must not read, delete or apply any
  file outside that directory. Driven through the real ``tui_gateway`` stdio backend (``slash.exec``),
  after real staged writes: the fake model calls the ``memory`` and ``skill_manage`` tools with
  ``memory.write_approval`` / ``skills.write_approval`` on, so the store directories exist exactly as
  they do for a user.
* ``skill_manage delete`` (#120528) must refuse a pinned skill and the essential ``hermes-agent`` skill
  whatever spelling names it: bare (``my-skill``) or categorized (``research/my-skill``). The pin is set
  with the real ``hermes curator pin`` CLI; the deletes are real tool calls in real agent turns on the
  same backend (``hermes chat -q`` does not offer ``skill_manage``: one-shot runs hide it).

Every breach asserts on disk (victim bytes, skill dir, MEMORY.md) and on the command output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.e2e.core.security import _helpers as H
from tests.e2e.core.security._traversal import digest, result_json, tool_text
from tests.e2e.core.tenancy._helpers import TuiBackend
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX absolute-path spellings and process groups")

_ISSUE_STORE = "#119997 pending id is used as a path: ../ or absolute ids reach files outside pending/<subsystem>/"
_ISSUE_DELETE = "#120528 skill_manage delete skips the pin/essential guard for a category/name spelling"

# scenario -> (slash command, where the victim lives). "home" victims are <HERMES_HOME>/auth.json spelled
# ``../../auth``; "outside" victims live outside the Hermes home and are spelled by absolute path.
TRAVERSALS: dict[str, tuple[str, str]] = {
    "memory_reject_dotdot": ("memory reject", "home"),
    "memory_approve_dotdot": ("memory approve", "home"),
    "memory_reject_absolute": ("memory reject", "outside"),
    "memory_approve_absolute": ("memory approve", "outside"),
    "skills_reject_dotdot": ("skills reject", "home"),
    "skills_diff_dotdot": ("skills diff", "home"),
    "skills_diff_absolute": ("skills diff", "outside"),
}
DELETES: dict[str, tuple[str, str]] = {  # scenario -> (name given to skill_manage, skill dir under skills/)
    "pinned_by_category": ("research/my-skill", "research/my-skill"),
    "essential_by_category": ("autonomous-ai-agents/hermes-agent", "autonomous-ai-agents/hermes-agent"),
}
KNOWN: dict[str, str] = {**dict.fromkeys(TRAVERSALS, _ISSUE_STORE), **dict.fromkeys(DELETES, _ISSUE_DELETE)}

_STORE_CONFIG = "memory:\n  memory_enabled: true\n  write_approval: true\nskills:\n  write_approval: true\n"
_SKILL = "---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n{body}\n"


def _skill_md(name: str, body: str = "Body.") -> str:
    return _SKILL.format(name=name, desc=f"e2e skill {name}", body=body)


def _tool_results(srv: FakeLLMServer) -> list[dict[str, Any]]:
    return [result_json(tool_text(m.get("content")))
            for m in srv.main_requests()[-1]["messages"] if m.get("role") == "tool"]


def _slash(b: TuiBackend, sid: str, command: str) -> str:
    reply = b.call("slash.exec", {"session_id": sid, "command": command}, timeout=90)
    assert "result" in reply, f"slash.exec {command!r} failed: {reply.get('error')}"
    return str(reply["result"].get("output", ""))


# ── pending store (#119997) ─────────────────────────────────────────────────


@dataclass
class Store:
    b: TuiBackend
    sid: str
    hermes_home: Path
    outside: Path
    memory_ids: list[str]
    memory_texts: list[str]
    skill_ids: list[str]
    skill_texts: list[str]

    def pending_ids(self, subsystem: str) -> set[str]:
        d = self.hermes_home / "pending" / subsystem
        return {p.stem for p in d.glob("*.json")} if d.is_dir() else set()

    def memory_blob(self) -> str:
        d = self.hermes_home / "memories"
        return "".join(p.read_text(encoding="utf-8", errors="replace") for p in d.glob("*")) if d.is_dir() else ""


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    root = tmp_path_factory.mktemp("store")
    home, outside = root / "home", root / "outside"
    outside.mkdir()
    key = H.canary("sk-store")
    memory_texts = [H.canary(f"staged-memory-{i}") for i in range(3)]
    skill_texts = [H.canary(f"staged-skill-{i}") for i in range(2)]
    script: list[Any] = [ToolCall("memory", {"action": "add", "target": "memory", "content": t}) for t in memory_texts]
    script += [ToolCall("skill_manage", {"action": "create", "name": f"staged-skill-{i}",
                                         "content": _skill_md(f"staged-skill-{i}", t)})
               for i, t in enumerate(skill_texts)]
    with FakeLLMServer([*script, Text("staged")], api_key=key) as srv:
        H.write_home(home / ".hermes", srv.base_url, api_key=key, config=_STORE_CONFIG)
        b = TuiBackend(home, root / "tui.log")
        try:
            sid = b.ok("session.create", {})["session_id"]
            b.turn(sid, "stage five writes")
            staged = _tool_results(srv)
            assert len(staged) == 5 and all(r.get("staged") for r in staged), f"writes were not staged: {staged}"
            ids = [r["pending_id"] for r in staged]
            s = Store(b, sid, home / ".hermes", outside, ids[:3], memory_texts, ids[3:], skill_texts)
            assert s.pending_ids("memory") == set(ids[:3]) and s.pending_ids("skills") == set(ids[3:]), \
                f"pending store on disk does not match the staged ids: {sorted((s.hermes_home / 'pending').rglob('*'))}"
            yield s
        finally:
            b.close()


def _plant_victim(s: Store, where: str) -> tuple[Path, str, str]:
    """A record-shaped JSON file outside the store: ``(path, id spelling, canary)``. Record-shaped so a
    traversal that reaches it is observable on every verb (reject deletes it, approve applies its payload,
    diff prints its summary)."""
    victim, spelled = ((s.hermes_home / "auth.json", "../../auth") if where == "home"
                       else (s.outside / "victim.json", str(s.outside / "victim")))
    mark = H.canary("victim")
    victim.write_text(json.dumps({"id": spelled, "subsystem": "memory", "summary": mark, "origin": "foreground",
                                  "created_at": 0, "payload": {"action": "add", "target": "memory", "content": mark}}),
                      encoding="utf-8")
    return victim, spelled, mark


@pytest.mark.parametrize("scenario", [H.known_param(n, KNOWN) for n in TRAVERSALS])
def test_pending_id_never_leaves_the_store(store: Store, scenario: str) -> None:
    command, where = TRAVERSALS[scenario]
    victim, spelled, mark = _plant_victim(store, where)
    before, pending_before = victim.read_bytes(), {k: store.pending_ids(k) for k in ("memory", "skills")}

    out = _slash(store.b, store.sid, f"/{command} {spelled}")

    assert out.strip(), f"/{command} returned no output"
    pending_after = {k: store.pending_ids(k) for k in ("memory", "skills")}
    checks = (
        (not victim.exists() or victim.read_bytes() != before, f"{victim.name} was deleted or rewritten"),
        (mark in out, "the command output carries the victim's content"),
        (mark in store.memory_blob(), "the victim's payload was applied to the memory store"),
        (pending_after != pending_before, f"real staged records changed: {pending_before} -> {pending_after}"),
    )
    if breaches := [msg for failed, msg in checks if failed]:
        raise H.BoundaryBreach(f"/{command} {spelled!r}: " + "; ".join(breaches) + f"\noutput: {out[:400]}")


def test_control_valid_reject_drops_only_that_record(store: Store) -> None:
    target, keep = store.memory_ids[0], set(store.memory_ids[2:])
    out = _slash(store.b, store.sid, f"/memory reject {target}")
    assert target in out and target not in store.pending_ids("memory"), f"valid reject did not drop {target}: {out}"
    assert keep <= store.pending_ids("memory"), "rejecting one record dropped another"
    assert store.memory_texts[0] not in store.memory_blob(), "a rejected write reached the memory store"


def test_control_valid_approve_applies_the_staged_write(store: Store) -> None:
    target = store.memory_ids[1]
    out = _slash(store.b, store.sid, f"/memory approve {target}")
    assert store.memory_texts[1] in store.memory_blob(), f"approved write missing from MEMORY.md; output: {out}"
    assert target not in store.pending_ids("memory"), "an applied record stayed pending"


def test_control_valid_diff_shows_the_staged_skill(store: Store) -> None:
    out = _slash(store.b, store.sid, f"/skills diff {store.skill_ids[0]}")
    assert store.skill_texts[0] in out, f"/skills diff did not render the staged content: {out[:400]}"


# ── skill delete guard (#120528) ────────────────────────────────────────────


@dataclass
class DeleteRun:
    results: dict[str, dict[str, Any]] = field(default_factory=dict)  # scenario -> tool result
    survived: dict[str, bool] = field(default_factory=dict)  # scenario -> skill dir present right after


# Executed in this order, one agent turn each, so a later delete can never mask an earlier outcome.
_DELETE_STEPS: tuple[tuple[str, str, str], ...] = (
    ("control_pinned_bare", "my-skill", "research/my-skill"),
    ("control_essential_bare", "hermes-agent", "autonomous-ai-agents/hermes-agent"),
    *((n, name, rel) for n, (name, rel) in DELETES.items()),
    ("control_unpinned_category", "research/free-skill", "research/free-skill"),
)


@pytest.fixture(scope="module")
def deletes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeleteRun]:
    root = tmp_path_factory.mktemp("deletes")
    home = root / "home"
    skills = home / ".hermes" / "skills"
    for rel in ("research/my-skill", "research/free-skill", "autonomous-ai-agents/hermes-agent"):
        (skills / rel).mkdir(parents=True)
        (skills / rel / "SKILL.md").write_text(_skill_md(rel.rsplit("/", 1)[1]), encoding="utf-8")
    key = H.canary("sk-delete")
    run = DeleteRun()
    with FakeLLMServer([], api_key=key) as srv:
        H.write_home(home / ".hermes", srv.base_url, api_key=key)
        pin = H.run_hermes(["curator", "pin", "my-skill"], home, timeout=90)
        assert pin.returncode == 0, f"hermes curator pin failed: {pin.stdout}\n{pin.stderr[-2000:]}"
        b = TuiBackend(home, root / "tui.log")
        try:
            sid = b.ok("session.create", {})["session_id"]
            for step, (scenario, name, rel) in enumerate(_DELETE_STEPS, start=1):
                srv.push(ToolCall("skill_manage", {"action": "delete", "name": name}), Text("ok"))
                b.turn(sid, f"delete {name}")
                results = _tool_results(srv)  # the session history carries every earlier step's result too
                assert len(results) == step, f"{scenario}: expected {step} tool results, got {results}"
                run.results[scenario] = results[-1]
                run.survived[scenario] = (skills / rel / "SKILL.md").exists()
        finally:
            b.close()
    yield run


@pytest.mark.parametrize("scenario", [H.known_param(n, KNOWN) for n in DELETES])
def test_delete_refuses_pinned_and_essential_by_category(deletes: DeleteRun, scenario: str) -> None:
    name, rel = DELETES[scenario]
    if not deletes.survived[scenario]:
        raise H.BoundaryBreach(f"skill_manage delete {name!r} removed skills/{rel}: {deletes.results[scenario]}")
    assert deletes.results[scenario].get("success") is False, deletes.results[scenario]


@pytest.mark.parametrize("scenario, guard", [("control_pinned_bare", "pinned"), ("control_essential_bare", "essential")])
def test_control_bare_name_delete_is_refused(deletes: DeleteRun, scenario: str, guard: str) -> None:
    res = deletes.results[scenario]
    assert deletes.survived[scenario], f"{scenario}: the skill was deleted: {res}"
    assert res.get("success") is False and guard in str(res.get("error", "")).lower(), res


def test_control_unpinned_delete_by_category_succeeds(deletes: DeleteRun) -> None:
    res = deletes.results["control_unpinned_category"]
    assert res.get("success") is True and not deletes.survived["control_unpinned_category"], res
