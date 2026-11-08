"""Older stores through the conversation verbs, with main's store as the oracle.

Each fixture is built by main (``tests/persistence_sqlite/upgrade/_fixtures.py``) and copied per chain. A chain runs
three steps, each in its own process: the subject opens the copy and reads the fixture's rows and writes a session
through the conversation verbs (sessions, messages, search, usage), main reopens the copy writable, reads what the
subject wrote and appends, and the subject reads that back. The subject is the backend, or main's ``SessionDB`` through
its own API as the control; both chains must give the same answers and leave the same schema.

As a script, ``python test_conversation_upgrade.py <side> <step> <json-args>`` runs one step with PYTHONPATH at the
tree that side imports; it imports no store module before the step runs.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

FIXTURES = ("current", "v21", "v24", "fts_v0_inline", "fts_v2", "fts_stale", "fts_trigram_dropped")
SUBJECTS = ("m1", "backend")
SESSION = "conv-session"
T0 = 1_700_000_000.0
FIXTURE_QUERIES = ("payment", "迁移内容")
_RESULT_PREFIX = "CONVERSATION-RESULT "
_STEP_TIMEOUT_S = 180


# ── Steps, run in the subprocess ─────────────────────────────────────────────────────────────────────────────


class _M1Side:
    """Main's SessionDB through its own API, answers shaped like the backend's."""

    def __init__(self, db: str):
        from hermes_state import SessionDB

        self.db = SessionDB(db_path=Path(db))

    def get(self, session_id):
        row = self.db.get_session(session_id)
        return None if row is None else [row["model"], row["input_tokens"], row["output_tokens"],
                                         row["api_call_count"], row["message_count"], row["title"]]

    def transcript(self, session_id):
        return [[row["role"], row["content"]] for row in self.db.get_messages(session_id)]

    def search(self, query):
        return [[row["id"], row["session_id"], row["role"], row["snippet"]] for row in self.db.search_messages(query)]

    def route(self, session_id):
        row = self.db.get_recent_session_model_route(session_id)
        return None if row is None else [row["model"], row["billing_provider"] or None, row["billing_base_url"] or None,
                                         row["billing_mode"] or None, row["api_call_count"]]

    def create(self, session_id):
        self.db.create_session(session_id, "cli", model="model-z")

    def append(self, session_id, role, content, timestamp):
        self.db.append_message(session_id, role, content=content, timestamp=timestamp)

    def account(self, session_id):
        self.db.queue_token_counts(session_id, input_tokens=11, output_tokens=3, api_call_count=1, model="model-z",
                                   billing_provider="prov", billing_base_url="https://api.example.test", source="cli")
        return self.db.flush_token_counts()

    def set_title(self, session_id, title):
        return self.db.set_session_title(session_id, title)

    def close(self):
        self.db.close()


class _BackendSide:
    """The backend's conversation repositories over the copy."""

    def __init__(self, db: str):
        from persistence import ProfileScope
        from persistence.sqlite import SQLiteBackend

        scope = ProfileScope(profile_dir=Path(db).parent, name="upgrade")
        self.backend = SQLiteBackend({scope.hermes_home_key: Path(db)})
        self.store = self.backend.open(scope)

    def get(self, session_id):
        record = self.store.sessions.get(session_id)
        return None if record is None else [record.model, record.input_tokens, record.output_tokens,
                                            record.api_call_count, record.message_count, record.title]

    def transcript(self, session_id):
        return [[row.role, row.content] for row in self.store.messages.read_transcript(session_id)]

    def search(self, query):
        return [[hit.message_id, hit.session_id, hit.role, hit.snippet]
                for hit in self.store.search.search_messages(query)]

    def route(self, session_id):
        usage = self.store.usage.recent_route(session_id)
        if usage is None:
            return None
        route = usage.route
        return [route.model, route.billing_provider, route.billing_base_url, route.billing_mode, usage.api_call_count]

    def create(self, session_id):
        from persistence.conversation.sessions import NewSession

        self.store.sessions.create(NewSession(session_id=session_id, source="cli", model="model-z"))

    def append(self, session_id, role, content, timestamp):
        from persistence.conversation.messages import NewMessage

        self.store.messages.append(session_id, NewMessage(role=role, content=content, timestamp=timestamp))

    def account(self, session_id):
        from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta

        self.store.usage.queue_usage(session_id, UsageDelta(
            tokens=TokenCounts(input_tokens=11, output_tokens=3), api_call_count=1, source="cli",
            route=BillingRoute(model="model-z", billing_provider="prov", billing_base_url="https://api.example.test")))
        return self.store.usage.flush_usage()

    def set_title(self, session_id, title):
        return self.store.sessions.set_title(session_id, title)

    def close(self):
        self.backend.close()


_SIDES = {"m1": _M1Side, "backend": _BackendSide}


def _step_conversation(side: str, db: str) -> dict:
    """Read the fixture's rows, then write one session, its messages, a usage delta and a title."""
    store = _SIDES[side](db)
    try:
        out = {"fixture_get": store.get("cli-1"), "fixture_transcript": store.transcript("cli-1"),
               "fixture_search": {query: store.search(query) for query in FIXTURE_QUERIES},
               "fixture_route": store.route("cli-1")}
        store.create(SESSION)
        store.append(SESSION, "user", "conv zebra payment", T0)
        store.append(SESSION, "assistant", "conv reply", T0 + 1)
        out["flushed"] = store.account(SESSION)
        out["titled"] = store.set_title(SESSION, "conv title")
        out.update(get=store.get(SESSION), transcript=store.transcript(SESSION), search=store.search("zebra"),
                   route=store.route(SESSION))
    finally:
        store.close()
    return out


def _step_m1_reopen(side: str, db: str) -> dict:
    """Main reopens writable, reads the subject's session and appends to it."""
    store = _M1Side(db)
    try:
        out = {"get": store.get(SESSION), "transcript": store.transcript(SESSION), "search": store.search("zebra"),
               "route": store.route(SESSION), "fixture_get": store.get("cli-1")}
        store.append(SESSION, "user", "m1 followup quokka", T0 + 2)
    finally:
        store.close()
    return out


def _step_read_back(side: str, db: str) -> dict:
    store = _SIDES[side](db)
    try:
        return {"get": store.get(SESSION), "transcript": store.transcript(SESSION), "search": store.search("quokka")}
    finally:
        store.close()


_STEPS = {"conversation": _step_conversation, "m1_reopen": _step_m1_reopen, "read_back": _step_read_back}


# ── The tests ───────────────────────────────────────────────────────────────────────────────────────────────


def _oracle():
    # Imported on use: as a script this module may run on a pinned oracle tree that carries no such package.
    from tests.persistence_sqlite.upgrade import _oracle as oracle

    return oracle


def _run(side: str, step: str, env: dict, **args) -> dict:
    """Run one step of this module in a fresh interpreter importing *side*'s tree."""
    from tests.persistence_sqlite._legacy_store import oracle_tree

    tree = oracle_tree() if side == "m1" else _oracle().WORKTREE
    proc = subprocess.run(
        [sys.executable, str(Path(__file__)), side, step, json.dumps(args)],
        env={**env, "PYTHONPATH": str(tree)}, cwd=env["HERMES_HOME"], capture_output=True, text=True, check=False,
        timeout=_STEP_TIMEOUT_S,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    assert proc.returncode == 0 and lines, (
        f"{side} {step} exited {proc.returncode}\n--- stdout\n{proc.stdout[-3000:]}\n--- stderr\n{proc.stderr[-6000:]}")
    return json.loads(lines[-1][len(_RESULT_PREFIX):])


def _chain(built: Path, chain_dir: Path, subject: str, env: dict) -> dict:
    db = _oracle().copy_store(built, chain_dir)
    env = {**env, "HERMES_HOME": str(db.parent / "home")}
    out = {"conversation": _run(subject, "conversation", env, db=str(db)),
           "m1_reopen": _run("m1", "m1_reopen", env, db=str(db)),
           "read_back": _run(subject, "read_back", env, db=str(db))}
    snapshot = _oracle().snapshot(db)
    out["sqlite_master"] = snapshot["sqlite_master"]
    out["schema_version"] = snapshot["tables"]["schema_version"]
    out["fts_stamp"] = [row[1] for row in snapshot["tables"]["state_meta"] if row[0] == "fts_storage_version"]
    return out


@pytest.mark.parametrize("fixture", FIXTURES)
def test_conversation_verbs_on_an_older_store_answer_like_m1(fixture, tmp_path):
    env = dict(os.environ)
    built = tmp_path / "fixture"
    (built / "home").mkdir(parents=True)
    _oracle().run_step("m1", "build", {**env, "HERMES_HOME": str(built / "home")}, db=str(built / "state.db"),
                       name=fixture)
    with ThreadPoolExecutor(max_workers=len(SUBJECTS)) as pool:
        futures = {subject: pool.submit(_chain, built, tmp_path / subject, subject, env) for subject in SUBJECTS}
        chains = {subject: future.result() for subject, future in futures.items()}
    m1, backend = chains["m1"], chains["backend"]
    for key in m1:
        assert backend[key] == m1[key], key

    written = backend["conversation"]
    assert written["flushed"] is True and written["titled"] is True
    seen = backend["m1_reopen"]
    assert seen["get"] == ["model-z", 11, 3, 1, 2, "conv title"]
    assert seen["transcript"] == [["user", "conv zebra payment"], ["assistant", "conv reply"]]
    assert [hit[1] for hit in seen["search"]] == [SESSION]
    assert seen["route"] == ["model-z", "prov", "https://api.example.test", None, 1]
    assert seen["fixture_get"] == written["fixture_get"]
    back = backend["read_back"]
    assert back["transcript"][-1] == ["user", "m1 followup quokka"] and back["get"][4] == 3
    assert [hit[1] for hit in back["search"]] == [SESSION]


if __name__ == "__main__":
    _side, _step, _args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    _result = _STEPS[_step](_side, **_args)
    print(_RESULT_PREFIX + json.dumps(_result), flush=True)
