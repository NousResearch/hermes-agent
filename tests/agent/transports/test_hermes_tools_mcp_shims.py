"""Tests for the stateless memory/session_search shims in the hermes-tools
MCP server (#26567).

Natively `memory` and `session_search` are `_AGENT_LOOP_TOOLS`: the generic
dispatcher refuses them because they need live AIAgent state. The shims
supply that state statelessly — `load_on_disk_store()` per call for memory,
a read-only `SessionDB` + the calling session's id from env for
session_search — so an agent whose loop is owned by an external runtime
(claude-agent-sdk, codex app-server) regains both tools.

No `mcp` package required: the dispatch functions are plain module-level
callables; only `_build_server()` (not under test here) needs FastMCP.

Plant-the-failure discipline: the DB-missing path must yield an EXPLICIT
error (never a silently-empty result), and the refusal in
`handle_function_call` must remain intact for non-shim callers.
"""

import json

import pytest

from agent.transports.hermes_tools_mcp_server import (
    _stateless_shim_defs,
    dispatch_memory,
    dispatch_session_search,
)


@pytest.fixture()
def tmp_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # `set_current_session_id()` writes HERMES_SESSION_ID process-globally, so a
    # developer's own live session id would otherwise leak into the suite and make
    # the canonical-env tests below pass spuriously.
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_MCP_STATE_DB", raising=False)
    return home


class TestMemoryShim:
    def test_write_lands_in_canonical_memories_dir(self, tmp_hermes_home):
        out = json.loads(
            dispatch_memory(
                {"action": "add", "target": "memory", "content": "auth refactor merged to main"}
            )
        )
        assert out.get("success") is True
        memory_file = tmp_hermes_home / "memories" / "MEMORY.md"
        assert memory_file.exists()
        assert "auth refactor merged to main" in memory_file.read_text()

    def test_native_caps_enforced(self, tmp_hermes_home):
        # The shim reuses the native store: an oversized add must be rejected
        # with the native consolidation error, not silently truncated.
        out = json.loads(
            dispatch_memory({"action": "add", "target": "memory", "content": "x" * 5000})
        )
        assert out.get("success") is False
        assert "exceed" in json.dumps(out).lower()

    def test_batch_operations_supported(self, tmp_hermes_home):
        out = json.loads(
            dispatch_memory(
                {
                    "target": "memory",
                    "operations": [
                        {"action": "add", "content": "fact alpha"},
                        {"action": "add", "content": "fact beta"},
                    ],
                }
            )
        )
        assert out.get("success") is True
        content = (tmp_hermes_home / "memories" / "MEMORY.md").read_text()
        assert "fact alpha" in content and "fact beta" in content

    def test_fails_closed_when_external_provider_configured(
        self, tmp_hermes_home, monkeypatch
    ):
        # #26604 precondition: a shim write cannot mirror through
        # MemoryProvider hooks (no MemoryManager in this subprocess), so a
        # configured external backend must refuse the dispatch — silent
        # store divergence is the failure this prevents.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"provider": "honcho"}}
        )
        out = json.loads(
            dispatch_memory({"action": "add", "target": "memory", "content": "x"})
        )
        assert out.get("success") is False
        assert "honcho" in out.get("error", "")
        assert not (tmp_hermes_home / "memories" / "MEMORY.md").exists()

        batch = json.loads(
            dispatch_memory(
                {"target": "memory", "operations": [{"action": "add", "content": "y"}]}
            )
        )
        assert batch.get("success") is False

    def test_builtin_provider_value_is_not_external(self, tmp_hermes_home, monkeypatch):
        # Control (non-vacuous): 'builtin' means the on-disk store — the
        # guard must not fire, and the write must land.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"provider": "builtin"}}
        )
        out = json.loads(
            dispatch_memory({"action": "add", "target": "memory", "content": "kept"})
        )
        assert out.get("success") is True
        assert "kept" in (tmp_hermes_home / "memories" / "MEMORY.md").read_text()

    def test_shim_unregistered_when_external_provider_configured(
        self, tmp_hermes_home, monkeypatch
    ):
        # Registration-level twin of the dispatch guard: the tool should not
        # even be offered to the model when it can only refuse.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config",
            lambda *a, **k: {"memory": {"memory_enabled": True, "provider": "mem0"}},
        )
        names = [name for name, _desc, _schema, _fn in _stateless_shim_defs()]
        assert "memory" not in names
        assert "session_search" in names


class TestSessionSearchShim:
    def _seed_db(self, path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=path)
        db.create_session("sess-hist-1", source="telegram")
        db.append_message("sess-hist-1", "user", "when did we merge the auth refactor?")
        db.append_message("sess-hist-1", "assistant", "The auth refactor merged on Thursday.")
        db.close()

    def test_search_returns_seeded_rows(self, tmp_hermes_home, monkeypatch):
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = dispatch_session_search({"query": "auth refactor"})
        assert "auth refactor" in out
        assert "sess-hist-1" in out

    def test_missing_db_yields_explicit_error(self, tmp_hermes_home, monkeypatch):
        # RED-first: a missing state DB must surface as an explicit error,
        # never as a silently-empty result set.
        monkeypatch.setenv(
            "HERMES_MCP_STATE_DB", str(tmp_hermes_home / "nope" / "state.db")
        )
        out = json.loads(dispatch_session_search({"query": "anything"}))
        assert out.get("success") is False
        assert "state DB" in out.get("error", "")

    def test_session_id_read_from_canonical_env(self, tmp_hermes_home, monkeypatch):
        # The shim must read the CANONICAL `HERMES_SESSION_ID` — the name Hermes
        # actually produces (`set_current_session_id` -> `_VAR_MAP` ->
        # `_inject_session_context_env` -> the HOST process's spawn env; a codex
        # MCP child additionally needs the entry to name it in `env_vars` — see
        # the `_SESSION_ID_ENV` note in the server module). A bespoke name has a
        # producer in NO launch path; the canonical name is delivered wherever
        # the host forwards or sets it, and exclusion stays fail-open (inactive)
        # where it doesn't.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        monkeypatch.setenv("HERMES_SESSION_ID", "sess-current-9")

        captured = {}
        import tools.session_search_tool as sst

        real = sst.session_search

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(sst, "session_search", spy)
        dispatch_session_search({"query": "auth"})
        assert captured.get("current_session_id") == "sess-current-9"

    def test_calling_session_excluded_via_production_producer(
        self, tmp_hermes_home, monkeypatch
    ):
        # Producer/consumer NAME agreement, proven through the REAL producer:
        # `set_current_session_id()` is what Hermes itself calls. Establishing the
        # precondition that way — rather than a bare `setenv` of the very name
        # under test — makes a name mismatch fail loudly here, which is how the
        # original defect survived review. Scope honestly: producer and shim share
        # this test process, so this pins the NAME contract, not delivery across a
        # host's process boundary (see the `_SESSION_ID_ENV` note for who
        # forwards it).
        from gateway.session_context import set_current_session_id
        from hermes_state import SessionDB

        db_path = tmp_hermes_home / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("sess-other-1", source="telegram")
        db.append_message(
            "sess-other-1", "assistant", "the auth refactor merged on Thursday"
        )
        db.create_session("sess-mine-2", source="telegram")
        db.append_message(
            "sess-mine-2", "assistant", "the auth refactor notes are mine"
        )
        db.close()
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))

        # setenv first purely so monkeypatch restores the environment at teardown
        # (the producer writes os.environ directly); the value under test is the
        # one written by the real producer on the very next line.
        monkeypatch.setenv("HERMES_SESSION_ID", "")
        set_current_session_id("sess-mine-2")

        out = dispatch_session_search({"query": "auth refactor", "limit": 10})
        assert "sess-other-1" in out
        assert "sess-mine-2" not in out

    def test_zero_hit_multiterm_query_relaxes_to_or(self, tmp_hermes_home, monkeypatch):
        # FTS5 ANDs terms: models write "topic word word word" queries and get
        # 0 hits for content that matches one distinctive term (observed live
        # twice). The shim retries ONCE with OR-joined terms, deterministic,
        # and annotates the result honestly.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = json.loads(
            dispatch_session_search({"query": "auth refactor deployment window"})
        )
        assert out.get("count", 0) >= 1
        assert out.get("relaxed_query") == "auth OR refactor OR deployment OR window"
        assert "sess-hist-1" in json.dumps(out)

    def test_explicit_fts_operators_are_never_relaxed(self, tmp_hermes_home, monkeypatch):
        # A query that already uses FTS operators is the caller's intent —
        # no second-guessing.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = json.loads(
            dispatch_session_search({"query": '"deployment window" OR rollout'})
        )
        assert out.get("count") == 0
        assert "relaxed_query" not in out

    def test_zero_hit_single_term_returns_honest_zero(self, tmp_hermes_home, monkeypatch):
        # Nothing to relax on a single term: an honest empty result, never a
        # fabricated one.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = json.loads(dispatch_session_search({"query": "kubernetes"}))
        assert out.get("count") == 0
        assert "relaxed_query" not in out

    def test_uninitialized_db_yields_explicit_error(self, tmp_hermes_home, monkeypatch):
        # Validator C3: a 0-byte state.db (crashed first init) passes the
        # exists() guard and used to return a SILENT success/count:0.
        db_path = tmp_hermes_home / "state.db"
        db_path.touch()  # present but uninitialized
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = json.loads(dispatch_session_search({"query": "anything"}))
        assert out.get("success") is False
        assert "not initialized" in out.get("error", "")

    def test_db_opened_read_only(self, tmp_hermes_home, monkeypatch):
        # The shim must never hand a writable DB handle to a model-facing
        # subprocess. SessionDB(read_only=True) attaches with mode=ro.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))

        captured = {}
        import agent.transports.hermes_tools_mcp_server as srv
        import hermes_state

        real = hermes_state.SessionDB

        class SpyDB(real):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(hermes_state, "SessionDB", SpyDB)
        dispatch_session_search({"query": "auth"})
        assert captured.get("read_only") is True


class TestShimRegistration:
    def test_both_shims_defined_by_default(self, tmp_hermes_home):
        names = [name for name, _desc, _schema, _fn in _stateless_shim_defs()]
        assert names == ["memory", "session_search"]

    def test_memory_shim_respects_config_disable(self, tmp_hermes_home, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"memory_enabled": False}}
        )
        names = [name for name, _desc, _schema, _fn in _stateless_shim_defs()]
        assert "memory" not in names
        assert "session_search" in names

    def test_shim_signatures_carry_the_registry_schema(self, tmp_hermes_home):
        # Pin of the schema-inference regression: FastMCP derives the served
        # schema from the handler's signature, so the signature synthesized
        # from the registry schema must expose the real parameters — never a
        # bare ``kwargs`` (pydantic renders that as a REQUIRED "kwargs"
        # field, failing EVERY call at the validation layer).
        from agent.transports.hermes_tools_mcp_server import (
            _signature_from_schema,
        )

        for name, _desc, schema, _fn in _stateless_shim_defs():
            sig, _annots = _signature_from_schema(schema)
            params = list(sig.parameters)
            assert "kwargs" not in params, f"{name} would serve an inferred schema"
            assert params, f"{name} signature is empty"
        shim_schemas = {n: s for n, _d, s, _f in _stateless_shim_defs()}
        mem_sig, _ = _signature_from_schema(shim_schemas["memory"])
        assert "target" in mem_sig.parameters
        ss_sig, _ = _signature_from_schema(shim_schemas["session_search"])
        assert "query" in ss_sig.parameters

    def test_agent_loop_refusal_stays_intact_for_other_callers(self):
        # The shims must NOT weaken the generic dispatcher: a stateless
        # handle_function_call("memory", ...) still refuses.
        from model_tools import handle_function_call

        out = handle_function_call("memory", {"action": "add", "content": "x"})
        assert "must be handled by the agent loop" in out
