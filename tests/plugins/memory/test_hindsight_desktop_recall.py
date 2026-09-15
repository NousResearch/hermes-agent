"""Desktop workstream-scoped recall (SPEC-INFRA-DESKRECALL-001, §3.8).

Desktop sessions carry chat_id=None/thread_id=None, so the Discord
thread-routing override never fired for them and recall fell back to the
global allowlist. The plugin now resolves, per recall call, the session cwd
(session ContextVar override OR terminal-scope fallback, §3.1) against
``desktop_context_root/<domain>/`` and applies that domain key's
``extra_tags`` as an ``any_strict`` filter (§3.2/§3.4) — with a fail-open
matrix that keeps every unrouted session byte-identical.

Upstream-extraction contract (Correction #2/#10): reflect forwards the
domain filter ONLY on an active Desktop domain match — unconfigured and
non-Desktop reflect stays byte-identical to upstream (no tags at all);
recall keeps the configured baseline filter.

These tests pin the outgoing-filter behavior through fake ``arecall`` /
``areflect`` clients capturing kwargs (sync-mode automatic injection is what
``MemoryManager.prefetch_all`` drives — it calls ``provider.prefetch()``),
the two-session interleaved isolation, the Discord regression controls, and
the recall_sync trips-closed guard.
"""

import contextvars
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import runtime_cwd
from plugins.memory.hindsight import HindsightMemoryProvider


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_cwd(tmp_path, monkeypatch):
    """No TERMINAL_CWD, and no session-cwd ContextVar binding leaks between tests."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    token = runtime_cwd._SESSION_CWD.set(runtime_cwd._UNSET)
    yield
    runtime_cwd._SESSION_CWD.reset(token)


def _fact(text, tags):
    return SimpleNamespace(text=text, tags=tags)


def _capturing_client(results=None, reflect_text="synthesized", recall_error=None):
    """Fake Hindsight client capturing outgoing arecall/areflect kwargs."""
    recall_calls = []
    reflect_calls = []

    def _arecall(**kwargs):
        recall_calls.append(dict(kwargs))
        if recall_error is not None:
            raise recall_error
        return SimpleNamespace(results=results or [])

    def _areflect(**kwargs):
        reflect_calls.append(dict(kwargs))
        if recall_error is not None:
            raise recall_error
        return SimpleNamespace(text=reflect_text)

    client = SimpleNamespace()
    client.arecall = AsyncMock(side_effect=_arecall)
    client.areflect = AsyncMock(side_effect=_areflect)
    client.recall_calls = recall_calls
    client.reflect_calls = reflect_calls
    return client


def _desktop_provider(tmp_path, monkeypatch, *, config=None, routing=None,
                      routing_raw=None, platform="desktop", thread_id="",
                      results=None, reflect_text="synthesized", recall_error=None,
                      drop_keys=()):
    """Initialized provider with a capturing fake client.

    ``routing`` (dict) is written as thread_routing.json; ``routing_raw``
    (str) writes arbitrary file content for malformed-table cases; neither
    means no routing file at all. ``drop_keys`` removes config keys after the
    merge so tests can exercise GENUINE key absence (vs. a null value).
    """
    cfg = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "recall_sync": True,
        "desktop_context_root": str(tmp_path / "desktop-context"),
    }
    cfg.update(config or {})
    for key in drop_keys:
        cfg.pop(key, None)
    cfg_path = tmp_path / "hindsight" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    routing_path = tmp_path / "hindsight" / "thread_routing.json"
    if routing is not None:
        routing_path.parent.mkdir(parents=True, exist_ok=True)
        routing_path.write_text(json.dumps(routing))
    if routing_raw is not None:
        routing_path.parent.mkdir(parents=True, exist_ok=True)
        routing_path.write_text(routing_raw)

    provider = HindsightMemoryProvider()
    kwargs = {"session_id": "test-session", "platform": platform}
    if thread_id:
        kwargs["thread_id"] = thread_id
    provider.initialize(**kwargs)
    client = _capturing_client(results=results, reflect_text=reflect_text,
                               recall_error=recall_error)
    provider._client = client
    return provider, client, tmp_path / "desktop-context"


def _bind(root, domain):
    """Bind the session-cwd ContextVar through the REAL setter (integration
    path, R2-3) and return the reset token."""
    return runtime_cwd.set_session_cwd(str(root / domain))


def _pack(root, *domains):
    root.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        (root / domain).mkdir()


# Distinctive nonempty baseline used across the fail-open matrix (R5 residual
# 3c): fail-open means the CONFIGURED filter survives verbatim, so the
# None-baseline shape ("tags" absent from outgoing kwargs) cannot be the only
# pinned behavior.
_BASELINE_CFG = {"recall_tags": ["baseline-tag"], "recall_tags_match": "all_strict"}
_BASELINE_TAGS = ["baseline-tag"]
_BASELINE_MATCH = "all_strict"


def _assert_baseline_kwargs(call):
    """The configured baseline filter reached the backend unchanged."""
    assert call["tags"] == _BASELINE_TAGS
    assert call["tags_match"] == _BASELINE_MATCH


def _assert_no_tag_kwargs(call):
    """Upstream reflect contract: NO tags/tags_match key reaches the
    backend unless a Desktop domain match actively fired (Correction #10a)."""
    assert "tags" not in call
    assert "tags_match" not in call


# ---------------------------------------------------------------------------
# §3.8.1 — outgoing-filter assertions (recall, reflect, tool path)
# ---------------------------------------------------------------------------


class TestOutgoingFilterDomain:
    ROUTING = {
        "infrastructure": {"extra_tags": ["infrastructure"]},
        "investments": {"extra_tags": ["investments"]},
    }

    def test_recall_forwards_domain_filter_and_passes_results_through(self, tmp_path, monkeypatch):
        mixed = _fact("shared infra+investments fact", ["infrastructure", "investments"])
        disjoint = _fact("investments-only fact", ["investments"])
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, results=[mixed, disjoint])
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            out = p._recall("icenova investments")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        kwargs = client.recall_calls[0]
        assert kwargs["tags"] == ["infrastructure"]
        assert kwargs["tags_match"] == "any_strict"
        # Disjoint-domain tags are never requested...
        assert "investments" not in kwargs["tags"]
        # ...and OR semantics live on the server: the plugin must NOT
        # post-filter a mixed-tag fact out (any_strict admits every fact
        # carrying >=1 domain tag, §3.5).
        assert [r.text for r in out] == [mixed.text, disjoint.text]
        # The stored baseline is left untouched (resolved per call, not stored).
        assert p._recall_tags is None
        assert p._recall_tags_match == "any"

    def test_reflect_entry_point_forwards_domain_filter(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure")
        p._prefetch_method = "reflect"
        token = _bind(root, "infrastructure")
        try:
            direct = p._reflect("direct question")
            text = p.prefetch("prefetch reflect question")  # sync-mode auto-injection
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert direct == "synthesized"
        assert client.reflect_calls[0]["tags"] == ["infrastructure"]
        assert client.reflect_calls[0]["tags_match"] == "any_strict"
        assert client.reflect_calls[1]["tags"] == ["infrastructure"]
        assert "synthesized" in text  # nonempty eligible injection through prefetch

    def test_tool_path_recall_provenance(self, tmp_path, monkeypatch):
        # Upstream tool-handler contract (Correction #10a): the explicit
        # hindsight_recall tool funnels through _recall(), so the ACTIVE
        # domain filter reaches the backend exactly as it does for
        # automatic injection — pinned on the captured outgoing kwargs.
        # The locally added "tags=..." provenance line is NOT an upstream
        # behavior and is not pinned here; the upstream result format
        # (numbered facts, or the empty-bank string) is.
        fact = _fact("infra status fact", ["infrastructure"])
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, results=[fact])
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            raw = p.handle_tool_call("hindsight_recall", {"query": "where are we"})
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        payload = json.loads(raw)
        assert payload["result"] == "1. infra status fact"
        assert client.recall_calls[0]["tags"] == ["infrastructure"]

    def test_empty_bank_positive_control(self, tmp_path, monkeypatch):
        # Absence of hits alone is NOT a success signal: recall must still be
        # CALLED with the domain filter and return zero results without error.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, results=[])
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            out = p._recall("anything")
            raw = p.handle_tool_call("hindsight_recall", {"query": "anything"})
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert out == []
        assert len(client.recall_calls) == 2
        assert all(c["tags"] == ["infrastructure"] for c in client.recall_calls)
        assert json.loads(raw)["result"] == "No relevant memories found."

    def test_reflect_empty_bank_positive_control(self, tmp_path, monkeypatch):
        # Mirror of the recall empty-bank control for the _reflect entry point
        # (§3.8.1): with the backend's reflect response configured EMPTY, the
        # entry point must still CALL the backend once with the domain filter
        # and return the empty result without error — distinct from the
        # timeout/error path, which raises and records no success signal.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, results=[],
            reflect_text="")
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            out = p._reflect("anything")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert out == ""
        assert len(client.reflect_calls) == 1
        assert client.reflect_calls[0]["tags"] == ["infrastructure"]
        assert client.reflect_calls[0]["tags_match"] == "any_strict"

    def test_active_route_equal_to_configured_baseline_still_activates_and_never_leaks(self, tmp_path, monkeypatch):
        # Correction #10d boundary: the active domain's (tags, mode) tuple
        # EQUALS the configured baseline. Activation must come from the
        # helper's per-call return flag — not from tuple comparison and not
        # from shared per-call mutable state. Discriminator: only an ACTIVE
        # match makes reflect forward tags; an identical baseline tuple on a
        # later, unrouted call must forward nothing at all.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            config={"recall_tags": ["infrastructure"], "recall_tags_match": "any_strict"})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        # Active-domain call: domain filter forwarded on BOTH entry points,
        # even though it is tuple-identical to the configured baseline.
        assert client.recall_calls[0]["tags"] == ["infrastructure"]
        assert client.recall_calls[0]["tags_match"] == "any_strict"
        assert client.reflect_calls[0]["tags"] == ["infrastructure"]
        assert client.reflect_calls[0]["tags_match"] == "any_strict"
        # Same provider, no session-cwd binding this time: recall keeps the
        # identical configured baseline, reflect forwards NOTHING — no
        # sticky activation leaked from the previous active evaluation.
        p._recall("q-unrouted")
        p._reflect("q-unrouted")
        assert client.recall_calls[1]["tags"] == ["infrastructure"]
        assert client.recall_calls[1]["tags_match"] == "any_strict"
        _assert_no_tag_kwargs(client.reflect_calls[1])


# ---------------------------------------------------------------------------
# §3.8.2 — two-session interleaved isolation + real setter binding
# ---------------------------------------------------------------------------


class TestTwoSessionIsolation:
    ROUTING = {
        "infrastructure": {"extra_tags": ["infrastructure"]},
        "investments": {"extra_tags": ["investments"]},
    }

    def test_interleaved_contexts_stay_isolated(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure", "investments")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)  # process cwd sits in a third, unrelated dir

        ctx_infra = contextvars.copy_context()
        ctx_infra.run(lambda: runtime_cwd.set_session_cwd(str(root / "infrastructure")))
        ctx_invest = contextvars.copy_context()
        ctx_invest.run(lambda: runtime_cwd.set_session_cwd(str(root / "investments")))

        ctx_infra.run(lambda: p._recall("infra question"))
        ctx_invest.run(lambda: p._recall("invest question"))
        ctx_infra.run(lambda: p._recall("infra question 2"))

        got = [c["tags"] for c in client.recall_calls]
        assert got == [["infrastructure"], ["investments"], ["infrastructure"]]
        assert all(c["tags_match"] == "any_strict" for c in client.recall_calls)

    def test_real_setter_binding_in_main_context(self, tmp_path, monkeypatch):
        # Integration path (R2-3): bind through the session-cwd ContextVar
        # setter in the caller's own context — not hand-set ContextVars only.
        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            tags, tags_match, activated = p._effective_recall_filter()
            p._recall("scoped question")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert tags == ["infrastructure"]
        assert tags_match == "any_strict"
        assert activated is True
        assert client.recall_calls[0]["tags"] == ["infrastructure"]


# ---------------------------------------------------------------------------
# §3.8.3 — fail-open matrix
# ---------------------------------------------------------------------------


class TestFailOpenMatrix:
    ROUTING = {
        "infrastructure": {"extra_tags": ["infrastructure"]},
        "investments": {"extra_tags": ["investments"]},
        # Active route for the symlink-escape test (R5 residual 3a): the
        # lexical link component must NOT route — only its resolved
        # (outside-root) target matters.
        "escape-link": {"extra_tags": ["escape-link"]},
    }

    def test_unbound_context_terminal_scope_wins_not_launch_dir(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure", "investments")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("TERMINAL_CWD", str(root / "infrastructure"))
        p._recall("q")
        # Terminal-scope fallback wins; the process launch dir is never consulted.
        assert client.recall_calls[0]["tags"] == ["infrastructure"]

    def test_cleared_context_no_terminal_cwd_global_baseline(self, tmp_path, monkeypatch):
        # Intentional None-baseline control (R5 residual 3c): unfiltered
        # baseline stays pinned as "no tags kwarg at all" alongside the
        # nonempty-baseline matrix above.
        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure")
        p._recall("q")
        assert "tags" not in client.recall_calls[0]
        assert p._recall_tags is None

    def test_nonexistent_session_override_is_final_over_valid_terminal_cwd(self, tmp_path, monkeypatch):
        # A NONEMPTY but nonexistent session override returns None and does NOT
        # fall through to a valid competing terminal cwd (§3.1 finality).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "investments")
        monkeypatch.setenv("TERMINAL_CWD", str(root / "investments"))
        token = runtime_cwd.set_session_cwd(str(tmp_path / "ghost-dir"))
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_outside_root(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        token = runtime_cwd.set_session_cwd(str(elsewhere))
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_cwd_equals_root(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = runtime_cwd.set_session_cwd(str(root))
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_unmatched_domain(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "unrouted-domain")
        token = runtime_cwd.set_session_cwd(str(root / "unrouted-domain"))
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_non_desktop_platform_recall_baseline_reflect_no_tags(self, tmp_path, monkeypatch):
        # Correction #10b outgoing-kwargs control: a NON-desktop platform
        # with nonempty configured recall_tags keeps the configured filter
        # on recall, while reflect omits tags and tags_match entirely
        # (upstream reflect contract — no Desktop domain match can fire).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, platform="cli", routing=self.ROUTING,
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])

    @pytest.mark.parametrize("entry", [{}, {"extra_tags": None}, {"extra_tags": []}])
    def test_empty_null_missing_tags_keep_baseline(self, tmp_path, monkeypatch, entry):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing={"infrastructure": entry},
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_blank_tags_keep_baseline(self, tmp_path, monkeypatch):
        # Correction #10a: recall keeps the configured baseline filter,
        # reflect omits BOTH tags and tags_match (no activation).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": ["  ", ""]}},
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])

    def test_root_key_absent_branch_inert(self, tmp_path, monkeypatch):
        # GENUINE absence (R5 residual 3b): config built with NO
        # desktop_context_root key at all — a null value is a separate case.
        # Correction #10b control with nonempty recall_tags: recall keeps
        # the configured filter; reflect forwards nothing at all.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            config=_BASELINE_CFG, drop_keys=("desktop_context_root",))
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])

    def test_root_key_null_value_branch_inert(self, tmp_path, monkeypatch):
        # Null VALUE (key present, set to None) is also inert — kept as its
        # own labeled case, distinct from genuine absence above.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            config={**_BASELINE_CFG, "desktop_context_root": None})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    @pytest.mark.parametrize("raw", ["{not valid json", "[1, 2]", "null"])
    def test_malformed_non_object_routing_table(self, tmp_path, monkeypatch, raw):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing_raw=raw, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_missing_routing_file(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=None, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_dormant_domain_route_is_absent(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": ["infrastructure"], "dormant": True}},
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_symlink_boundary_across_root_edge(self, tmp_path, monkeypatch):
        # "escape-link" has an ACTIVE route in the table (R5 residual 3a):
        # rejection must come from canonical resolve-based outside-root
        # detection — if cwd.resolve() were dropped, the lexical path WOULD
        # route and this test would fail.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        outside = tmp_path / "outside-real"
        outside.mkdir()
        link = root / "escape-link"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable on this host")
        token = runtime_cwd.set_session_cwd(str(link))
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        # Path.resolve() collapses the symlink: the cwd lands OUTSIDE the root.
        _assert_baseline_kwargs(client.recall_calls[0])

    def test_root_pointing_at_file_resolution_failure(self, tmp_path, monkeypatch):
        # Path-resolution failure: desktop_context_root points at a FILE, so no
        # cwd can resolve under it -> baseline, not a crash.
        root_file = tmp_path / "rootfile"
        root_file.write_text("x")
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            config={**_BASELINE_CFG, "desktop_context_root": str(root_file)})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])


# ---------------------------------------------------------------------------
# §3.8.3a — cwd-resolution exception paths (R5 residual 1)
# ---------------------------------------------------------------------------


class TestCwdResolutionFailOpen:
    ROUTING = {"infrastructure": {"extra_tags": ["infrastructure"]}}

    def test_unresolvable_user_override_fails_open_to_baseline(self, tmp_path, monkeypatch, caplog):
        # A bound ~user session override whose home cannot be resolved raises
        # RuntimeError out of resolve_context_cwd() (§3.1): recall fails
        # open to the configured baseline, reflect forwards nothing
        # (Correction #10a) — neither call aborts.
        caplog.set_level(logging.DEBUG, logger="plugins.memory.hindsight")
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING, config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = runtime_cwd.set_session_cwd("~hermes-nosuchuser-999/work")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert len(client.recall_calls) == 1
        assert len(client.reflect_calls) == 1
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])
        # Pin the EXCEPTION path (not the None-cwd path): the narrow
        # (OSError, RuntimeError) guard fired and logged the concrete reason.
        failures = [r for r in caplog.records
                    if "context cwd resolution failed" in r.getMessage()]
        assert failures, "resolver failure must hit the fail-open guard"
        assert any("RuntimeError" in r.getMessage() for r in failures)

    def test_terminal_policy_unavailable_refusal_propagates(self, tmp_path, monkeypatch):
        # R5 boundary: an active refusal scope is a deliberate refusal, not a
        # cwd-resolution failure — it must propagate past the narrow
        # (OSError, RuntimeError) guard and never reach the backend.
        from tools.terminal_scope import TerminalPolicyUnavailable

        p, client, root = _desktop_provider(tmp_path, monkeypatch, routing=self.ROUTING)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")

        def _refuse():
            raise TerminalPolicyUnavailable("policy unreadable in this scope")

        monkeypatch.setattr(runtime_cwd, "resolve_context_cwd", _refuse)
        try:
            with pytest.raises(TerminalPolicyUnavailable):
                p._recall("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert client.recall_calls == []


# ---------------------------------------------------------------------------
# §3.4 — whole-list extra_tags validation (R5 residual 2)
# ---------------------------------------------------------------------------


class TestTagListValidation:
    def test_blank_element_rejects_whole_list_both_entry_points(self, tmp_path, monkeypatch):
        # One blank element invalidates the WHOLE list: never silently
        # sanitized down to ["infrastructure"] — recall keeps the baseline
        # and reflect forwards nothing (Correction #10a, §3.4).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": ["infrastructure", ""]}},
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])

    def test_non_str_element_entry_skipped_at_load(self, tmp_path, monkeypatch):
        # The real loader rejects a non-str element by dropping the whole
        # entry: the domain becomes unrouted -> baseline on both entry points.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": [123]}},
            config=_BASELINE_CFG)
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        _assert_baseline_kwargs(client.recall_calls[0])
        _assert_no_tag_kwargs(client.reflect_calls[0])

    def test_nonblank_whitespace_tag_forwarded_verbatim(self, tmp_path, monkeypatch):
        # A whitespace-wrapped tag is nonblank -> the route ACTIVATES and the
        # string is forwarded VERBATIM on both entry points (no strip/rewrite).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": [" infrastructure "]}})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert client.recall_calls[0]["tags"] == [" infrastructure "]
        assert client.recall_calls[0]["tags_match"] == "any_strict"
        assert client.reflect_calls[0]["tags"] == [" infrastructure "]
        assert client.reflect_calls[0]["tags_match"] == "any_strict"


# ---------------------------------------------------------------------------
# §3.8.4 — Discord regression controls
# ---------------------------------------------------------------------------


class TestDiscordRegression:
    def test_discord_thread_override_unchanged(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, platform="discord", thread_id="thread-A",
            routing={
                "discord:thread-A": {"extra_tags": ["domain-a"]},
                "infrastructure": {"extra_tags": ["infrastructure"]},
            })
        p._recall("q")
        kwargs = client.recall_calls[0]
        assert kwargs["tags"] == ["channel:discord:thread-A", "domain-a"]
        assert kwargs["tags_match"] == "any_strict"

    def test_dormant_discord_entry_keeps_dormant_semantics(self, tmp_path, monkeypatch):
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, platform="discord", thread_id="thread-B",
            routing={"discord:thread-B": {"extra_tags": ["domain-b"], "dormant": True}})
        p._recall("q")
        assert "tags" not in client.recall_calls[0]

    def test_desktop_domain_branch_never_shadows_thread_override(self, tmp_path, monkeypatch):
        # platform == "desktop" AND a platform:thread override already applied:
        # the thread override stays authoritative (§3.2 disjoint condition).
        # Correction #10b: the control starts from a NONEMPTY CONFIGURED
        # recall_tags/recall_tags_match baseline, so the override must be shown
        # to REPLACE that baseline (not intersect/merge with it), and neither
        # the configured baseline tags nor the Desktop domain's tags may leak.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, platform="desktop", thread_id="thread-C",
            config=_BASELINE_CFG,
            routing={
                "desktop:thread-C": {"extra_tags": ["thread-scoped"]},
                "infrastructure": {"extra_tags": ["infrastructure"]},
            })
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        kwargs = client.recall_calls[0]
        assert kwargs["tags"] == ["channel:desktop:thread-C", "thread-scoped"]
        assert kwargs["tags_match"] == "any_strict"
        # Replace, don't merge: neither the configured baseline filter nor the
        # competing Desktop domain's extra_tags leak into the override.
        assert "baseline-tag" not in kwargs["tags"]
        assert "infrastructure" not in kwargs["tags"]
        # Correction #10b: the platform:thread override is NOT an active
        # Desktop domain match -> reflect forwards no tag filter.
        _assert_no_tag_kwargs(client.reflect_calls[0])


# ---------------------------------------------------------------------------
# §3.8.5 — automatic injection + observable failure signals
# ---------------------------------------------------------------------------


class TestAutomaticInjection:
    ROUTING = {"infrastructure": {"extra_tags": ["infrastructure"]}}

    def test_sync_prefetch_injects_domain_scoped_results(self, tmp_path, monkeypatch):
        # prefetch() is the provider surface MemoryManager.prefetch_all drives
        # for automatic injection (sync mode here, matching the deployed config).
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            results=[_fact("infra memory one", ["infrastructure"]),
                     _fact("infra memory two", ["infrastructure", "investments"])])
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            text = p.prefetch("current turn question")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert "infra memory one" in text
        assert "infra memory two" in text
        kwargs = client.recall_calls[0]
        assert kwargs["tags"] == ["infrastructure"]
        assert kwargs["tags_match"] == "any_strict"
        status = p.recall_status()
        assert status is not None
        assert status.count == 2  # observable success signal, not just non-empty

    def test_prefetch_failure_surfaces_in_log_and_never_crashes(self, tmp_path, monkeypatch, caplog):
        # Recall errors must surface via the log envelope (not silently render
        # empty context) AND must not crash the conversation — optional memory
        # failures stay optional. Distinct from the valid-empty-bank case.
        caplog.set_level(logging.DEBUG, logger="plugins.memory.hindsight")
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch, routing=self.ROUTING,
            recall_error=TimeoutError("bank timeout"))
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            text = p.prefetch("current turn question")  # must not raise
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert text == ""
        assert p.recall_status() is None  # no false success signal
        failures = [r for r in caplog.records
                    if "Hindsight recall failed" in r.getMessage()]
        assert failures, "recall failure must surface in the log envelope, not silently"
        assert any(r.exc_info and r.exc_info[0] is TimeoutError for r in failures)


# ---------------------------------------------------------------------------
# §3.8.6 — sync-mode guard (trips closed, logged, every routing evaluation)
# ---------------------------------------------------------------------------


class TestSyncModeGuard:
    def test_sync_false_trips_closed_and_logs_every_evaluation(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.DEBUG, logger="plugins.memory.hindsight")
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": ["infrastructure"]}},
            config={"recall_sync": False})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")  # routable cwd bound from the start
        try:
            p._recall("initial cwd evaluation")
            p._reflect("second evaluation")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        assert "tags" not in client.recall_calls[0]
        assert "tags" not in client.reflect_calls[0]
        trips = [r for r in caplog.records if "trips closed" in r.getMessage()]
        assert len(trips) >= 2  # logged for EVERY routing evaluation

    def test_sync_false_keeps_installed_global_list(self, tmp_path, monkeypatch):
        # Feature disablement, not confidentiality: the baseline global list
        # stays in force verbatim when the desktop branch trips closed.
        # Correction #10b control: reflect forwards no tag filter at all.
        p, client, root = _desktop_provider(
            tmp_path, monkeypatch,
            routing={"infrastructure": {"extra_tags": ["infrastructure"]}},
            config={"recall_sync": False, "recall_tags": ["global-tag"]})
        _pack(root, "infrastructure")
        token = _bind(root, "infrastructure")
        try:
            p._recall("q")
            p._reflect("q")
        finally:
            runtime_cwd._SESSION_CWD.reset(token)
        kwargs = client.recall_calls[0]
        assert kwargs["tags"] == ["global-tag"]
        assert kwargs["tags_match"] == "any"
        _assert_no_tag_kwargs(client.reflect_calls[0])
