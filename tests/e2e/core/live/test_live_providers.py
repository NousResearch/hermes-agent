"""LIVE provider canary: real AIAgent + real adapters against real vendor APIs.

Secrets-gated and excluded from the default run (``-m live`` to select). Each
case skips cleanly when its credential env var is absent. Classes covered:
C9 (provider wire-format drift), C17 (prompt-cache hits), C11 (real auth +
credential routing + ``/models`` parse). See ``_helpers.py`` for the matrix.

Spend: cheap models only, three scripted turns (plus at most one cache warm-up
retry on cache-capable routes), bounded max_tokens/iterations and a hard
per-test token + dollar guard. Usage and estimated cost are printed per test
(``LIVE-USAGE {...}``) and appended to ``$HERMES_LIVE_USAGE_FILE`` when set.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.e2e.core.live._helpers import (
    LISTING_PROVIDERS,
    LIVE_CASES,
    LOOKUP_VALUES,
    MAX_TOKENS_PER_TEST,
    MAX_USD_PER_TEST,
    TOOL_NAME,
    TURN1,
    TURN2,
    TURN3,
    LiveCase,
    assistant_tool_calls,
    cache_markers,
    emit_usage,
    install_wire_recorder,
    leaked_markers,
    live_key,
    model_override,
    pad_text,
    parse_tool_args,
    reasoning_replay_keys,
    register_lookup_tool,
    usage_line,
    write_live_home,
)

pytestmark = pytest.mark.live

_LISTING_CACHE: dict[str, list[str]] = {}


@pytest.fixture(scope="module", autouse=True)
def _lookup_tool():
    remove = register_lookup_tool()
    yield
    remove()


@pytest.fixture()
def live_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # HOME too (a sibling, not the parent: state.db's live-system guard treats
    # $HOME/.hermes as production), so nothing can resolve the developer's real home.
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes_home"
    home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _resolve(provider: str, model: str | None) -> dict:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested=provider, target_model=model)


def _live_listing(provider: str, runtime: dict) -> list[str]:
    """The same live fetchers the /model picker uses, WITHOUT the curated fallback
    (a silent fallback is exactly what hides a broken listing)."""
    api_key, base_url = runtime.get("api_key") or "", runtime.get("base_url") or ""
    if provider == "openai":
        from hermes_cli.models import fetch_api_models

        return list(fetch_api_models(api_key, base_url, timeout=20.0) or [])
    if provider == "nous":
        from hermes_cli.auth import fetch_nous_models

        return list(fetch_nous_models(inference_base_url=base_url, api_key=api_key) or [])
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    assert profile is not None, f"no provider profile registered for {provider!r}"
    return list(profile.fetch_models(api_key=api_key, base_url=base_url, timeout=20.0) or [])


def _normalize_id(model_id: str) -> str:
    return model_id.split("/", 1)[1] if model_id.startswith("models/") else model_id


def _listing(provider: str) -> list[str]:
    if provider not in _LISTING_CACHE:
        runtime = _resolve(provider, None)
        _LISTING_CACHE[provider] = [_normalize_id(str(m)) for m in _live_listing(provider, runtime)]
    return _LISTING_CACHE[provider]


def _pick_model(case: LiveCase) -> str:
    override = model_override(case)
    if override:
        return override
    listed = set(_listing(case.provider))
    if not listed:
        # A broken listing is test_models_listing_parses' verdict; still run the wire canary.
        return case.model_prefs[0]
    for pref in case.model_prefs:
        if pref in listed:
            return pref
    pytest.fail(
        f"{case.id}: none of {case.model_prefs} is in {case.provider}'s live /models listing "
        f"({len(listed)} ids) — update LIVE_CASES in tests/e2e/core/live/_helpers.py"
    )


def _host_ok(host: str, allowed: tuple[str, ...]) -> bool:
    return any(host == h or host.endswith("." + h) for h in allowed)


# /models ------------------------------------------------------------------------


_KNOWN_LISTING_BUGS = {
    # Native /v1beta/models rejects a Bearer AI-Studio key (401) and the picker silently
    # falls back to the curated list. strict: flips red once the fix lands (drop the entry).
    "gemini": "#62259 Gemini live model discovery sends Bearer auth (fix PRs #62267/#116509)",
}


@pytest.mark.parametrize("provider", [
    pytest.param(p, marks=pytest.mark.xfail(reason=_KNOWN_LISTING_BUGS[p], strict=True))
    if p in _KNOWN_LISTING_BUGS else p
    for p in sorted(LISTING_PROVIDERS)
])
def test_models_listing_parses(provider: str, live_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_env = LISTING_PROVIDERS[provider]
    secret = live_key(key_env)
    monkeypatch.setenv(key_env, secret)
    wire = install_wire_recorder(monkeypatch, secret)
    case = next(c for c in LIVE_CASES if c.provider == provider)
    write_live_home(live_home, provider, case.model_prefs[0])

    runtime = _resolve(provider, None)
    assert runtime.get("api_key") == secret, f"{provider}: resolver did not pick the {key_env} credential"
    assert _host_ok(urlparse(runtime["base_url"]).hostname or "", case.hosts), runtime["base_url"]

    models = [_normalize_id(str(m)) for m in _live_listing(provider, runtime)]
    assert models, f"{provider}: live /models returned nothing ({wire.describe(wire.records)})"
    assert all(isinstance(m, str) and m.strip() for m in models)
    wanted = {pref for c in LIVE_CASES if c.provider == provider for pref in c.model_prefs}
    assert wanted & set(models), f"{provider}: none of the canary models {sorted(wanted)} are listed"
    leaks = [r for r in wire.records if r.carries_key and not _host_ok(r.host, case.hosts)]
    assert not leaks, f"{provider}: credential sent to a foreign host: {wire.describe(leaks)}"
    print(f"LIVE-MODELS provider={provider} count={len(models)}", flush=True)


# 3-turn canary --------------------------------------------------------------------


def _run_turn(agent, text: str, history, wire, label: str):
    start = len(wire.records)
    result = agent.run_conversation(text, conversation_history=history)
    calls = wire.inference(start)
    main = wire.main_turn(start)
    assert main, f"{label}: no agent-loop request carried the test tool ({wire.describe(calls)})"
    # Agent-loop requests must never be rejected: a 4xx (other than 429) is wire drift.
    # Auxiliary calls (title generation) may take a designed 400->adapt retry (e.g.
    # "reasoning is mandatory"); the test end asserts the last aux attempt succeeded.
    bad = [r for r in main if 400 <= r.status < 500 and r.status != 429]
    assert not bad, f"{label}: provider rejected the request (wire drift): {wire.describe(bad)}"
    assert not result.get("failed"), f"{label}: turn failed: {result.get('error')} ({wire.describe(calls)})"
    assert main[-1].status == 200, f"{label}: final request did not succeed: {wire.describe(main)}"
    final = result.get("final_response") or ""
    assert final.strip(), f"{label}: empty final response"
    assert not leaked_markers(final), f"{label}: markup leaked into final text: {leaked_markers(final)} in {final!r}"
    return result, main


@pytest.mark.parametrize("case", LIVE_CASES, ids=[c.id for c in LIVE_CASES])
def test_three_turn_tool_conversation(case: LiveCase, live_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = live_key(case.key_env)
    monkeypatch.setenv(case.key_env, secret)
    wire = install_wire_recorder(monkeypatch, secret)
    model = _pick_model(case)
    write_live_home(live_home, case.provider, model)

    runtime = _resolve(case.provider, model)
    assert runtime.get("api_key") == secret, f"{case.id}: resolver did not pick the {case.key_env} credential"

    from hermes_constants import parse_reasoning_effort
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(live_home / "state.db")
    agent = AIAgent(
        provider=runtime.get("provider"), api_mode=runtime.get("api_mode"),
        base_url=runtime.get("base_url"), api_key=runtime.get("api_key"),
        credential_pool=runtime.get("credential_pool"), model=model,
        session_db=db, session_id=f"live-{case.id}", quiet_mode=True, platform="cli",
        enabled_toolsets=["live_probe"], skip_context_files=True, skip_memory=True,
        skip_background_review=True, max_tokens=2048, max_iterations=6,
        reasoning_config=parse_reasoning_effort("low"), run_budget_seconds=240,
    )
    usage: dict = {}
    try:
        # Turn 1: one forced tool call; the tool result must round-trip into the answer.
        r1, _ = _run_turn(agent, TURN1 + pad_text(case.pad_tokens), None, wire, f"{case.id} turn1")
        groups = assistant_tool_calls(r1["messages"])
        assert groups, f"{case.id} turn1: model made no tool call"
        args = [parse_tool_args(c) for g in groups for c in g]
        assert {"key": "alpha"} in [{"key": a.get("key")} for a in args], args
        assert all(c["function"]["name"] == TOOL_NAME for g in groups for c in g)
        tool_msgs = [m for m in r1["messages"] if m.get("role") == "tool"]
        assert any(LOOKUP_VALUES["alpha"] in str(m.get("content")) for m in tool_msgs)
        assert LOOKUP_VALUES["alpha"] in r1["final_response"], r1["final_response"]

        # Turn 2: two tool calls in ONE assistant message; both results round-trip.
        before = len(r1["messages"])
        r2, _ = _run_turn(agent, TURN2, r1["messages"], wire, f"{case.id} turn2")
        new_groups = assistant_tool_calls(r2["messages"][before:])
        keys = {parse_tool_args(c).get("key") for g in new_groups for c in g}
        assert {"beta", "gamma"} <= keys, f"{case.id} turn2: tool calls {new_groups}"
        assert any(len(g) >= 2 for g in new_groups), (
            f"{case.id} turn2: expected parallel tool calls in one message, got {[len(g) for g in new_groups]}")
        for key in ("beta", "gamma"):
            assert LOOKUP_VALUES[key] in r2["final_response"], r2["final_response"]

        # Turn 3: history reuse (tool results + reasoning replay) under a cache breakpoint.
        cache_before = agent.session_cache_read_tokens
        r3, calls3 = _run_turn(agent, TURN3, r2["messages"], wire, f"{case.id} turn3")
        first3 = calls3[0].body
        sent = str(first3)
        for value in LOOKUP_VALUES.values():
            assert value in sent, f"{case.id} turn3: tool result {value} missing from replayed history"
            assert value in r3["final_response"], f"{case.id} turn3: {r3['final_response']!r}"
        replay = sorted(reasoning_replay_keys(first3))
        cache_read_turn3 = agent.session_cache_read_tokens - cache_before

        if case.cache_expected:
            markers = cache_markers(first3)
            assert 1 <= len(markers) <= 4, f"{case.id} turn3: cache breakpoints {markers}"
            for m in (first3.get("messages") or []):
                if m.get("role") == "tool" and not case.native_anthropic:
                    assert "cache_control" not in m, "top-level cache_control on role:tool (OpenRouter hangs)"
                content = m.get("content")
                for part in content if isinstance(content, list) else []:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        inner = part.get("content")
                        assert not (isinstance(inner, list) and cache_markers(inner)), (
                            "cache_control inside tool_result.content[] (#89886: non-retryable 400)")
            if cache_read_turn3 <= 0:
                # Retry-tolerant: the first write can land on another replica; warm once.
                cache_before = agent.session_cache_read_tokens
                _run_turn(agent, TURN3, r3["messages"], wire, f"{case.id} turn3-warm")
                cache_read_turn3 = agent.session_cache_read_tokens - cache_before
            assert cache_read_turn3 > 0, f"{case.id}: no prompt-cache read on the follow-up turn"

        # Persisted == sent: the three tool results are in state.db for this session.
        rows = db.get_messages(agent.session_id)
        persisted = " ".join(str(r.get("content")) for r in rows if r.get("role") == "tool")
        for value in LOOKUP_VALUES.values():
            assert value in persisted, f"{case.id}: tool result {value} not persisted"

        leaks = [r for r in wire.records if r.carries_key and not _host_ok(r.host, case.hosts)]
        assert not leaks, f"{case.id}: credential sent to a foreign host: {wire.describe(leaks)}"
        main_ids = {id(r) for r in wire.main_turn()}
        aux = [r for r in wire.inference() if id(r) not in main_ids]
        assert not aux or 200 <= aux[-1].status < 300, f"{case.id}: auxiliary call never recovered: {wire.describe(aux)}"
        usage = {"turn3_cache_read": cache_read_turn3, "turn3_replay_keys": replay}
    finally:
        line = usage_line(case, model, agent, wire) | usage
        emit_usage(line)
        if hasattr(agent, "close"):
            agent.close()
    total = line["input"] + line["output"] + line["cache_read"] + line["cache_write"]
    assert total <= MAX_TOKENS_PER_TEST, f"{case.id}: spend guard tripped ({total} tokens)"
    assert line["list_price_ceiling_usd"] <= MAX_USD_PER_TEST, f"{case.id}: spend guard tripped {line}"
