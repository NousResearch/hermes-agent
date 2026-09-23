"""C11 routing truth table: every config shape users write routes the prompt AND the key to
exactly one host.

Class: provider / model routing and credential resolution (issue_classes.md C11). Users hit it
as "I picked my custom provider and the prompt went to the default cloud", "`-m alias` sent the
alias key to the wrong host" (#103933/#107191/#109440), "a bare custom provider fell through to a
stale OPENAI_BASE_URL", "fallback shipped the primary key to the fallback host", "a rate-limited
pool key spilled onto another provider".

Harness: ``Fleet`` stands up one real loopback OpenAI-compatible host per provider identity
(main, alias, named, legacy, fallback, pool, aux, deleg, profile, plus the ambient
env-configured identities ``decoy`` = OPENAI_BASE_URL/OPENAI_API_KEY, ``cloud`` =
OPENROUTER_BASE_URL/OPENROUTER_API_KEY and ``vendor`` = ANTHROPIC_API_KEY). Each accepts only its
own random key. A loopback CONNECT trap (HTTPS_PROXY) records any egress to a real inference API. EVERY scenario
configures ALL identities (the full config a real user accumulates) and differs only in the
selection, so a misroute always has somewhere to land and be caught. Hermes runs for real in a
child process (``hermes chat -q``, ``hermes -z``, the stdio ``tui_gateway`` the TUI/Desktop
drive) with a tmp HOME/HERMES_HOME and every credential env var stripped.

One invariant (``check_routing``) after every leg: requests exist only on the selected hosts,
each carrying only that host's own key, and the answer the user sees came from that host.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pytest

from tests.e2e.core._pending_fixes import Gap, expect_gaps
from tests.fakes.fake_llm_provider import MODEL_ID, Error, ToolCall

from ._routing_helpers import (
    EgressTrap,
    Fleet,
    RoutingLeak,
    RunResult,
    TuiGateway,
    always,
    assert_routing,
    check_routing,
    describe,
    inference_hosts,
    pool_auth,
    reject_key,
    run_hermes,
    write_home,
)

ROLES = ("main", "alias", "named", "legacy", "fallback", "pool", "aux", "deleg", "profile", "decoy", "cloud", "vendor")
Q = ["chat", "-q", "hello"]


def base_config(f: Fleet) -> tuple[dict[str, Any], dict[str, str]]:
    """Every identity configured at once; scenarios only change what is SELECTED."""
    config: dict[str, Any] = {
        "model": {"provider": "custom", "base_url": f.url("main"), "default": "model-main",
                  "key_env": "C11_MAIN_KEY", "context_length": 128000},
        "providers": {
            "named-host": {"name": "NamedHost", "base_url": f.url("named"), "key_env": "C11_NAMED_KEY",
                           "model": "model-named"},
            "pool-host": {"name": "PoolHost", "base_url": f.url("pool"), "model": "model-pool"},
        },
        "custom_providers": [
            {"name": "legacy-host", "base_url": f.url("legacy"), "api_key": f.key("legacy"), "model": "model-legacy"},
        ],
        "model_aliases": {
            "alias-host": {"model": "model-alias", "provider": "custom", "base_url": f.url("alias"),
                           "api_key": f.key("alias")},
            "alias-env": {"model": "model-alias-env", "provider": "custom", "base_url": f.url("alias"),
                          "key_env": "C11_ALIAS_KEY"},
        },
        "agent": {"api_max_retries": 1},
    }
    env = {
        "C11_MAIN_KEY": f.key("main"),
        "C11_NAMED_KEY": f.key("named"),
        "C11_ALIAS_KEY": f.key("alias"),
        "C11_FALLBACK_KEY": f.key("fallback"),
        # Ambient identities a real .env carries: a routing bug that "falls back to the default
        # provider" lands on one of these two hosts and is caught.
        "OPENAI_API_KEY": f.key("decoy"),
        "OPENAI_BASE_URL": f.url("decoy"),
        "OPENROUTER_API_KEY": f.key("cloud"),
        "OPENROUTER_BASE_URL": f.url("cloud"),
        # A built-in vendor key with no host of its own: it must never appear on ANY fleet host.
        "ANTHROPIC_API_KEY": f.key("vendor"),
    }
    return config, env


# Scenario table -------------------------------------------------------------------


@dataclass
class Leg:
    argv: list[str]
    hosts: tuple[str, ...]                     # hosts this leg may reach (each with its own key only)
    must_hit: dict[str, int] = field(default_factory=dict)
    answer_from: str | None = None             # host whose answer the user must see (rc 0)
    rc: str = "ok"                              # ok | fail | any
    profile: str | None = None                 # run with HERMES_HOME = profiles/<name>
    fail_text: str | None = None               # rc="fail": the error the user must see
    timeout: float = 240.0


@dataclass
class Case:
    id: str
    legs: list[Leg]
    setup: Callable[[dict[str, Any], dict[str, str], Fleet], dict[str, Any] | None] = lambda c, e, f: None
    key_counts: dict[str, int] = field(default_factory=dict)
    profiles: Callable[[Fleet], dict[str, tuple[dict[str, Any], dict[str, str]]]] = lambda f: {}
    # extra ordered-key expectation on one host: (host, [key indexes in first-seen order])
    key_order: tuple[str, list[int]] | None = None


def _model(**block: Any) -> Callable[[dict[str, Any], dict[str, str], Fleet], None]:
    def setup(c: dict[str, Any], _e: dict[str, str], _f: Fleet) -> None:
        c["model"] = {"context_length": 128000, **block}
    return setup


def _main_api_key_literal(c: dict[str, Any], e: dict[str, str], f: Fleet) -> None:
    c["model"].pop("key_env")
    c["model"]["api_key"] = f.key("main")
    e.pop("C11_MAIN_KEY")


def _bare_custom(c: dict[str, Any], e: dict[str, str], _f: Fleet) -> None:
    c["model"] = {"provider": "custom", "default": "model-main", "context_length": 128000}
    # Bare custom + an OpenRouter key/mirror is a deliberate contract (it resolves to OpenRouter,
    # tests/hermes_cli/test_runtime_provider_resolution.py); with neither, nothing may be reached
    # -- not the env OPENAI_BASE_URL decoy, not the real OpenRouter API (egress trap).
    e.pop("OPENROUTER_BASE_URL")
    e.pop("OPENROUTER_API_KEY")


def _fallback_named_after(status: int) -> Callable[[dict[str, Any], dict[str, str], Fleet], None]:
    def setup(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> None:
        c["fallback_providers"] = [{"provider": "named-host", "model": "model-named"}]
        f.script("main", always(lambda: Error(status, "scripted primary failure")))
    return setup


def _fallback_custom_after(status: int) -> Callable[[dict[str, Any], dict[str, str], Fleet], None]:
    def setup(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> None:
        c["fallback_providers"] = [{"provider": "custom", "base_url": f.url("fallback"),
                                    "key_env": "C11_FALLBACK_KEY", "model": "model-fallback"}]
        f.script("main", always(lambda: Error(status, "scripted primary failure")))
    return setup


def _pool(n_keys: int, *, first_key_status: int | None, all_fail: bool = False):
    def setup(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> dict[str, Any]:
        c["model"] = {"provider": "pool-host", "default": "model-pool", "context_length": 128000}
        c["agent"]["api_max_retries"] = 3  # rate limits retry once, then rotate
        if all_fail:
            f.script("pool", always(lambda: Error(429, "scripted pool exhaustion")))
        elif first_key_status is not None:
            f.script("pool", reject_key(f.key("pool", 0), first_key_status))
        return pool_auth("pool-host", f.keys["pool"][:n_keys])
    return setup


def _aux_title_endpoint(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> None:
    c["auxiliary"] = {"title_generation": {"base_url": f.url("aux"), "api_key": f.key("aux"), "model": "model-aux"}}


def _aux_bare_custom(c: dict[str, Any], e: dict[str, str], _f: Fleet) -> None:
    # Aux task pinned to bare ``custom`` while no custom endpoint exists anywhere (see _bare_custom):
    # the title call must not ride the env OPENAI_BASE_URL decoy.
    c["model"] = {"provider": "named-host", "default": "model-named", "context_length": 128000}
    c["auxiliary"] = {"title_generation": {"provider": "custom", "model": "model-aux"}}
    e.pop("OPENROUTER_BASE_URL")
    e.pop("OPENROUTER_API_KEY")


def _delegate_first(f: Fleet) -> None:
    calls = {"n": 0}

    def respond(_record: dict[str, Any]) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolCall("delegate_task", {"goal": "Reply with the single word ok.", "context": "none"})
        return None
    f.script("main", respond)


def _delegation_endpoint(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> None:
    c["delegation"] = {"base_url": f.url("deleg"), "api_key": f.key("deleg"), "model": "model-deleg"}
    _delegate_first(f)


def _delegation_named(c: dict[str, Any], _e: dict[str, str], f: Fleet) -> None:
    c["delegation"] = {"provider": "named-host", "model": "model-named"}
    _delegate_first(f)


def _delegation_openai_alias(c: dict[str, Any], e: dict[str, str], f: Fleet) -> None:
    # ``provider: openai`` = the user's OPENAI_BASE_URL/OPENAI_API_KEY pair (here: the deleg host).
    e["OPENAI_BASE_URL"], e["OPENAI_API_KEY"] = f.url("deleg"), f.key("deleg")
    c["delegation"] = {"provider": "openai", "model": "model-deleg"}
    _delegate_first(f)


def _profile_work(f: Fleet) -> dict[str, tuple[dict[str, Any], dict[str, str]]]:
    config, env = base_config(f)
    config["model"] = {"provider": "custom", "base_url": f.url("profile"), "default": "model-profile",
                       "key_env": "C11_PROFILE_KEY", "context_length": 128000}
    env = {k: v for k, v in env.items() if k != "C11_MAIN_KEY"}
    env["C11_PROFILE_KEY"] = f.key("profile")
    return {"work": (config, env)}


CASES: list[Case] = [
    Case("custom_base_url_key_env", [Leg(Q, ("main",), {"main": 1}, "main")]),
    Case("custom_base_url_api_key_literal", [Leg(Q, ("main",), {"main": 1}, "main")], _main_api_key_literal),
    # Fails fast with the no-credentials error: a hang to the harness kill (rc -9) is not a fail-fast.
    Case("bare_custom_without_base_url_fails_fast",
         [Leg(Q, (), rc="fail", fail_text="provider 'custom' resolved without credentials", timeout=90)],
         _bare_custom),
    Case("named_providers_entry_key_env",
         [Leg(Q, ("named",), {"named": 1}, "named")], _model(provider="named-host", default="model-named")),
    Case("legacy_custom_providers_entry_api_key",
         [Leg(Q, ("legacy",), {"legacy": 1}, "legacy")], _model(provider="custom:legacy-host", default="model-legacy")),
    Case("alias_startup_q_api_key", [Leg([*Q, "-m", "alias-host"], ("alias",), {"alias": 1}, "alias")]),
    Case("alias_startup_q_key_env", [Leg([*Q, "-m", "alias-env"], ("alias",), {"alias": 1}, "alias")]),
    Case("alias_startup_oneshot_z", [Leg(["-z", "hello", "-m", "alias-host"], ("alias",), {"alias": 1}, "alias")]),
    Case("fallback_named_after_401",
         [Leg(Q, ("main", "named"), {"main": 1, "named": 1}, "named")], _fallback_named_after(401)),
    Case("fallback_custom_after_429",
         [Leg(Q, ("main", "fallback"), {"main": 1, "fallback": 1}, "fallback")], _fallback_custom_after(429)),
    Case("pool_two_keys_429_rotates_on_same_host",
         [Leg(Q, ("pool",), {"pool": 2}, "pool")], _pool(2, first_key_status=429), key_counts={"pool": 2},
         key_order=("pool", [0, 1])),
    Case("pool_single_key_exhausted_no_spill",
         [Leg(Q, ("pool",), {"pool": 1}, rc="any")], _pool(1, first_key_status=None, all_fail=True)),
    Case("profile_override_routes_per_profile",
         [Leg(Q, ("main",), {"main": 1}, "main"),
          Leg(["-p", "work", *Q], ("profile",), {"profile": 1}, "profile")],
         profiles=_profile_work),
    Case("aux_bare_custom_never_reaches_env_openai_base_url",
         [Leg(Q, ("named",), {"named": 1}, "named")], _aux_bare_custom),
    Case("aux_title_endpoint_separate_from_main",
         [Leg(Q, ("main", "aux"), {"main": 1, "aux": 1}, "main")], _aux_title_endpoint),
    Case("delegation_endpoint_child_routed",
         [Leg(Q, ("main", "deleg"), {"main": 2, "deleg": 1}, "main")], _delegation_endpoint),
    Case("delegation_named_provider_child_routed",
         [Leg(Q, ("main", "named"), {"main": 2, "named": 1}, "main")], _delegation_named),
    Case("delegation_provider_openai_alias_child_routed",
         [Leg(Q, ("main", "deleg"), {"main": 2, "deleg": 1}, "main")], _delegation_openai_alias),
]


# Execution ------------------------------------------------------------------------


@dataclass
class LegOutcome:
    leg: Leg
    run: RunResult
    log: dict[str, list[dict[str, Any]]]
    egress: list[tuple[str, str]]


def _run_case(case: Case, root: Path) -> tuple[Fleet, list[LegOutcome]]:
    fleet = Fleet(ROLES, key_counts=case.key_counts).start()
    trap = EgressTrap()
    try:
        home = root / case.id / "home"
        config, env = base_config(fleet)
        auth = case.setup(config, env, fleet)
        write_home(home / ".hermes", config, env, auth)
        for name, (pconfig, penv) in case.profiles(fleet).items():
            write_home(home / ".hermes" / "profiles" / name, pconfig, penv)
        outcomes = []
        for leg in case.legs:
            marks, egress_mark = fleet.marks(), len(trap.attempts)
            hermes_home = home / ".hermes" / "profiles" / leg.profile if leg.profile else None
            run = run_hermes(leg.argv, home, hermes_home=hermes_home, proxy=trap.url, timeout=leg.timeout)
            outcomes.append(LegOutcome(leg, run, fleet.since(marks), trap.attempts[egress_mark:]))
        return fleet, outcomes
    finally:
        fleet.stop()
        trap.stop()


@pytest.fixture(scope="module")
def cli_outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """All CLI scenarios run concurrently (each owns its own fleet and HOME)."""
    root = tmp_path_factory.mktemp("c11-cli")
    with cf.ThreadPoolExecutor(max_workers=6, thread_name_prefix="c11-case") as pool:
        futures = {case.id: pool.submit(_run_case, case, root) for case in CASES}
        return {cid: fut for cid, fut in futures.items()}


# Live gaps, applied only while their probe reproduces them (see _pending_fixes).
GAP_120299 = Gap(120299, "#120299: bare `provider: custom` with no endpoint falls through to OpenRouter and ships the "
                         "OPENAI_BASE_URL-bound OPENAI_API_KEY there; the pre-request Anthropic credential refresh "
                         "puts ANTHROPIC_API_KEY on an alias's foreign host", raises=RoutingLeak)
GAP_120295 = Gap(120295, "#120295: `/model <id> --provider X` adopts another provider's alias (endpoint + key) "
                         "for the same model id", raises=RoutingLeak)
KNOWN_GAPS = {"bare_custom_without_base_url_fails_fast": (GAP_120299,)}


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in CASES])
def test_cli_routing_truth_table(case: Case, cli_outcomes: dict[str, Any], request: pytest.FixtureRequest) -> None:
    expect_gaps(request, *KNOWN_GAPS.get(case.id, ()))
    fleet, outcomes = cli_outcomes[case.id].result(timeout=900)
    for i, out in enumerate(outcomes):
        leg, run = out.leg, out.run
        ctx = (f"[{case.id} leg {i}: hermes {' '.join(leg.argv)}] rc={run.rc} ({run.seconds:.1f}s)\n"
               f"requests:\n{describe(out.log)}\nstdout tail:\n{run.stdout[-1500:]}\nstderr tail:\n{run.stderr[-1500:]}")
        assert_routing(check_routing(fleet, out.log, {h: fleet.keys[h] for h in leg.hosts}, leg.must_hit), ctx)
        leaked = [t for _m, t in out.egress if urlparse(t).hostname in inference_hosts()]
        if leaked:
            raise RoutingLeak(f"prompt egress to a real inference API: {leaked}\n{ctx}")
        if leg.rc == "ok":
            assert run.rc == 0, ctx
        elif leg.rc == "fail":
            assert run.rc not in (0, None, -9), "expected a fail-fast exit, not success or the harness kill\n" + ctx
            assert leg.fail_text in " ".join((run.stdout + run.stderr).split()), f"expected {leg.fail_text!r}\n" + ctx
        if leg.answer_from:
            assert f"answer-from-{leg.answer_from}" in run.stdout, ctx
    if case.key_order:
        host, order = case.key_order
        seen: list[str] = []
        for r in (r for out in outcomes for r in out.log[host] if r["kind"] == "main"):
            if not seen or seen[-1] != r["auth"]:
                seen.append(r["auth"])
        assert seen == [f"Bearer {fleet.key(host, i)}" for i in order], (
            f"{host} main-request keys in order: {seen}")


# Mid-session /model switches through the real tui_gateway (TUI / Desktop backend) -------------


@dataclass
class Switch:
    value: str | None      # config.set model value; None = no switch (first turn)
    host: str              # host this turn must land on, with its own key only
    ok: bool = True        # False: the turn is expected to fail (e.g. exhausted pool)
    keyless: bool = False  # the host may also see NO credential (withholding a key is not a leak)


def test_tui_gateway_model_switch_routing(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """One live session walks the switch matrix; after every switch the next turn lands on
    exactly the selected host with exactly its key, and nothing reaches any other host.
    Leg 5 is #120295's cell, leg 6 #120299's."""
    expect_gaps(request, GAP_120295, GAP_120299)
    fleet = Fleet(ROLES).start()
    trap = EgressTrap()
    home = tmp_path / "home"
    config, env = base_config(fleet)
    # Built-in vendor label + foreign host, no key: ANTHROPIC_API_KEY must stay home (#28660).
    config["model_aliases"]["builtin-label-lan"] = {
        "model": MODEL_ID, "provider": "anthropic", "base_url": fleet.url("alias")}  # id the host lists
    pool_state = {"fail": False}
    fleet.script("pool", lambda _r: Error(429, "scripted pool exhaustion") if pool_state["fail"] else None)
    write_home(home / ".hermes", config, env, pool_auth("pool-host", fleet.keys["pool"]))
    gw = TuiGateway(home, proxy=trap.url)
    try:
        gw.wait(gw.event("gateway.ready"), timeout=120)
        created = gw.call("session.create", {"cols": 100})
        sid = created["result"]["session_id"]
        legs = [
            Switch(None, "main"),
            Switch("alias-host", "alias"),
            Switch("model-named --provider named-host", "named"),
            Switch("model-legacy --provider custom:legacy-host", "legacy"),
            Switch("alias-env", "alias"),
            Switch("model-alias --provider named-host", "named"),
            Switch("builtin-label-lan", "alias", ok=False, keyless=True),
            Switch("model-legacy --provider custom:legacy-host", "legacy"),
            Switch("model-pool --provider pool-host", "pool", ok=False),
            Switch(None, "pool"),
        ]
        for i, leg in enumerate(legs):
            marks = fleet.marks()
            if leg.value:
                reply = gw.call("config.set", {"session_id": sid, "key": "model", "value": leg.value})
                assert "result" in reply, f"leg {i} switch {leg.value!r} refused: {reply}"
            pool_state["fail"] = leg.host == "pool" and not leg.ok
            done = gw.turn(sid, f"turn {i}")
            if i == 0:
                gw.seen_or_wait(gw.event("session.title", sid), timeout=120)  # first-turn aux call settles
            log = fleet.since(marks)
            payload = (done.get("params") or {}).get("payload") or {}
            ctx = f"leg {i} ({leg.value!r} -> {leg.host}): {done.get('params', {}).get('type')} {str(payload)[:300]}\n{describe(log)}"
            print(ctx)
            allowed = {*fleet.keys[leg.host], *(("", "no-key-required") if leg.keyless else ())}
            assert_routing(check_routing(fleet, log, {leg.host: allowed}, {leg.host: 1}), ctx)
            if leg.ok:
                assert payload.get("text") == f"answer-from-{leg.host}", ctx
    finally:
        gw.close()
        fleet.stop()
        trap.stop()
