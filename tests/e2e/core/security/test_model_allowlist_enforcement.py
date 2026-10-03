"""A user's ``security.model_allowlist`` bounds every route Hermes picks for itself (#128524).

Class: provider / model routing and credential resolution. The report: a user whose policy was
"local ``qwen3-vl:8b`` for vision, ``deepseek-v4-flash`` via OpenRouter, no other models ever" had the
agent, on a rate-limited call, spend their OpenRouter credit on models they never authorized, with no
consent prompt. The sanctioned-model rule lived in a skill, so nothing stopped the hop.

The invariant: when a model is outside the allowlist, NO request carrying that provider's key ever
reaches that provider — neither through the main agent's fallback chain nor through a delegated
child's own ``delegation.fallback_providers``. The allowlist removes the route; the primary keeps
serving, and the turn ends on the user's own model or fails cleanly.

Every scenario is a blocked/allowed PAIR on one config that differs only in the allowlist, and both
legs are judged by the same oracle. That pairing is what makes the result mean something: a harness
that had merely broken fallback would pass the blocked leg and fail the allowed one.

Harness: a real ``AIAgent`` in a child process (own HERMES_HOME, real SessionDB, real
``delegate_task``) against loopback ``FakeLLMServer`` hosts, each accepting only its own minted key.
``paid`` is the unauthorized aggregator stand-in reachable only through a fallback entry, so one
recorded request on it IS an unauthorized spend. Every credential env var is stripped from the child
and every host is loopback, so nothing can reach a real inference API.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

import hermes_yaml as yaml
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall

REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = Path(__file__).with_name("_allowlist_driver.py")

PRIMARY, PAID, CHILD = "primary", "paid", "child"
HOSTS = (PRIMARY, PAID, CHILD)
# The id on the unauthorized host: an aggregator-qualified paid model, exactly the shape a fallback
# entry carries and a user never wrote down.
PAID_MODEL = "vendor/paid-model"
MAIN_MODEL, CHILD_MODEL = "model-main", "model-child"
TURN_DEADLINE_S = 240.0


@dataclass
class Scenario:
    """One allowlist, and one 429 on whichever model the turn is running on."""

    id: str
    allowlist: list[str]
    delegation: bool = False
    # Hosts that must be reached at least `n` times (the primary, plus the paid hop when allowed).
    must_hit: dict[str, int] = field(default_factory=dict)
    # Hosts that must receive no billable request at all.
    forbidden: tuple[str, ...] = (PAID,)


class Fleet:
    """One loopback provider per identity; each accepts only its own minted key."""

    def __init__(self) -> None:
        self.keys = {h: f"sk-allow-{h}-{uuid.uuid4().hex[:12]}" for h in HOSTS}
        self.servers: dict[str, FakeLLMServer] = {}
        self._scripts: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def start(self) -> "Fleet":
        for host in HOSTS:
            server = FakeLLMServer(self._dispatch(host), api_key=self.keys[host], record_get=True)
            server.start()
            self.servers[host] = server
        return self

    def stop(self) -> None:
        for server in self.servers.values():
            server.stop()

    def url(self, host: str) -> str:
        return self.servers[host].base_url

    def script(self, host: str, responder: Callable[[dict[str, Any]], Any]) -> None:
        self._scripts[host] = responder

    def _dispatch(self, host: str) -> Callable[[dict[str, Any]], Any]:
        def respond(record: dict[str, Any]) -> Any:
            responder = self._scripts.get(host)
            reply = responder(record) if responder else None
            return reply if reply is not None else Text(f"answer-from-{host}")
        return respond

    def mark(self) -> dict[str, int]:
        return {h: len(s.requests) for h, s in self.servers.items()}

    def since(self, marks: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
        return {h: list(s.requests[marks.get(h, 0):]) for h, s in self.servers.items()}


def _bearer(record: dict[str, Any]) -> str:
    auth = record.get("auth") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return (record.get("headers") or {}).get("x-api-key", "")


def _chats(log: dict[str, list[dict[str, Any]]], host: str) -> list[dict[str, Any]]:
    return [r for r in log.get(host, []) if r.get("kind") in {"main", "aux"}]


def _describe(log: dict[str, list[dict[str, Any]]]) -> str:
    rows = "\n".join(
        f"  {host}: " + ", ".join(f"{r.get('kind')}[{_bearer(r)[:20]}]" for r in records)
        for host, records in log.items() if records
    )
    return rows or "  (no requests anywhere)"


def _script(fleet: Fleet, scenario: Scenario) -> None:
    if not scenario.delegation:
        # The primary is rate-limited: the fallback under test is the only route left.
        fleet.script(PRIMARY, lambda _r: Error(429, "scripted rate limit"))
        return
    # Delegation: the PARENT succeeds and delegates; the CHILD is rate-limited, so the hop under
    # test is the child's own ``delegation.fallback_providers`` (the subagent vector from the report).
    calls = {"n": 0}

    def respond(_record: dict[str, Any]) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return ToolCall("delegate_task", {"goal": "Reply with the single word ok.", "context": "none"})
        return None

    fleet.script(PRIMARY, respond)
    fleet.script(CHILD, lambda _r: Error(429, "scripted rate limit"))


def _write_config(root: Path, fleet: Fleet, scenario: Scenario) -> tuple[Path, Path]:
    config: dict[str, Any] = {
        "model": {"provider": "custom", "base_url": fleet.url(PRIMARY), "default": MAIN_MODEL,
                  "api_key": fleet.keys[PRIMARY], "context_length": 128000},
        "security": {"model_allowlist": list(scenario.allowlist)},
        "agent": {"api_max_retries": 1, "auto_recovery_cycles": 0, "max_turns": 6},
        "compression": {"enabled": False},
        "memory": {"enabled": False},
        "updates": {"check": False},
        # The unauthorized route on the MAIN agent: a paid aggregator model as the fallback.
        "fallback_providers": [{"provider": "custom", "base_url": fleet.url(PAID),
                                "api_key": fleet.keys[PAID], "model": PAID_MODEL}],
    }
    if scenario.delegation:
        # A child on its own endpoint with its OWN chain naming the same unauthorized model. That
        # block is normalized by scoped_fallback_chain, a different reader from the main chain.
        config["delegation"] = {"base_url": fleet.url(CHILD), "api_key": fleet.keys[CHILD],
                                "model": CHILD_MODEL,
                                "fallback_providers": [{"provider": "custom",
                                                        "base_url": fleet.url(PAID),
                                                        "api_key": fleet.keys[PAID],
                                                        "model": PAID_MODEL}]}
    home = root / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (hermes_home / ".env").write_text("", encoding="utf-8")
    return home, hermes_home


def _child_env(home: Path, hermes_home: Path) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not (k.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_BASE_URL"))
                or k.startswith(("HERMES_", "OPENROUTER", "ANTHROPIC", "OPENAI", "NOUS_"))
                or k in {"PYTEST_CURRENT_TEST"})
    }
    env.update({
        "HOME": str(home), "HERMES_HOME": str(hermes_home), "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1", "TZ": "UTC", "NO_COLOR": "1",
        # HOME is the tmp root, so the live-DB guard would read the tmp state.db as production.
        "HERMES_STATE_DB_GUARD_BYPASS": "1",
    })
    return env


def _run_scenario(scenario: Scenario, root: Path) -> dict[str, Any]:
    fleet = Fleet().start()
    try:
        root.mkdir(parents=True, exist_ok=True)
        _script(fleet, scenario)
        home, hermes_home = _write_config(root, fleet, scenario)
        spec = root / "spec.json"
        spec.write_text(json.dumps({"session_id": f"allow-{scenario.id}"}), encoding="utf-8")
        marks = fleet.mark()
        proc = subprocess.Popen(
            [sys.executable, str(DRIVER), str(spec)], cwd=str(REPO_ROOT),
            env=_child_env(home, hermes_home), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            out, err = proc.communicate(timeout=TURN_DEADLINE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            err += f"\n[driver] killed after {TURN_DEADLINE_S}s"
        return {"log": fleet.since(marks), "rc": proc.returncode, "out": out, "err": err,
                "keys": dict(fleet.keys)}
    finally:
        fleet.stop()


SCENARIOS = [
    # Blocked, main agent: the paid model is outside the allowlist, so the 429 must NOT become a paid
    # request. The turn may end any way; what matters is that the host stays untouched.
    Scenario("main_blocked", allowlist=[MAIN_MODEL, CHILD_MODEL], must_hit={PRIMARY: 1}),
    # Control: the same config with the paid model allowlisted, so the same 429 MUST reach it.
    Scenario("main_allowed", allowlist=[MAIN_MODEL, CHILD_MODEL, PAID_MODEL],
             must_hit={PRIMARY: 1, PAID: 1}, forbidden=()),
    # Blocked, delegated child: the child is the rate-limited one and its OWN fallback_providers
    # names the paid model. This is the subagent vector from the report.
    Scenario("delegation_blocked", allowlist=[MAIN_MODEL, CHILD_MODEL], delegation=True,
             must_hit={PRIMARY: 2, CHILD: 1}),
    Scenario("delegation_allowed", allowlist=[MAIN_MODEL, CHILD_MODEL, PAID_MODEL],
             delegation=True, must_hit={PRIMARY: 2, CHILD: 1, PAID: 1}, forbidden=()),
]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("allowlist")
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="allowlist") as pool:
        return {s.id: pool.submit(_run_scenario, s, root / s.id) for s in SCENARIOS}


@pytest.mark.parametrize("scenario", [pytest.param(s, id=s.id) for s in SCENARIOS])
def test_model_allowlist_bounds_every_automatic_route(
    scenario: Scenario, outcomes: dict[str, Any]
) -> None:
    out = outcomes[scenario.id].result(timeout=TURN_DEADLINE_S + 180)
    log, keys = out["log"], out["keys"]
    ctx = (f"[{scenario.id}] driver rc={out['rc']}\nrequests:\n{_describe(log)}\n"
           f"driver stdout:\n{out['out'][-2000:]}\ndriver stderr:\n{out['err'][-2000:]}")

    # 1. Nothing billable on a forbidden host: no chat request at all, and no request carrying its
    #    key in any form (a model-list probe is not a spend, but a key leaving is still a leak).
    for host in scenario.forbidden:
        served = _chats(log, host)
        assert not served, (
            f"{host} served {len(served)} chat request(s) for an out-of-allowlist model "
            f"(models={[r.get('model') for r in served]}) — an unauthorized spend\n{ctx}")
        for record in log.get(host, []):
            assert _bearer(record) not in keys.values(), (
                f"{host} received a request carrying its own billable key\n{ctx}")

    # 2. Routes that ARE allowed keep working, so the guard filters rather than disables.
    for host, minimum in scenario.must_hit.items():
        served = _chats(log, host)
        assert len(served) >= minimum, (
            f"{host} served {len(served)} chat request(s), expected >= {minimum}\n{ctx}")
        for record in served:
            assert _bearer(record) == keys[host], (
                f"{host} served a request carrying another identity's key\n{ctx}")
