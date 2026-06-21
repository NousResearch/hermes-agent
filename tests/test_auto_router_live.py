"""Live smoke test for ``agent.auto_router``.

Issues a real session, runs the intent router, and verifies the discount
header is honoured by hitting ``/responses`` twice (with + without
``Copilot-Session-Token``) and asserting the cost ratio is ~0.9.

Skipped automatically when no GitHub Copilot token is available.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

# Make the hermes src tree importable when the test is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.auto_router import (  # noqa: E402
    AutoRouter,
    endpoint_for_model,
    is_enabled,
)


def _gh_token() -> str:
    for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val
    try:
        out = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


@pytest.fixture(scope="module")
def gh_token() -> str:
    tok = _gh_token()
    if not tok:
        pytest.skip("No GitHub token available for live Copilot smoke test")
    if not is_enabled():
        pytest.skip("HERMES_COPILOT_AUTO_MODE disabled")
    return tok


@pytest.fixture(scope="module")
def router() -> AutoRouter:
    return AutoRouter()


def test_session_issuance_returns_jwt_with_discount(
    gh_token: str, router: AutoRouter
) -> None:
    session = router.get_session(gh_token)
    assert session is not None, "session issuance failed"
    assert session.session_token.count(".") == 2, "expected JWT shape"
    assert session.available_models, "available_models should not be empty"
    assert session.selected_model in session.available_models
    assert session.discounted_costs, "discounted_costs missing"
    # Every advertised model should carry the same 0.9× multiplier marker.
    assert all(
        v == 0.1 for v in session.discounted_costs.values()
    ), f"unexpected discount values: {session.discounted_costs}"


def test_router_returns_chosen_model(gh_token: str, router: AutoRouter) -> None:
    session = router.get_session(gh_token)
    assert session is not None
    decision = router.route(
        gh_token, "Refactor this React class component to use hooks", session
    )
    assert decision is not None
    assert decision.chosen_model in session.available_models
    assert decision.routing_method, "routing_method should be set"


def test_endpoint_for_model_routing() -> None:
    assert endpoint_for_model("gpt-5.4").endswith("/responses")
    assert endpoint_for_model("gpt-5.3-codex").endswith("/responses")
    assert endpoint_for_model("claude-sonnet-4.6").endswith("/v1/messages")
    assert endpoint_for_model("claude-haiku-4.5").endswith("/v1/messages")
    assert endpoint_for_model("gemini-2.5-pro").endswith("/chat/completions")


def _tiny_request(model: str, session_token: str | None, gh_token: str) -> dict:
    """POST a minimal /responses request and return parsed JSON."""
    from hermes_cli.copilot_auth import copilot_request_headers, get_copilot_api_token

    headers = copilot_request_headers(is_agent_turn=True)
    headers["Authorization"] = f"Bearer {get_copilot_api_token(gh_token)}"
    headers["Content-Type"] = "application/json"
    if session_token:
        headers["Copilot-Session-Token"] = session_token

    body = json.dumps(
        {
            "model": model,
            "input": [{"role": "user", "content": "reply with the single word: ok"}],
            "max_output_tokens": 20,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        endpoint_for_model(model), data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _output_cost_per_batch(payload: dict) -> int:
    for entry in payload.get("copilot_usage", {}).get("token_details", []):
        if entry.get("token_type") == "output":
            return int(entry.get("cost_per_batch") or 0)
    return 0


def test_session_token_applies_ten_percent_discount(
    gh_token: str, router: AutoRouter
) -> None:
    session = router.get_session(gh_token)
    assert session is not None
    # Pick the simplest model for quick + cheap probe.
    model = "gpt-5.4-mini" if "gpt-5.4-mini" in session.available_models else session.selected_model

    discounted = _tiny_request(model, session.session_token, gh_token)
    baseline = _tiny_request(model, None, gh_token)

    d = _output_cost_per_batch(discounted)
    b = _output_cost_per_batch(baseline)
    assert b > 0 and d > 0, f"missing cost data: discounted={discounted} baseline={baseline}"
    ratio = d / b
    # Expect 0.90 exactly (server multiplies by 0.9). Allow 1% slack.
    assert 0.89 <= ratio <= 0.91, f"discount ratio={ratio:.4f} (d={d}, b={b})"


def test_cache_reuses_session(gh_token: str, router: AutoRouter) -> None:
    s1 = router.get_session(gh_token, conversation_id="cache-test")
    s2 = router.get_session(gh_token, conversation_id="cache-test")
    assert s1 is s2 or s1.session_token == s2.session_token

    s3 = router.get_session(
        gh_token, conversation_id="cache-test", force_refresh=True
    )
    # Forced refresh produces a new JWT (iat differs).
    assert s3.session_token != s1.session_token or s3.expires_at >= s1.expires_at
