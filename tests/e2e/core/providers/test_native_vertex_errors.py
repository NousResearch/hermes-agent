"""Vertex AI documented errors through the real CLI: retried per semantics or surfaced exactly once.

Each scenario is one ``hermes chat -q`` turn against its own fake (``tests/fakes/providers/vertex.py``);
all run concurrently. Vertex's OpenAI-compatible endpoint returns google.rpc errors in a list-wrapped
envelope (``[{"error": {"code", "message", "status"}}]``); the fake uses exactly that.

* 429 RESOURCE_EXHAUSTED (quota) and 503 UNAVAILABLE are transient: retried, then the answer.
* 400 INVALID_ARGUMENT and 403 PERMISSION_DENIED are terminal: one request, message shown once.
* 401 UNAUTHENTICATED on every bearer: one token refresh + retry at most, then shown once.
* OAuth ``invalid_grant`` at the token endpoint: nothing is sent to Vertex; actionable error.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("google.auth", reason="Vertex minting needs google-auth (CI installs it)")

from tests.e2e.core.providers._native_helpers import ChatResult, KnownSymptom, make_home, run_chat  # noqa: E402
from tests.fakes.providers.vertex import (  # noqa: E402
    PROJECT,
    REGION,
    SA_EMBEDDED_PROJECT,
    Fail,
    FakeVertex,
    Say,
    TokenPolicy,
    hermes_setup,
)

KNOWN: dict[str, str] = {
    "guidance:permission_denied": "#121295 Vertex 401/403 are reported as 'rejected your API key' (Vertex has no API "
                                  "key; the fix is the service account / IAM role)",
    "guidance:unauthenticated": "#121295 Vertex 401/403 are reported as 'rejected your API key' (Vertex has no API "
                                "key; the fix is the service account / IAM role)",
}

QUOTA_MSG = ("Resource exhausted. Please try again later. Please refer to "
             "https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.")
UNAVAILABLE_MSG = "The service is currently unavailable."
INVALID_MSG = "Request contains an invalid argument."
DENIED_MSG = (f"Permission 'aiplatform.endpoints.predict' denied on resource '//aiplatform.googleapis.com/projects/"
              f"{PROJECT}/locations/{REGION}/publishers/google/models/gemini-3-flash-preview' (or it may not exist).")
UNAUTH_FRAGMENT = "Request had invalid authentication credentials"


@dataclass
class Case:
    script: list[Any]
    policy: TokenPolicy = field(default_factory=TokenPolicy)
    answer: str | None = None  # the final answer when the error is transient
    max_attempts: int = 2  # agent.api_max_retries (total attempts per API call)


CASES: dict[str, Case] = {
    "rate_limited": Case([Fail(429, QUOTA_MSG), Fail(429, QUOTA_MSG), Say("Answer after the quota recovered.")],
                         answer="Answer after the quota recovered.", max_attempts=3),
    "unavailable": Case([Fail(503, UNAVAILABLE_MSG), Say("Answer after UNAVAILABLE.")], answer="Answer after UNAVAILABLE."),
    "invalid_argument": Case([Fail(400, INVALID_MSG)]),
    "permission_denied": Case([Fail(403, DENIED_MSG)]),
    "unauthenticated": Case([Say("never served")], TokenPolicy(reject_bearers=True)),
    "invalid_grant": Case([Say("never served")], TokenPolicy(error=(400, "invalid_grant", "Invalid JWT Signature."))),
}


def _run(tmp: Path, name: str, case: Case) -> dict[str, Any]:
    fake = FakeVertex(tmp / name / "fake", project=PROJECT, region=REGION, sa_project=SA_EMBEDDED_PROJECT,
                      script=list(case.script))
    fake.start()
    fake.token_policy = case.policy
    nh = make_home(tmp / name / "h", **hermes_setup(fake, extra_config={"agent": {"api_max_retries": case.max_attempts}}))
    return {"fake": fake, "nh": nh, "turn": run_chat(nh, "Say hello.", env=fake.child_env(), args=("-t", "file"))}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("vertex_errors")
    with ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        futures = {name: pool.submit(_run, tmp, name, case) for name, case in CASES.items()}
        out = {name: f.result() for name, f in futures.items()}
    yield out
    for res in out.values():
        res["fake"].stop()


def _output(turn: ChatResult) -> str:
    return turn.stdout + turn.stderr


@pytest.mark.parametrize("name", ["rate_limited", "unavailable"])
def test_transient_error_is_retried_then_answers(results: dict[str, Any], name: str) -> None:
    res, case = results[name], CASES[name]
    fake, turn = res["fake"], res["turn"]
    assert turn.returncode == 0 and case.answer in turn.stdout, turn.describe()
    statuses = [r.get("status") for r in fake.requests]
    assert statuses == [f.status for f in case.script if isinstance(f, Fail)] + [200], statuses
    bodies = [r["body"]["messages"] for r in fake.requests]
    assert all(b == bodies[0] for b in bodies), "a retry changed the conversation it resent"
    assert len({r["auth"] for r in fake.requests}) == 1 and len(fake.token_requests) == 1, "retry re-minted needlessly"


@pytest.mark.parametrize(("name", "vendor_text"), [("invalid_argument", INVALID_MSG), ("permission_denied", DENIED_MSG)])
def test_terminal_error_surfaced_once_without_retry(results: dict[str, Any], name: str, vendor_text: str) -> None:
    res = results[name]
    fake, turn = res["fake"], res["turn"]
    assert len(fake.requests) == 1, f"terminal {name} was retried: {[r.get('status') for r in fake.requests]}"
    assert turn.returncode != 0, turn.describe()
    assert _output(turn).count(vendor_text) == 1, f"vendor message not surfaced exactly once:\n{turn.describe()}"


def test_rejected_bearer_refreshes_once_then_surfaces(results: dict[str, Any]) -> None:
    """Every bearer 401s: Hermes may re-mint and retry once, never loop, and shows the error once."""
    res = results["unauthenticated"]
    fake, turn = res["fake"], res["turn"]
    statuses = [r.get("status") for r in fake.requests]
    assert statuses and set(statuses) == {401} and len(statuses) <= 2, statuses
    assert turn.returncode != 0
    assert _output(turn).count(UNAUTH_FRAGMENT) == 1, turn.describe()


def test_oauth_invalid_grant_sends_nothing_and_names_the_credential(results: dict[str, Any]) -> None:
    """The SA key is refused at Google's token endpoint: no Vertex request goes out with a missing
    bearer, and the user is told which credential setting to fix."""
    res = results["invalid_grant"]
    fake, turn = res["fake"], res["turn"]
    assert fake.token_requests and all(t.get("error") == "invalid_grant" for t in fake.token_requests)
    assert fake.token_requests[0]["claims"], "the JWT assertion itself failed verification"
    assert not fake.requests, f"requests reached Vertex without a token: {[r['auth'][:16] for r in fake.requests]}"
    assert turn.returncode != 0
    out = _output(turn)
    assert "VERTEX_CREDENTIALS_PATH" in out or "GOOGLE_APPLICATION_CREDENTIALS" in out, turn.describe()


def _guidance_param(name: str) -> Any:
    return pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN[f"guidance:{name}"]))


@pytest.mark.parametrize("name", [_guidance_param("permission_denied"), _guidance_param("unauthenticated")])
def test_auth_failure_guidance_is_vertex_specific(results: dict[str, Any], name: str) -> None:
    """Vertex authenticates with an OAuth service account, so the guidance must not send the user to
    rotate an 'API key' that does not exist."""
    res = results[name]
    turn = res["turn"]
    if not (res["fake"].requests and turn.returncode != 0):
        raise RuntimeError(f"{name}: the auth failure never happened:\n{turn.describe()}")
    guidance = _output(turn).split("Provider said:")[0].lower()
    if "api key" in guidance:
        raise KnownSymptom(f"Vertex auth failure blamed on an API key:\n{turn.stdout}")
