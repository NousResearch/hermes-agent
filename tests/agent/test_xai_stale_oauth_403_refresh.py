"""Regression tests: xAI signals an expired OAuth token with 403, not 401.

xAI answers an access token that has simply aged out with HTTP 403 and a
body carrying either ``[WKE=unauthenticated:...]`` or ``OAuth2 access token
could not be validated`` -- never a 401.  #29344 taught the credential-pool
recovery path to tell those bodies apart from the entitlement 403s that a
refresh can never fix, but the singleton ``codex_responses`` refresh branch
in ``run_conversation`` kept gating on ``status_code == 401`` alone.

Consequence in the field: a long-running gateway on ``xai-oauth`` held a
token that died ~6h earlier and had no path back.  Neither recovery arm
fired -- the pool arm returns early whenever the agent carries no
credential pool, and the singleton arm could not see a 403 -- so Grok
stayed dead until the process was restarted.  ``hermes update`` restarts
the stack and re-mints, which is why updating appeared to "fix Grok" for
exactly one token lifetime at a time.

These lock in the shared predicate and the widened gate.
"""

import re
from pathlib import Path

import pytest

from agent.agent_runtime_helpers import is_xai_stale_oauth_error


# The exact body observed in production gateway logs.
REAL_EXPIRED_BODY = (
    'HTTP 403: {"code":"unauthenticated:bad-credentials",'
    '"error":"The OAuth2 access token could not be validated."}'
)

ENTITLEMENT_BODY = (
    "You have either run out of available resources or do not have an "
    "active Grok subscription"
)


@pytest.mark.parametrize(
    "error_context,status_code,expected",
    [
        # -- recoverable: the token is merely stale -------------------------
        ({"message": REAL_EXPIRED_BODY}, 403, True),
        ({"message": "boom [WKE=unauthenticated:bad-credentials]"}, 403, True),
        ({"error": "The OAuth2 access token could not be validated."}, 403, True),
        # xAI has shipped both casings; the match must be case-insensitive
        ({"message": "OAUTH2 ACCESS TOKEN COULD NOT BE VALIDATED"}, 403, True),
        # -- NOT recoverable: refreshing cannot fix these -------------------
        ({"message": ENTITLEMENT_BODY}, 403, False),
        (
            {
                "message": "oauth authentication is currently not allowed "
                "for this organization"
            },
            403,
            False,
        ),
        # -- wrong status: the predicate is 403-only ------------------------
        ({"message": REAL_EXPIRED_BODY}, 401, False),
        ({"message": REAL_EXPIRED_BODY}, 429, False),
        ({"message": REAL_EXPIRED_BODY}, None, False),
        # -- degenerate input never raises ----------------------------------
        (None, 403, False),
        ({}, 403, False),
        ("not-a-dict", 403, False),
        ({"message": None, "code": None}, 403, False),
    ],
)
def test_stale_oauth_predicate(error_context, status_code, expected):
    assert is_xai_stale_oauth_error(error_context, status_code) is expected


def test_singleton_refresh_gate_accepts_the_403():
    """The codex_responses refresh branch must not be 401-only anymore.

    Asserted against source text because exercising the branch for real
    means standing up the whole streaming conversation loop; the gate is a
    four-line boolean whose regression mode is silent.
    """
    src = Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
    body = src.read_text(encoding="utf-8")

    gate = re.search(
        r"if \(\s*\n\s*agent\.api_mode == .codex_responses.\s*\n"
        r"\s*and agent\.provider in \{.openai-codex., .xai-oauth.\}\s*\n"
        r"\s*and ([^\n]+)\n",
        body,
    )
    assert gate, "codex_responses auth-refresh gate not found"
    condition = gate.group(1)
    assert "401" in condition, "401 handling must be preserved"
    assert "_xai_stale_oauth" in condition, (
        "the gate still ignores xAI's 403 expiry signal -- a stale "
        "xai-oauth token has no recovery path"
    )


def test_pool_path_reuses_the_shared_predicate():
    """The pool arm must call the helper, not keep a second copy of it."""
    src = Path(__file__).resolve().parents[2] / "agent" / "agent_runtime_helpers.py"
    body = src.read_text(encoding="utf-8")

    recover = body[body.index("def recover_with_credential_pool(") :]
    assert "is_xai_stale_oauth_error(error_context, status_code)" in recover
    assert "_is_xai_auth_failure" not in recover, (
        "inline duplicate of the predicate is back; the two arms will drift"
    )


def test_oauth_active_key_is_foreign_pool_entry():
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import oauth_active_key_is_foreign_pool_entry

    assert oauth_active_key_is_foreign_pool_entry(SimpleNamespace(_credential_pool=None), "k") is False

    class _Pool:
        def entries(self):
            return [SimpleNamespace(runtime_api_key="pooled-key")]

    agent = SimpleNamespace(_credential_pool=_Pool())
    assert oauth_active_key_is_foreign_pool_entry(agent, "pooled-key") is True
    assert oauth_active_key_is_foreign_pool_entry(agent, "stale-in-memory") is False
    assert oauth_active_key_is_foreign_pool_entry(agent, "") is False


def test_resolve_stale_oauth_pool_entry_does_not_guess_manual():
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import resolve_stale_oauth_pool_entry

    manual = SimpleNamespace(id="manual-1", source="manual", runtime_api_key="other")
    device = SimpleNamespace(id="dc-1", source="device_code", runtime_api_key="tok")

    class _Pool:
        def __init__(self, entries):
            self._entries = entries

        def entries(self):
            return self._entries

    assert resolve_stale_oauth_pool_entry(
        SimpleNamespace(_credential_pool_entry_id=None), _Pool([manual])
    ) is None
    assert resolve_stale_oauth_pool_entry(
        SimpleNamespace(_credential_pool_entry_id="manual-1"), _Pool([manual])
    ) is manual
    assert resolve_stale_oauth_pool_entry(
        SimpleNamespace(_credential_pool_entry_id=None), _Pool([device])
    ) is device
    imported = SimpleNamespace(
        id="dc-imported", source="manual:device_code", runtime_api_key="tok"
    )
    assert resolve_stale_oauth_pool_entry(
        SimpleNamespace(_credential_pool_entry_id=None), _Pool([imported])
    ) is imported


def _xai_agent(*, api_key="expired-in-memory-token"):
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = AIAgent(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        model="grok-4.6",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "xai-oauth"
    agent._interrupt_requested = False
    agent._credential_pool_entry_id = None
    agent._swap_credential = MagicMock()
    return agent


def test_recover_adopts_reminted_device_code_entry_without_second_refresh():
    """8:34 hole: expired live key, sole device_code entry already reminted."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent.error_classifier import FailoverReason

    agent = _xai_agent()
    calls = []
    only = SimpleNamespace(
        id="only-entry",
        source="device_code",
        runtime_api_key="fresh-store-token",
    )

    class _FakePool:
        provider = "xai-oauth"

        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            calls.append({"api_key_hint": api_key_hint, "credential_id": credential_id})
            return None

        def mark_exhausted_and_rotate(self, **_kwargs):
            raise AssertionError("must not rotate when the reminted entry can be adopted")

        def entries(self):
            return [only]

        def has_available(self):
            return True

        def current(self):
            return None

    agent._credential_pool = _FakePool()
    recovered, _ = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"message": REAL_EXPIRED_BODY},
    )

    assert recovered is True
    assert calls == []
    agent._swap_credential.assert_called_once_with(only)


def test_recover_does_not_guess_sole_manual_other_account():
    from types import SimpleNamespace

    from agent.error_classifier import FailoverReason

    agent = _xai_agent()
    calls = []
    manual = SimpleNamespace(
        id="manual-other",
        source="manual",
        runtime_api_key="other-account-token",
    )

    class _FakePool:
        provider = "xai-oauth"

        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            calls.append({"api_key_hint": api_key_hint, "credential_id": credential_id})
            return None

        def mark_exhausted_and_rotate(self, **_kwargs):
            return None

        def entries(self):
            return [manual]

        def has_available(self):
            return True

        def current(self):
            return None

    agent._credential_pool = _FakePool()
    recovered, _ = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"message": REAL_EXPIRED_BODY},
    )

    assert recovered is False
    assert all(c["credential_id"] is None for c in calls)
    agent._swap_credential.assert_not_called()


def test_recover_adopts_reminted_entry_by_credential_id():
    from types import SimpleNamespace

    from agent.error_classifier import FailoverReason

    agent = _xai_agent()
    agent._credential_pool_entry_id = "manual-1"
    reminted = SimpleNamespace(
        id="manual-1",
        source="manual",
        runtime_api_key="fresh-manual-token",
    )
    calls = []

    class _FakePool:
        provider = "xai-oauth"

        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            calls.append({"api_key_hint": api_key_hint, "credential_id": credential_id})
            return None

        def mark_exhausted_and_rotate(self, **_kwargs):
            raise AssertionError("must adopt the reminted id, not rotate")

        def entries(self):
            return [reminted]

        def has_available(self):
            return True

        def current(self):
            return None

    agent._credential_pool = _FakePool()
    recovered, _ = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"message": REAL_EXPIRED_BODY},
    )

    assert recovered is True
    agent._swap_credential.assert_called_once_with(reminted)
    assert calls == []


def test_recover_refreshes_device_code_entry_when_runtime_still_matches():
    from types import SimpleNamespace

    from agent.error_classifier import FailoverReason

    agent = _xai_agent(api_key="same-expired-token")
    calls = []
    refreshed = SimpleNamespace(id="dc-1")
    only = SimpleNamespace(
        id="dc-1",
        source="device_code",
        runtime_api_key="same-expired-token",
    )

    class _FakePool:
        provider = "xai-oauth"

        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            calls.append({"api_key_hint": api_key_hint, "credential_id": credential_id})
            if credential_id == "dc-1":
                return refreshed
            return None

        def mark_exhausted_and_rotate(self, **_kwargs):
            raise AssertionError("must refresh the device_code entry by id")

        def entries(self):
            return [only]

        def has_available(self):
            return True

        def current(self):
            return None

    agent._credential_pool = _FakePool()
    recovered, _ = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"message": REAL_EXPIRED_BODY},
    )

    assert recovered is True
    assert calls == [
        {"api_key_hint": "same-expired-token", "credential_id": "dc-1"}
    ]
    agent._swap_credential.assert_called_once_with(refreshed)

