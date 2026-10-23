"""Auxiliary calls must not keep presenting a Claude Code token that was already replaced.

The compressor (and every other auxiliary task) routes onto the main session with the
session's own key, captured when the agent was built. A Claude Code subscription token is
refreshed by `claude` every few hours, and since the main client mints its bearer per
request it never 401s — so nothing ever updates the captured key. Hours later `/compress`
sends the superseded token and the API answers 401 "OAuth access token has been revoked"
while the token on disk works. Measured 2026-10-07: 17 compression failures in one session,
none in the conversation itself.
"""
import pytest

from agent import auxiliary_client as aux
from agent.auxiliary_claude_code_token import supersede_stale_anthropic_oauth_key

LIVE = "sk-ant-oat01-" + "L" * 80 + "AA"
STALE = "sk-ant-oat01-" + "S" * 80 + "AA"
NATIVE = "https://api.anthropic.com"


class _Pool:
    def __init__(self, owned=()):
        self._owned = set(owned)

    def entry_id_for_api_key(self, key):
        return "owned" if key in self._owned else None


@pytest.fixture
def disk(monkeypatch):
    """The Claude Code credential currently on disk."""
    state = {"creds": {"accessToken": LIVE, "refreshToken": "r", "expiresAt": 0, "source": "test"},
             "pool": _Pool()}
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials",
                        lambda: state["creds"])
    monkeypatch.setattr(aux, "load_pool", lambda provider: state["pool"])
    return state


def test_a_superseded_session_token_is_replaced_by_the_one_on_disk(disk):
    """The regression: the session hands over the token it was built with."""
    assert supersede_stale_anthropic_oauth_key(STALE, NATIVE) == LIVE


def test_the_main_route_builds_the_aux_client_with_the_live_token(disk, monkeypatch):
    """End to end through the route the compressor takes, up to the client factory."""
    built = []
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client",
                        lambda token, base_url=None, **kw: built.append(token) or object())
    monkeypatch.setattr(aux, "_aux_probe_active", lambda: False)
    monkeypatch.setattr(aux, "_select_pool_entry", lambda provider: (False, None))

    aux._try_main_provider_route("anthropic", "claude-opus-5-5", NATIVE, STALE, "anthropic_messages")

    assert built and built[-1] == LIVE, f"aux client built with {built!r}"


def test_the_current_token_is_kept(disk):
    assert supersede_stale_anthropic_oauth_key(LIVE, NATIVE) == LIVE


def test_a_token_the_pool_owns_is_kept(disk):
    """A second account added with `hermes auth add` is pinned on purpose: never swap it."""
    disk["pool"] = _Pool(owned={STALE})
    assert supersede_stale_anthropic_oauth_key(STALE, NATIVE) == STALE


def test_an_api_key_is_kept(disk):
    """Console API keys do not rotate; the disk token belongs to someone else's billing."""
    key = "sk-ant-api03-" + "K" * 80
    assert supersede_stale_anthropic_oauth_key(key, NATIVE) == key


def test_a_third_party_anthropic_endpoint_is_kept(disk):
    """Third-party anthropic_messages providers must never pick up the ~/.claude token (#1739)."""
    assert supersede_stale_anthropic_oauth_key(STALE, "https://api.minimax.io/anthropic") == STALE


def test_no_credential_on_disk_keeps_the_session_key(disk):
    disk["creds"] = None
    assert supersede_stale_anthropic_oauth_key(STALE, NATIVE) == STALE


def test_the_401_refresher_reports_a_newer_token_on_disk(disk):
    """Second line of defence: a 401 on a superseded key is recoverable, not final.

    The token can be replaced between building the client and sending the request. The
    refresher used to answer False here — "not the current token, not in the pool" — and the
    call raised with a working credential one file read away.
    """
    assert aux._refresh_anthropic_credentials(STALE) is True


def test_the_401_refresher_does_not_claim_recovery_for_an_api_key(disk):
    assert aux._refresh_anthropic_credentials("sk-ant-api03-" + "K" * 80) is False
