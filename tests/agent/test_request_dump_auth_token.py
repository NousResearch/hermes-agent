"""request_dump Authorization must reflect the credential actually in use.

OAuth / setup-token Anthropic clients are constructed with ``auth_token=`` and
leave ``api_key=None``; the debug dump used to read only ``api_key`` and
reported ``"Bearer None"`` for fully authenticated requests (#91319).
"""

import json

import pytest

# Placeholder-shaped fake credentials (no real key format), built by
# repetition so the expected masked form is derivable without embedding
# anything that resembles a usable secret.
_FAKE_OAUTH = "oauth-" + "a" * 20          # len 26 → mask: first8...last4
_FAKE_KEY = "key" + "b" * 20               # len 23 → mask: first8...last4


class _StubClient:
    def __init__(self, api_key=None, auth_token=None):
        self.api_key = api_key
        self.auth_token = auth_token


class _StubAgent:
    def __init__(self, tmp_path, client):
        self.client = client
        self.session_id = "sess-dump-test"
        self.logs_dir = tmp_path / "logs"
        self.logs_dir.mkdir(parents=True)
        self.base_url = "https://api.anthropic.com"
        self.api_mode = "anthropic_messages"
        self.verbose_logging = False
        self.log_prefix = "[test] "

    def _vprint(self, *args, **kwargs):
        return None

    def _mask_api_key_for_logs(self, key):
        # Delegate to the production masker so the stub cannot drift from
        # the real masking contract (length thresholds, sentinels).
        from run_agent import AIAgent

        return AIAgent._mask_api_key_for_logs(self, key)


@pytest.fixture(autouse=True)
def _identity_safe_sid(monkeypatch):
    import run_agent
    monkeypatch.setattr(
        run_agent, "_safe_session_filename_component", lambda sid: sid,
    )


def _dump(tmp_path, client):
    from agent.agent_runtime_helpers import dump_api_request_debug
    agent = _StubAgent(tmp_path, client)
    path = dump_api_request_debug(agent, {"model": "claude-x", "messages": []}, reason="test")
    assert path is not None and path.exists()
    return json.loads(path.read_text())


def _masked(value: str) -> str:
    return f"{value[:8]}...{value[-4:]}"


class TestRequestDumpAuthorization:

    def test_oauth_auth_token_not_reported_as_none(self, tmp_path):
        """OAuth credential (auth_token=, api_key=None) must show up masked,
        not as the misleading "Bearer None"."""
        payload = _dump(tmp_path, _StubClient(api_key=None, auth_token=_FAKE_OAUTH))
        assert payload["request"]["headers"]["Authorization"] != "Bearer None"
        assert payload["request"]["headers"]["Authorization"] == f"Bearer {_masked(_FAKE_OAUTH)}"

    def test_api_key_path_unchanged(self, tmp_path):
        """Plain api_key credentials keep the existing masked output."""
        payload = _dump(tmp_path, _StubClient(api_key=_FAKE_KEY))
        assert payload["request"]["headers"]["Authorization"] == f"Bearer {_masked(_FAKE_KEY)}"

    def test_api_key_wins_over_auth_token(self, tmp_path):
        """When both are present the SDK sends api_key — the dump must match."""
        payload = _dump(
            tmp_path,
            _StubClient(api_key=_FAKE_KEY, auth_token=_FAKE_OAUTH),
        )
        assert payload["request"]["headers"]["Authorization"] == f"Bearer {_masked(_FAKE_KEY)}"

    def test_no_credentials_still_reports_none(self, tmp_path):
        """No credential at all keeps the explicit "Bearer None" marker."""
        payload = _dump(tmp_path, _StubClient(api_key=None, auth_token=None))
        assert payload["request"]["headers"]["Authorization"] == "Bearer None"

    def test_callable_auth_token_reports_entra_id_sentinel(self, tmp_path):
        """A callable auth_token (Entra ID bearer provider) resolves through
        the fallback to the documented sentinel — the callable itself is
        never invoked in the log path."""
        payload = _dump(
            tmp_path, _StubClient(api_key=None, auth_token=lambda: _FAKE_OAUTH)
        )
        assert (
            payload["request"]["headers"]["Authorization"] == "Bearer <entra-id-bearer>"
        )
