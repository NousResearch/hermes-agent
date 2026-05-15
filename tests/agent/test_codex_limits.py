import io
import json
import urllib.error

from agent import codex_limits


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _http_error(code):
    return urllib.error.HTTPError(
        "https://chatgpt.com/backend-api/wham/usage",
        code,
        "auth failed",
        hdrs=None,
        fp=io.BytesIO(b"{}"),
    )


def test_codex_limits_app_server_fixture_formats_model_buckets():
    provider, payload = codex_limits.fixture_payload("app-server")

    state = codex_limits.normalize(provider, payload)
    pretty = codex_limits.format_pretty(state)

    assert state["source"]["provider"] == "app_server"
    assert [bucket["limit_id"] for bucket in state["rate_limits"]] == ["codex", "codex_model"]
    assert state["rate_limits"][0]["five_h"]["remaining_percent"] == 75
    assert "Rate limits remaining: 5h 75% reset" in pretty
    assert "GPT-Codex model bucket: 5h 89% reset" in pretty


def test_codex_limits_wham_fixture_normalizes_used_to_remaining():
    provider, payload = codex_limits.fixture_payload("wham")

    state = codex_limits.normalize(provider, payload)

    bucket = state["rate_limits"][0]
    assert state["source"]["provider"] == "codex_wham"
    assert bucket["five_h"]["remaining_percent"] == 76
    assert bucket["week"]["remaining_percent"] == 93


def test_codex_limits_json_contains_no_raw_auth_material():
    provider, payload = codex_limits.fixture_payload("app-server")

    encoded = json.dumps(codex_limits.normalize(provider, payload))

    assert "access_token" not in encoded
    assert "refresh_token" not in encoded
    assert "account_id" not in encoded
    assert "chatgpt_account_id" not in encoded


def test_resolve_hermes_codex_auth_prefers_credential_pool(monkeypatch):
    class FakeEntry:
        access_token = "pool-token"
        label = "openai-second"
        id = "second"

    class FakePool:
        def select(self):
            return FakeEntry()

        def try_refresh_current(self):
            raise AssertionError("should not force-refresh unless requested")

    import agent.credential_pool as credential_pool
    import hermes_cli.auth as hermes_auth

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: FakePool())
    monkeypatch.setattr(
        hermes_auth,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("singleton auth should not be used")),
    )

    auth = codex_limits.resolve_hermes_codex_auth()

    assert auth.access_token == "pool-token"
    assert auth.source == "credential-pool:openai-second"


def test_fetch_wham_retries_once_with_refreshed_auth_on_401(monkeypatch):
    calls = []

    def fake_auth(*, force_refresh=False):
        calls.append(force_refresh)
        token = "fresh-token" if force_refresh else "stale-token"
        return codex_limits.AuthMaterial(access_token=token, account_id="acct")

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise _http_error(401)
        return _Response({
            "plan_type": "pro",
            "rate_limit": {
                "primary_window": {"used_percent": 12, "limit_window_seconds": 18000},
            },
        })

    monkeypatch.setattr(codex_limits, "resolve_hermes_codex_auth", fake_auth)
    monkeypatch.setattr(codex_limits.urllib.request, "urlopen", fake_urlopen)

    payload = codex_limits.fetch_wham(timeout=5, usage_url="https://example.test/usage", auth_paths=[])

    assert payload["plan_type"] == "pro"
    assert calls == [False, True]
    assert requests[0].headers["Authorization"] == "Bearer stale-token"
    assert requests[1].headers["Authorization"] == "Bearer fresh-token"
