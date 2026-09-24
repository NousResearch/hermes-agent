"""``PooledCredential.runtime_base_url`` must re-read NOUS_INFERENCE_BASE_URL on every use.

With the documented dev/staging override set, a Nous 401 recovery refreshed the token and
retried against the stored production host instead of the override: the rotated bearer was
sent to a host the operator never chose, and any non-production gateway's turn failed with a
connection error. The stored ``inference_base_url`` cannot carry the override anyway — the
network-provenance allowlist heals that field back to production — so the only trustworthy
source is the profile-scoped env resolver, consulted at read time, exactly like the
openai-codex branch below it treats HERMES_CODEX_BASE_URL.
"""

from agent.credential_pool import PooledCredential

PRODUCTION = "https://inference-api.nousresearch.com/v1"


def _nous_entry(**kwargs) -> PooledCredential:
    defaults = dict(
        provider="nous",
        id="nous-1",
        label="nous",
        auth_type="oauth",
        priority=0,
        source="manual",
        access_token="token",
    )
    defaults.update(kwargs)
    return PooledCredential(**defaults)


def test_override_wins_over_healed_production_inference_url(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1")
    entry = _nous_entry(inference_base_url=PRODUCTION, base_url=PRODUCTION)
    assert entry.runtime_base_url == "http://127.0.0.1:8443/v1"


def test_without_override_inference_base_url_wins(monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    entry = _nous_entry(inference_base_url=PRODUCTION, base_url="https://fallback.example/v1")
    assert entry.runtime_base_url == PRODUCTION


def test_without_override_or_inference_falls_back_to_base_url(monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    entry = _nous_entry(base_url="https://fallback.example/v1")
    assert entry.runtime_base_url == "https://fallback.example/v1"


def test_override_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1/")
    entry = _nous_entry(inference_base_url=PRODUCTION)
    assert entry.runtime_base_url == "http://127.0.0.1:8443/v1"


def test_blank_override_falls_back_to_row_urls(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "   ")
    entry = _nous_entry(inference_base_url=PRODUCTION)
    assert entry.runtime_base_url == PRODUCTION


def test_other_providers_still_return_base_url(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "http://127.0.0.1:8443/v1")
    entry = PooledCredential(
        provider="openai",
        id="oa-1",
        label="openai",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-x",
        base_url="https://api.openai.com/v1",
    )
    assert entry.runtime_base_url == "https://api.openai.com/v1"
