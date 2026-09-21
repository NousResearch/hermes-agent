"""Tests for the LLM chain and the topic-leak sanitiser
(no raw git commit subjects in topics/titles)."""
import time
import pathlib

import pytest
import requests

import llm_generate as lg
import topics as tp


# ── LLM chain ──────────────────────────────────────────────────────────────

_BUILTIN_KEY_VARS = (
    "COMMANDCODE_API_KEY",
    "OLLAMA_API_KEY",
    "NVIDIA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_AI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENCODE_GO_API_KEY",
    "CONTENT_LLM_API_KEY",
    "OPENCODE_API_KEY",
    "OPENCODE_ZEN_API_KEY",
)


def test_llm_configs_use_governed_config_and_runtime_resolution(monkeypatch):
    config = {
        "model": {"provider": "primary", "default": "exact-model"},
        "fallback_providers": [
            {"provider": "fallback-a", "model": "exact-model"},
            {"provider": "fallback-c", "model": "exact-model"},
            {"provider": "fallback-b", "model": "exact-model", "base_url": "https://b.test/v1"},
        ],
    }
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": kwargs["requested"],
            "base_url": kwargs.get("explicit_base_url") or f"https://{kwargs['requested']}.test/v1",
            "api_key": f"pool-{kwargs['requested']}",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(lg, "_load_hermes_config", lambda: config)
    monkeypatch.setattr(lg, "_resolve_runtime", resolve)

    assert lg._llm_configs() == [
        {"base": "https://primary.test/v1", "model": "exact-model", "key": "pool-primary", "provider": "primary", "api_mode": "chat_completions"},
        {"base": "https://fallback-a.test/v1", "model": "exact-model", "key": "pool-fallback-a", "provider": "fallback-a", "api_mode": "chat_completions"},
        {"base": "https://fallback-c.test/v1", "model": "exact-model", "key": "pool-fallback-c", "provider": "fallback-c", "api_mode": "chat_completions"},
        {"base": "https://b.test/v1", "model": "exact-model", "key": "pool-fallback-b", "provider": "fallback-b", "api_mode": "chat_completions"},
    ]
    assert [call["target_model"] for call in calls] == [
        "exact-model", "exact-model", "exact-model", "exact-model",
    ]


def test_content_config_uses_surface_override_and_ignores_root(monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home

    # The fixture tree must NOT live under the real ~/.hermes: on hosts where
    # pytest's tmpdir sits inside the native Hermes root, get_default_hermes_root()
    # collapses HERMES_HOME back to the real root (it assumes profile mode), so
    # profile resolution would see the operator's real profiles tree instead of
    # the fixture. /tmp is outside every platform-native Hermes root.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory(dir="/tmp") as _td:
        root = pathlib.Path(_td) / "hermes"
        surface = root / "profiles" / "content-strategist"
        surface.mkdir(parents=True)
        # A named profile only counts as live when it carries an identity marker
        # (config.yaml/.env/SOUL.md/...). An empty dir resolves as "unavailable"
        # under current profile semantics.
        (surface / "profile.yaml").touch()
        monkeypatch.setenv("HERMES_HOME", str(root))

        seen = []
        governed = {"model": {"provider": "content", "default": "deepseek-v4-flash"}}

        def canonical_loader():
            seen.append(get_hermes_home())
            return governed if get_hermes_home() == surface else {
                "model": {"provider": "root", "default": "root-model"}
            }

        assert lg._load_hermes_config(config_loader=canonical_loader) == governed
        assert seen == [surface]
        assert get_hermes_home() == root


def test_content_config_missing_or_invalid_surface_fails_closed(monkeypatch, tmp_path):
    # The temp home must NOT live under the real ~/.hermes: get_default_hermes_root()
    # collapses any HERMES_HOME inside the native root back to the real root (it
    # assumes profile mode), so the fixture's "missing root" would resolve to the
    # operator's real profiles tree and never raise. Use /tmp directly — the
    # machine's TMPDIR may itself point inside the Hermes root.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory(dir="/tmp") as _td:
        missing_root = pathlib.Path(_td) / "missing-root"
        monkeypatch.setenv("HERMES_HOME", str(missing_root))
        with pytest.raises(RuntimeError, match="content-strategist.*unavailable"):
            lg._load_hermes_config()
    with pytest.raises(RuntimeError, match="content-strategist.*invalid config"):
        lg._load_hermes_config(home_resolver=lambda: tmp_path, config_loader=lambda: [])


def test_llm_configs_exclude_cross_model_fallback(monkeypatch):
    monkeypatch.setattr(lg, "_load_hermes_config", lambda: {
        "model": {"provider": "primary", "default": "deepseek-v4-flash"},
        "fallback_providers": [{"provider": "other", "model": "root-model"}],
    })
    monkeypatch.setattr(lg, "_resolve_runtime", lambda **kwargs: {
        "base_url": "https://example.test/v1", "api_key": "key",
    })
    assert [route["model"] for route in lg._llm_configs()] == ["deepseek-v4-flash"]


def test_llm_configs_do_not_filter_routes_by_environment(monkeypatch):
    monkeypatch.delenv("UNCONFIGURED_PROVIDER_KEY", raising=False)
    monkeypatch.setattr(lg, "_load_hermes_config", lambda: {
        "model": {"provider": "custom-provider", "default": "m"},
        "fallback_providers": [],
    })
    monkeypatch.setattr(lg, "_resolve_runtime", lambda **kwargs: {
        "provider": "custom", "base_url": "http://localhost:11434/v1",
        "api_key": "no-key-required", "api_mode": "chat_completions",
    })

    assert len(lg._llm_configs()) == 1


def test_call_llm_chain_attempts_every_governed_route(monkeypatch):
    configs = [
        {"provider": "one", "base": "https://one.test/v1", "model": "m", "key": "a"},
        {"provider": "two", "base": "https://two.test/v1", "model": "m", "key": "b"},
        {"provider": "three", "base": "https://three.test/v1", "model": "m", "key": "c"},
    ]
    attempted = []
    monkeypatch.setattr(lg, "_llm_configs", lambda longform=False: configs)
    monkeypatch.setattr(
        lg, "_call_llm",
        lambda _system, _user, cfg, **_kwargs: (
            attempted.append(cfg["provider"]) or ("done" if cfg["provider"] == "three" else None)
        ),
    )

    assert lg._call_llm_chain("system", "user") == "done"
    assert attempted == ["one", "two", "three"]


class _FakeEntry:
    """Mimics the PooledCredential attributes the chain reads."""

    def __init__(self, entry_id, key):
        self.id = entry_id
        self.runtime_api_key = key


class _FakePool:
    """Minimal credential pool double recording rotation calls."""

    def __init__(self, entries):
        self._entries = list(entries)
        self.calls = []

    def mark_exhausted_and_rotate(self, **kwargs):
        self.calls.append(kwargs)
        if self._entries:
            return self._entries.pop(0)
        return None


def test_call_llm_chain_rotates_pool_on_zero_credit_http_400(monkeypatch):
    """Zero-credit failures arrive as HTTP 400; the same provider's next
    credential must be tried before advancing to the next route (the
    blog-backlog-pregen balance=0 regression)."""
    pool = _FakePool([_FakeEntry("cred-2", "key-b")])
    configs = [{
        "provider": "p", "base": "https://p.test/v1", "model": "m",
        "key": "key-a", "_credential_pool": pool, "_credential_id": "cred-1",
    }]
    bodies = []
    monkeypatch.setattr(lg, "_llm_configs", lambda longform=False: configs)

    def fake_call(_system, _user, cfg, **_kwargs):
        bodies.append(cfg["key"])
        if cfg["key"] == "key-a":
            cfg["_failure_status"] = 400
            cfg["_failure_reason"] = "insufficient_credits"
            return None
        return "recovered-on-second-credential"

    monkeypatch.setattr(lg, "_call_llm", fake_call)

    assert lg._call_llm_chain("system", "user") == "recovered-on-second-credential"
    assert bodies == ["key-a", "key-b"]
    assert pool.calls == [{
        "status_code": 400, "credential_id": "cred-1",
        "api_key_hint": "key-a", "failure_reason": "insufficient_credits",
    }]


def test_call_llm_chain_does_not_retry_plain_http_400(monkeypatch):
    """A 400 without credit wording is a request problem: no pool rotation,
    no second attempt on the same provider."""
    pool = _FakePool([_FakeEntry("cred-2", "key-b")])
    configs = [{
        "provider": "p", "base": "https://p.test/v1", "model": "m",
        "key": "key-a", "_credential_pool": pool, "_credential_id": "cred-1",
    }]
    calls = []
    monkeypatch.setattr(lg, "_llm_configs", lambda longform=False: configs)

    def fake_call(_system, _user, cfg, **_kwargs):
        calls.append(cfg["key"])
        cfg["_failure_status"] = 400
        cfg["_failure_reason"] = None
        return None

    monkeypatch.setattr(lg, "_call_llm", fake_call)

    assert lg._call_llm_chain("system", "user") is None
    assert calls == ["key-a"]
    assert pool.calls == []


def test_classify_failure_reason_zero_credit_variants():
    assert lg._classify_failure_reason(400, "credit insufficient balance: balance=0") == "insufficient_credits"
    assert lg._classify_failure_reason(400, "Insufficient Balance in account") == "insufficient_credits"
    assert lg._classify_failure_reason(400, "quota exhausted for this key") == "insufficient_credits"
    assert lg._classify_failure_reason(402, "payment required") == "insufficient_credits"
    assert lg._classify_failure_reason(401, "unauthorised") == "auth"
    assert lg._classify_failure_reason(403, "forbidden") == "auth"
    assert lg._classify_failure_reason(429, "slow down") == "rate_limit"
    assert lg._classify_failure_reason(400, "invalid model name") is None
    assert lg._classify_failure_reason(500, "internal error") is None


def test_call_llm_records_failure_reason_on_http_error(monkeypatch):
    """_call_llm must stamp _failure_reason so the chain can classify."""
    class Response:
        status_code = 400
        text = "credit insufficient balance: balance=0"

        @staticmethod
        def json():
            return {}

    class Session:
        trust_env = True

        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(requests, "Session", Session)
    cfg = {"base": "https://example.test/v1", "model": "test", "key": "test"}
    assert lg._call_llm("system", "user", cfg) is None
    assert cfg["_failure_status"] == 400
    assert cfg["_failure_reason"] == "insufficient_credits"


def _clear_builtin_keys(monkeypatch):
    for name in _BUILTIN_KEY_VARS:
        monkeypatch.delenv(name, raising=False)

def test_longform_does_not_create_an_app_owned_route_tier(monkeypatch):
    monkeypatch.setattr(lg, "_load_hermes_config", lambda: {
        "model": {"provider": "p", "default": "same-model"},
        "fallback_providers": [],
    })
    monkeypatch.setattr(lg, "_resolve_runtime", lambda **_kwargs: {
        "provider": "p", "base_url": "https://p.test/v1", "api_key": "pool-key",
        "api_mode": "chat_completions",
    })
    assert lg._llm_configs(longform=True) == lg._llm_configs(longform=False)


def test_fabricated_numbers_catches_growth_metrics():
    import article_gates as ag
    assert ag._fabricated_numbers("Just hit 1,000 users this week", "")
    assert ag._fabricated_numbers("200 downloads in a day", "")
    assert ag._fabricated_numbers("conversion rose 45%", "")
    assert not ag._fabricated_numbers("I cut 3 features in 2 weeks", "")
    assert not ag._fabricated_numbers("we hit 1,000 users", "the launch reached 1,000 users")


def test_call_llm_retries_transient_http_error(monkeypatch):
    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    responses = [
        Response(503, text="temporarily unavailable"),
        Response(200, {"choices": [{"message": {"content": "recovered"}}]}),
    ]

    class Session:
        trust_env = True

        def post(self, *_args, **_kwargs):
            return responses.pop(0)

    session = Session()
    sleeps = []
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    result = lg._call_llm(
        "system",
        "user",
        {"base": "https://example.test/v1", "model": "test", "key": "test"},
    )

    assert result == "recovered"
    assert sleeps == [2.0]
    assert responses == []
    assert session.trust_env is False


def test_call_llm_retries_transient_transport_error(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "recovered"}}]}

    calls = []

    class Session:
        trust_env = True

        def post(self, *_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise requests.ConnectionError("temporary connection reset")
            return Response()

    sleeps = []
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    result = lg._call_llm(
        "system",
        "user",
        {"base": "https://example.test/v1", "model": "test", "key": "test"},
    )

    assert result == "recovered"
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_call_llm_malformed_json_degrades_to_none(monkeypatch):
    class Response:
        status_code = 200
        text = "not-json"

        @staticmethod
        def json():
            raise ValueError("malformed provider response")

    class Session:
        trust_env = True

        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(requests, "Session", Session)

    assert lg._call_llm(
        "system",
        "user",
        {"base": "https://example.test/v1", "model": "test", "key": "test"},
    ) is None


def test_call_llm_does_not_retry_auth_failure(monkeypatch):
    class Response:
        status_code = 401
        text = "unauthorised"

        @staticmethod
        def json():
            return {}

    calls = []

    class Session:
        trust_env = True

        def post(self, *_args, **_kwargs):
            calls.append(1)
            return Response()

    sleeps = []
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    result = lg._call_llm(
        "system",
        "user",
        {"base": "https://example.test/v1", "model": "test", "key": "bad"},
    )

    assert result is None
    assert len(calls) == 1
    assert sleeps == []


# ── Topic-leak sanitiser ───────────────────────────────────────────────────

def test_strips_conventional_commit_prefix():
    assert tp._clean_topic_summary(
        "feat(content): non-infographic scene transplant") == "non-infographic scene transplant"
    assert tp._clean_topic_summary("fix: gateway crash loop") == "gateway crash loop"
    assert tp._clean_topic_summary("refactor(api)!: drop v1") == "drop v1"


def test_strips_trailer():
    assert tp._clean_topic_summary(
        "feat: add thing\n\nCo-Authored-By: X <x@y.z>") == "add thing"


def test_merge_commit_yields_empty():
    assert tp._clean_topic_summary("Merge branch 'main' into feature") == ""


def test_clean_passes_through_normal_text():
    assert tp._clean_topic_summary("How I cut agent context 99x") == "How I cut agent context 99x"


def test_no_commit_prefix_in_built_topic():
    cleaned = tp._clean_topic_summary("chore(deps): bump three to 0.184")
    assert not cleaned.lower().startswith("chore")
    assert cleaned == "bump three to 0.184"
