"""Runtime self-heal for custom OpenAI-compatible base URLs missing ``/v1``:
verify ``{base}/v1/models`` answers, rewrite the session's base_url, retry.
Verdicts are cached per base URL. See ``_maybe_apply_v1_suffix_fallback``."""

import time

import pytest

from agent.conversation_loop import (
    _V1_PROBE_RETRY_SECONDS,
    _maybe_apply_v1_suffix_fallback,
)


class _StubClient:
    def __init__(self, base_url):
        self.base_url = base_url


class _StubAgent:
    log_prefix = ""

    def __init__(
        self,
        base_url="http://127.0.0.1:39080",
        api_mode="chat_completions",
        api_key="sk-local",
        azure=False,
    ):
        self.base_url = base_url
        self.api_mode = api_mode
        self.api_key = api_key
        self._azure = azure
        self._client_kwargs = {"api_key": api_key, "base_url": base_url}
        self.client = _StubClient(base_url)
        self.vprints = []

    def _is_azure_openai_url(self, base_url=None):
        return self._azure

    def _vprint(self, message, force=False):
        self.vprints.append(message)


def _models_ok(url, headers=None, timeout=None, **kwargs):
    class _Resp:
        ok = True

        def json(self):
            return {"data": [{"id": "deepseek/deepseek-v4-pro"}]}

    assert url == "http://127.0.0.1:39080/v1/models"
    return _Resp()


def _models_404(url, headers=None, timeout=None, **kwargs):
    class _Resp:
        ok = False

        def json(self):
            return {"error": "Unexpected endpoint or method."}

    return _Resp()


def test_rewrites_base_url_when_v1_models_answers(monkeypatch):
    monkeypatch.setattr("requests.get", _models_ok)
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert agent.base_url == "http://127.0.0.1:39080/v1"
    assert agent._client_kwargs["base_url"] == "http://127.0.0.1:39080/v1"
    assert agent.client.base_url == "http://127.0.0.1:39080/v1"
    assert agent.vprints  # the rewrite is surfaced to the user


def test_healed_base_url_not_rewritten_again(monkeypatch):
    monkeypatch.setattr("requests.get", _models_ok)
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    # The healed URL is versioned — a 404 on it never triggers another rewrite.
    assert _maybe_apply_v1_suffix_fallback(agent, 404) is False


def test_verified_base_reheals_after_revert_without_reprobe(monkeypatch):
    """Credential rotation / fallback activation restoring the un-suffixed
    base_url must not brick the session: the cached verdict re-heals it
    without a second probe."""
    calls = []

    def _counting_ok(url, **kwargs):
        calls.append(url)
        return _models_ok(url, **kwargs)

    monkeypatch.setattr("requests.get", _counting_ok)
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    agent.base_url = "http://127.0.0.1:39080"  # e.g. _swap_credential revert

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert agent.base_url == "http://127.0.0.1:39080/v1"
    assert agent._client_kwargs["base_url"] == "http://127.0.0.1:39080/v1"
    assert len(calls) == 1


def test_leaves_base_url_alone_when_v1_probe_fails(monkeypatch):
    """A model-level 404 on a correct base_url must not be 'healed'."""
    monkeypatch.setattr("requests.get", _models_404)
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is False
    assert agent.base_url == "http://127.0.0.1:39080"
    assert agent._client_kwargs["base_url"] == "http://127.0.0.1:39080"


def test_failed_probe_not_retried_within_ttl(monkeypatch):
    calls = []

    def _counting_404(url, **kwargs):
        calls.append(url)
        return _models_404(url, **kwargs)

    monkeypatch.setattr("requests.get", _counting_404)
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is False
    assert _maybe_apply_v1_suffix_fallback(agent, 404) is False
    # The /models probe ran exactly once — immediate retries don't re-probe.
    assert len(calls) == 1


def test_failed_probe_retried_after_ttl(monkeypatch):
    """A transient probe failure (e.g. the server restarting) must not
    disable the heal for the rest of a long-lived session."""
    calls = []
    responses = [_models_404, _models_ok]

    def _flaky(url, **kwargs):
        calls.append(url)
        return responses[min(len(calls), len(responses)) - 1](url, **kwargs)

    clock = [1000.0]
    monkeypatch.setattr("requests.get", _flaky)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is False
    clock[0] += _V1_PROBE_RETRY_SECONDS + 1

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert agent.base_url == "http://127.0.0.1:39080/v1"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "kwargs, status",
    [
        ({}, 400),  # only 404 triggers
        ({}, None),  # no status (transport error)
        ({"base_url": "http://127.0.0.1:39080/v1"}, 404),  # already has /v1
        ({"base_url": "http://127.0.0.1:39080/v1/"}, 404),  # trailing slash
        ({"base_url": "http://127.0.0.1:39080/v1beta"}, 404),  # versioned (Gemini-style)
        ({"base_url": "http://127.0.0.1:39080/api/v2"}, 404),  # versioned subpath
        ({"api_mode": "codex_responses"}, 404),  # not chat completions
        ({"api_mode": "anthropic_messages"}, 404),
        ({"base_url": "acp://copilot"}, 404),  # not an HTTP endpoint
        ({"base_url": ""}, 404),
        ({"azure": True}, 404),  # Azure paths are versioned differently
    ],
)
def test_guards_reject_non_matching_shapes(monkeypatch, kwargs, status):
    monkeypatch.setattr("requests.get", _models_ok)
    agent = _StubAgent(**kwargs)
    original = agent.base_url

    assert _maybe_apply_v1_suffix_fallback(agent, status) is False
    assert agent.base_url == original


def test_no_bearer_header_for_placeholder_key(monkeypatch):
    """The 'no-key-required' placeholder must not leak as a bearer token."""
    seen = {}

    def _capture(url, headers=None, timeout=None, **kwargs):
        seen["headers"] = headers
        return _models_ok(url, headers=headers, timeout=timeout, **kwargs)

    monkeypatch.setattr("requests.get", _capture)
    agent = _StubAgent(api_key="no-key-required")

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert not seen["headers"]


def test_bearer_header_sent_for_real_key(monkeypatch):
    seen = {}

    def _capture(url, headers=None, timeout=None, **kwargs):
        seen["headers"] = headers
        return _models_ok(url, headers=headers, timeout=timeout, **kwargs)

    monkeypatch.setattr("requests.get", _capture)
    agent = _StubAgent(api_key="sk-secret")

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert seen["headers"] == {"Authorization": "Bearer sk-secret"}


@pytest.mark.parametrize(
    "tls, expected_verify",
    [
        ({"ssl_ca_cert": "/etc/hermes/custom-ca.pem"}, "/etc/hermes/custom-ca.pem"),
        ({"ssl_verify": False}, False),
        ({"ssl_verify": False, "ssl_ca_cert": "/etc/ca.pem"}, False),
    ],
)
def test_probe_applies_custom_provider_tls(monkeypatch, tls, expected_verify):
    """The heal probe must honor the same per-custom-provider ssl_ca_cert /
    ssl_verify the chat client gets — an endpoint behind a private CA would
    otherwise 404 and then never self-heal because the probe fails TLS."""
    seen = {}

    def _capture(url, headers=None, timeout=None, verify=None, **kwargs):
        seen["verify"] = verify
        return _models_ok(url, headers=headers, timeout=timeout, **kwargs)

    monkeypatch.setattr("requests.get", _capture)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda cfg: [])
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_tls_settings",
        lambda base, providers=None, config=None: dict(tls),
    )
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert seen["verify"] == expected_verify


def test_probe_verify_defaults_when_no_tls_config(monkeypatch):
    """No matching custom-provider entry — fall back to the shared
    requests-verify resolution (env CA bundle or certifi default)."""
    seen = {}

    def _capture(url, headers=None, timeout=None, verify=None, **kwargs):
        seen["verify"] = verify
        return _models_ok(url, headers=headers, timeout=timeout, **kwargs)

    monkeypatch.setattr("requests.get", _capture)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda cfg: [])
    monkeypatch.setattr(
        "agent.model_metadata._resolve_requests_verify", lambda: "/env/bundle.pem"
    )
    agent = _StubAgent()

    assert _maybe_apply_v1_suffix_fallback(agent, 404) is True
    assert seen["verify"] == "/env/bundle.pem"
