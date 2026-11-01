"""Quarantine keys for named custom providers must track the provider's own endpoint.

``_custom_health_base_url`` resolved ``_resolves_to_custom`` before the named-provider
lookup, so a ``providers:`` entry keyed on a local-server alias (``ollama``, ``vllm`` —
names ``resolve_provider`` maps to ``custom`` but that are NOT shadowed builtins) was
health-keyed to the ambient custom endpoint instead of its configured ``base_url``.
Two distinct lanes then shared one ``_unhealthy_cache_key``: a 402 on the named
provider's endpoint quarantined every ambient-custom lane, and vice versa.
"""
import pytest
from unittest.mock import MagicMock, patch

from agent import auxiliary_client as ac
from agent.auxiliary_client import call_llm
from agent.auxiliary_health import _custom_health_base_url, _unhealthy_cache_key
from hermes_constants import get_hermes_home

AMBIENT = "https://a.example/v1"
OLLAMA_ENTRY = "https://b.example/v1"
FOO_ENTRY = "https://c.example/v1"


@pytest.fixture(autouse=True)
def _providers_config():
    """config.yaml in the per-test HERMES_HOME: ambient custom endpoint a.example, a
    named provider entry on the ``ollama`` alias pointing at b.example, and a plain
    named provider ``foo`` at c.example."""
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {AMBIENT}\n"
        "providers:\n"
        "  ollama:\n"
        f"    base_url: {OLLAMA_ENTRY}\n"
        "  llamacpp:\n"
        f"    base_url: {FOO_ENTRY}\n"
        "  foo:\n"
        f"    base_url: {FOO_ENTRY}\n"
    )
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


@pytest.mark.parametrize("alias,expected", [
    ("ollama", OLLAMA_ENTRY),
    ("custom:ollama", OLLAMA_ENTRY),
    ("llamacpp", FOO_ENTRY),
    ("custom:llamacpp", FOO_ENTRY),
])
def test_alias_named_provider_keys_to_its_own_endpoint(alias, expected):
    # Every local-server alias spelling must key to the named entry's endpoint,
    # never the ambient custom endpoint.
    assert _custom_health_base_url(alias) == expected


def test_alias_named_provider_key_matches_prefixed_spelling():
    # "ollama" and "custom:ollama" name the same config entry; one provider, one key.
    assert _unhealthy_cache_key("ollama") == _unhealthy_cache_key("custom:ollama")


def test_alias_named_provider_does_not_share_ambient_key():
    assert _unhealthy_cache_key("ollama") != _unhealthy_cache_key("vllm")


def test_marking_alias_named_provider_leaves_ambient_lane_healthy():
    ac._mark_provider_unhealthy("ollama")
    assert ac._is_provider_unhealthy("ollama")
    assert not ac._is_provider_unhealthy("vllm")
    assert not ac._is_provider_unhealthy("custom")


def test_concrete_endpoint_failure_quarantines_the_alias_lane():
    # The real mark path records the URL the request actually used (destination.base_url);
    # the check path derives the URL from the provider name. Both must land on one key.
    ac._mark_provider_unhealthy("ollama", base_url=OLLAMA_ENTRY)
    assert ac._is_provider_unhealthy("ollama")
    assert not ac._is_provider_unhealthy("vllm")


def test_ambient_failure_does_not_hide_the_alias_lane():
    ac._mark_provider_unhealthy("vllm", base_url=AMBIENT)
    assert ac._is_provider_unhealthy("vllm")
    assert ac._is_provider_unhealthy("custom")
    assert not ac._is_provider_unhealthy("ollama")


def test_alias_with_no_named_entry_still_keys_to_ambient():
    assert _custom_health_base_url("vllm") == AMBIENT
    assert _custom_health_base_url("custom") == AMBIENT


def test_disabled_alias_entry_does_not_claim_the_key():
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {AMBIENT}\n"
        "providers:\n"
        "  ollama:\n"
        "    enabled: false\n"
        f"    base_url: {OLLAMA_ENTRY}\n"
    )
    assert _custom_health_base_url("ollama") == AMBIENT


def test_plain_named_provider_still_keys_to_its_endpoint():
    assert _custom_health_base_url("custom:foo") == FOO_ENTRY
    assert _custom_health_base_url("foo") == FOO_ENTRY


def test_explicit_base_url_still_wins_over_named_entry():
    assert _custom_health_base_url("ollama", "https://x.example/v1") == "https://x.example/v1"


def test_legacy_custom_providers_entry_on_alias_keys_to_its_endpoint():
    # The legacy ``custom_providers:`` list shape must get the same identity fix.
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {AMBIENT}\n"
        "custom_providers:\n"
        "  - name: vllm\n"
        f"    base_url: {OLLAMA_ENTRY}\n"
    )
    assert _custom_health_base_url("vllm") == OLLAMA_ENTRY


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp


def _client(base_url: str, create):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = create
    return client


def test_call_llm_walk_serves_alias_named_provider_after_ambient_lane_fails():
    """E2E through call_llm: fallback_chain [vllm, ollama]. vllm's client 429s and is
    quarantined under the ambient endpoint's key; the ollama lane (named entry at
    b.example) must still be walked — pre-fix it shares that key and is skipped."""
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {AMBIENT}\n"
        "providers:\n"
        "  ollama:\n"
        f"    base_url: {OLLAMA_ENTRY}\n"
        "auxiliary:\n"
        "  title_generation:\n"
        "    fallback_chain:\n"
        "      - provider: vllm\n"
        "        model: gpt-4o-mini\n"
        "      - provider: ollama\n"
        "        model: gpt-4o-mini\n"
    )
    err = Exception("Error code: 429 - {'error': {'message': 'Rate limit exceeded', "
                    "'type': 'rate_limit_exceeded'}}")
    err.status_code = 429
    primary = _client("https://p.example/v1", MagicMock(side_effect=err))
    lane_vllm = _client(AMBIENT, MagicMock(side_effect=err))
    lane_ollama = _client(OLLAMA_ENTRY, MagicMock(return_value=_ok_response("OK")))

    def _resolve_entry(entry):
        return (lane_vllm if entry.get("provider") == "vllm" else lane_ollama,
                entry.get("model"))

    patches = (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("auto", None, None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "modelA")),
        patch("agent.auxiliary_client._resolve_fallback_entry", side_effect=_resolve_entry),
        patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "OK"
    assert lane_vllm.chat.completions.create.call_count == 1
    assert lane_ollama.chat.completions.create.call_count == 1
