"""``auxiliary.compression.local_override`` scopes compaction to listed local endpoints.

One config has to serve two cases on a machine that switches between a local/LAN chat
model and external providers: while the main model is a listed local endpoint, compaction
runs on the route named by ``local_override``; every other model keeps its own (or the
explicitly configured) route.
"""

from unittest.mock import patch

import pytest

from agent.auxiliary_client import _resolve_task_provider_model

LOCAL_V1 = "http://myhost:8000/v1"
LOCAL_BARE = "http://myhost:8000"
LAN_V1 = "http://192.0.2.50:8000/v1"
OVERRIDE_PROVIDER = "openrouter"
OVERRIDE_MODEL = "@preset/summarizer"


def _config(base=None):
    """Task config carrying the override block, optionally over a base route."""
    cfg = dict(base or {})
    cfg["local_override"] = {
        "provider": OVERRIDE_PROVIDER,
        "model": OVERRIDE_MODEL,
        "base_urls": [LOCAL_V1, LOCAL_BARE, LAN_V1],
    }
    return cfg


def _resolve(config, main_runtime, task="compression"):
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config):
        return _resolve_task_provider_model(task, main_runtime=main_runtime)


def _runtime(provider="custom", base_url=LOCAL_V1, model="local-model"):
    return {"provider": provider, "model": model, "base_url": base_url}


# --- the override fires ---------------------------------------------------


def test_listed_local_endpoint_uses_the_override_route():
    provider, model, _, _, _ = _resolve(_config(), _runtime())
    assert (provider, model) == (OVERRIDE_PROVIDER, OVERRIDE_MODEL)


def test_override_wins_over_an_explicit_base_route():
    """A per-call route substitution, not a fallback: it replaces the configured route."""
    base = {"provider": "openrouter", "model": "@preset/other"}
    provider, model, _, _, _ = _resolve(_config(base), _runtime())
    assert (provider, model) == (OVERRIDE_PROVIDER, OVERRIDE_MODEL)


@pytest.mark.parametrize("base_url", [LOCAL_V1 + "/", LOCAL_BARE, LAN_V1])
def test_match_ignores_trailing_slash_and_accepts_lan_ips(base_url):
    _, model, _, _, _ = _resolve(_config(), _runtime(base_url=base_url))
    assert model == OVERRIDE_MODEL


@pytest.mark.parametrize(
    "provider", ["custom", "custom:Local:8000", "ollama", "lmstudio", "vllm"]
)
def test_local_provider_families_match(provider):
    _, model, _, _, _ = _resolve(_config(), _runtime(provider=provider))
    assert model == OVERRIDE_MODEL


def test_override_carries_its_own_api_key():
    cfg = _config()
    cfg["local_override"]["api_key"] = "sk-not-a-secret"
    provider, model, base_url, api_key, _ = _resolve(cfg, _runtime())
    assert (provider, model, api_key) == (OVERRIDE_PROVIDER, OVERRIDE_MODEL, "sk-not-a-secret")
    assert base_url is None


def test_override_base_url_is_applied():
    """A base_url in the override lands on the call. The provider identity follows the
    resolver's existing "bare base_url means custom" rule, so only the endpoint and the
    model are asserted here."""
    cfg = _config()
    cfg["local_override"]["base_url"] = "https://api.example.invalid/v1"
    _, model, base_url, _, _ = _resolve(cfg, _runtime())
    assert model == OVERRIDE_MODEL
    assert base_url == "https://api.example.invalid/v1"


# --- the override does NOT fire -------------------------------------------


def test_unlisted_local_endpoint_keeps_the_base_route():
    base = {"provider": "openrouter", "model": "@preset/other"}
    provider, model, _, _, _ = _resolve(
        _config(base), _runtime(base_url="http://myhost:9999/v1")
    )
    assert (provider, model) == ("openrouter", "@preset/other")


@pytest.mark.parametrize("provider", ["openrouter", "deepseek", "anthropic", "openai"])
def test_remote_main_provider_ignores_the_override(provider):
    """Provider gate first: a remote main model never matches, whatever its base_url."""
    base = {"provider": "openrouter", "model": "@preset/other"}
    _, model, _, _, _ = _resolve(_config(base), _runtime(provider=provider))
    assert model == "@preset/other"


def test_absent_override_leaves_resolution_untouched():
    base = {"provider": "openrouter", "model": "@preset/other"}
    provider, model, _, _, _ = _resolve(base, _runtime())
    assert (provider, model) == ("openrouter", "@preset/other")


def test_empty_base_route_falls_through_to_auto_for_a_remote_main():
    """No base route + non-matching main => "auto": the main model summarizes."""
    provider, model, _, _, _ = _resolve(
        _config(), _runtime(provider="deepseek", base_url="")
    )
    assert (provider, model) == ("auto", None)


def test_other_tasks_ignore_the_override_block():
    provider, model, _, _, _ = _resolve(_config(), _runtime(), task="skills_hub")
    assert (provider, model) == ("auto", None)


def test_route_follows_each_calls_runtime():
    """Two calls in one process with different main runtimes route differently
    (mid-session /model switch, concurrent gateway sessions)."""
    config = _config()
    local = _resolve(config, _runtime())
    remote = _resolve(config, _runtime(provider="deepseek", base_url=""))
    assert local[1] == OVERRIDE_MODEL
    assert remote[:2] == ("auto", None)


# --- the route's own policy travels with it ---------------------------------
#
# ``reasoning_effort``/``extra_body`` describe how a specific model answers, exactly like
# ``base_url`` describes where the request goes. A substituted route that kept reading the base
# block's policy would ship the base route's reasoning controls to the override's model — and
# could not certify a fast lane the override itself declares. Same rule upstream applies once an
# operator pins a route (``auxiliary_task_config._layer_over_inherited`` drops the inherited
# route keys rather than merging them).

POLICY_LOCAL = {"provider": "custom", "base_url": LOCAL_V1}
POLICY_REMOTE = {"provider": "deepseek", "base_url": ""}


def _policy_config(base_effort, override_effort):
    cfg = {"provider": OVERRIDE_PROVIDER, "model": "@preset/base-summarizer"}
    if base_effort is not None:
        cfg["reasoning_effort"] = base_effort
    cfg["local_override"] = {
        "provider": OVERRIDE_PROVIDER, "model": OVERRIDE_MODEL, "base_urls": [LOCAL_V1],
    }
    if override_effort is not None:
        cfg["local_override"]["reasoning_effort"] = override_effort
    return cfg


def _extra_body(config, main_runtime):
    from agent.auxiliary_client import _get_task_extra_body
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config):
        return _get_task_extra_body("compression", main_runtime)


def test_override_reasoning_effort_reaches_the_override_route():
    """The override's own ``reasoning_effort`` is not silently dead config."""
    assert _extra_body(_policy_config(None, "none"), POLICY_LOCAL)["reasoning"]["enabled"] is False


def test_base_reasoning_effort_does_not_leak_onto_the_override_route():
    """The gate replaces the route's policy, not just its endpoint."""
    reasoning = _extra_body(_policy_config("high", "none"), POLICY_LOCAL)["reasoning"]
    assert reasoning["enabled"] is False
    assert reasoning.get("effort") != "high"


def test_unlisted_main_keeps_the_base_policy():
    """The same config, non-matching main: the base route keeps its own reasoning depth."""
    body = _extra_body(_policy_config("high", "none"), POLICY_REMOTE)
    assert body["reasoning"] == {"enabled": True, "effort": "high"}


def test_override_extra_body_replaces_the_base_extra_body():
    cfg = _policy_config(None, None)
    cfg["extra_body"] = {"thinking": {"type": "enabled"}}
    cfg["local_override"]["extra_body"] = {"thinking": {"type": "disabled"}}
    body = _extra_body(cfg, POLICY_LOCAL)
    assert body["thinking"] == {"type": "disabled"}
    assert _extra_body(cfg, POLICY_REMOTE)["thinking"] == {"type": "enabled"}


def test_fast_lane_certifies_a_non_reasoning_override_route():
    """The whole point of a cheap long-context route: its reasoning-off is honoured, so the
    compression call rides the fast lane instead of paying for reasoning it declared off."""
    from agent.auxiliary_client import resolve_compression_fast_lane, scoped_runtime_main
    cfg = _policy_config(None, "none")
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=cfg), \
            scoped_runtime_main(POLICY_LOCAL):
        lane = resolve_compression_fast_lane(OVERRIDE_PROVIDER, OVERRIDE_MODEL)
    assert lane.certified_non_reasoning is True
    assert lane.reasoning_config == {"enabled": False, "effort": "none"}


def test_fast_lane_still_certifies_only_the_route_that_declares_it():
    """A base route that never disabled reasoning gets no fast lane from the override's key."""
    from agent.auxiliary_client import resolve_compression_fast_lane, scoped_runtime_main
    cfg = _policy_config(None, "none")
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=cfg), \
            scoped_runtime_main(POLICY_LOCAL):
        lane = resolve_compression_fast_lane("openrouter", "@preset/base-summarizer")
    assert lane.certified_non_reasoning is False


def test_authoritative_task_keeps_its_pinned_route_unscoped():
    """A route-authoritative task arrives with a complete caller-supplied route (a fallback
    retry after every aux attempt failed). Re-adding the configured block's endpoint there is
    the leak the wrapper exists to prevent (#130895), so the override stays out of it."""
    from agent.auxiliary_task_config import authoritative_auxiliary_task
    task = authoritative_auxiliary_task("compression")
    cfg = {"provider": OVERRIDE_PROVIDER, "model": "@preset/base-summarizer"}
    cfg["local_override"] = {"provider": "openrouter", "model": OVERRIDE_MODEL, "base_urls": [LOCAL_V1]}
    with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=cfg):
        _, model, _, _, _ = _resolve_task_provider_model(task, main_runtime=POLICY_LOCAL)
    assert model == "@preset/base-summarizer"
