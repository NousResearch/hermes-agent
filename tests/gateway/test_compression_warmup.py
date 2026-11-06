"""The post-compression prefix-cache warmup must never fire on a cloud route.

The warmup replays the rebuilt context with ``max_tokens=1`` so a local model server
stores the prefix the next real turn will hit. Two properties are load-bearing:

* the route must be LOCAL — a cloud provider bills per token, so a warmup there is
  pure cost;
* the whole feature is opt-in (``gateway.prefix_cache_warmup.enabled``), so a user who
  never asked for it pays nothing.
"""

import pytest


def _runtime(**overrides):
    base = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "http://127.0.0.1:8085/v1",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://127.0.0.1:8085/v1", True),
        ("http://localhost:1234/v1", True),
        ("http://192.168.1.50:8000/v1", True),
        ("http://ollama-box:11434/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://api.minimax.com/v1", False),
        ("https://generativelanguage.googleapis.com/v1beta", False),
    ],
)
def test_local_detection_follows_the_shared_helper(base_url, expected):
    from gateway.compression_warmup import _is_local_runtime

    assert _is_local_runtime(_runtime(base_url=base_url)) is expected


def test_non_chat_completions_route_is_never_local():
    from gateway.compression_warmup import _is_local_runtime

    assert _is_local_runtime(_runtime(api_mode="anthropic_messages")) is False


def test_missing_base_url_is_never_local():
    from gateway.compression_warmup import _is_local_runtime

    assert _is_local_runtime(_runtime(base_url="")) is False


def test_feature_is_disabled_by_default(monkeypatch):
    """The config default is off — a user must opt in."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["prefix_cache_warmup"]["enabled"] is False


def test_disabled_warmup_returns_inside_the_profile_scope(monkeypatch):
    """A disabled warmup must not even enter its idle-wait loop.

    The flag is read inside the routed profile's scope, so the check has to happen
    there — a multiplexed gateway serves several profiles, each with its own
    config.yaml, and the wrong file would enable the feature for everyone.
    """
    import asyncio

    from gateway import compression_warmup

    monkeypatch.setattr(compression_warmup, "_warmup_enabled", lambda: False)

    entered_scope = []

    class _Scope:
        def __enter__(self):
            entered_scope.append(True)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(compression_warmup, "_profile_scope", lambda *a, **k: _Scope())

    class _Gateway:
        def _resolve_profile_home_for_source(self, source):
            return "/tmp/profile-home"

    async def _run():
        await compression_warmup._warmup_task(
            _Gateway(), session_key="key-1", session_entry=object(), source=object(),
            reason="manual-compress", generation=1,
        )

    asyncio.run(_run())
    assert entered_scope == [True], "the scope must be entered before the flag is read"
