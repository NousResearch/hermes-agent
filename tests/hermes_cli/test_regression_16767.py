import sys

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import DirectAlias
from hermes_cli.runtime_provider import _resolve_named_custom_runtime

def test_ensure_direct_aliases_publishes_atomically(monkeypatch):
    """_ensure_direct_aliases publishes a NEW dict rather than mutating in place.

    Originally this asserted the opposite (`id()` stable — "never rebind"), to
    stop a `from hermes_cli.model_switch import DIRECT_ALIASES` consumer being
    stranded on a stale empty dict.  Review of #67007 showed in-place mutation
    is not thread-safe: gateway /model runs the switch under
    ``asyncio.to_thread`` while ``resolve_alias()`` iterates the alias table, so
    an in-place prune can raise "dictionary changed size during iteration" in
    the reader.  The refresh now builds a new dict and rebinds atomically; the
    stale-from-import hazard is instead closed by the source contract in
    ``tests/hermes_cli/test_direct_alias_reload.py::
    test_no_from_import_of_direct_aliases`` (no such import exists in-tree — the
    only consumer, ``hermes_cli/oneshot.py``, reads the module attribute).
    """
    saved = ms.DIRECT_ALIASES
    try:
        ms.DIRECT_ALIASES = {"stale": DirectAlias("old-model", "custom", "")}
        previous = ms.DIRECT_ALIASES
        previous_snapshot = dict(previous)

        mock_data = {
            "my-custom-alias": DirectAlias("custom-model:v1", "custom", "https://example.com/v1")
        }
        # _load_direct_aliases returns (merged, ok).
        monkeypatch.setattr(ms, "_load_direct_aliases", lambda: (mock_data, True))

        ms._ensure_direct_aliases()

        assert ms.DIRECT_ALIASES is not previous, (
            "DIRECT_ALIASES was mutated in place — concurrent readers iterating "
            "it can be invalidated (see #67007 review)"
        )
        assert previous == previous_snapshot, (
            "the previously published dict was mutated after publication; a "
            "reader still holding it would see a torn table"
        )
        assert "my-custom-alias" in ms.DIRECT_ALIASES
        assert ms.DIRECT_ALIASES["my-custom-alias"].model == "custom-model:v1"
        assert "stale" not in ms.DIRECT_ALIASES
    finally:
        ms.DIRECT_ALIASES = saved

def test_chat_provider_argparse_acceptance(monkeypatch):
    """chat --provider <user-defined> is accepted by argparse (guards against restrictive choices)."""
    recorded: dict[str, str] = {}

    # Mock cmd_chat to record the provider passed to it
    def mock_cmd_chat(args):
        recorded["provider"] = args.provider

    monkeypatch.setattr("hermes_cli.main.cmd_chat", mock_cmd_chat)
    monkeypatch.setattr(sys, "argv", ["hermes", "chat", "--provider", "my-custom-key"])

    from hermes_cli.main import main
    main()

    assert recorded["provider"] == "my-custom-key"

def test_resolve_named_custom_runtime_honors_explicit_base_url(monkeypatch):
    """_resolve_named_custom_runtime honors (provider='custom', explicit_base_url=...)."""
    # Mock has_usable_secret to recognize our test key
    monkeypatch.setattr("hermes_cli.runtime_provider.has_usable_secret", lambda x: x == "test-api-key")
    
    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_api_key="test-api-key",
        explicit_base_url="http://example.test:1234/v1"
    )
    
    assert result is not None
    assert result["base_url"] == "http://example.test:1234/v1"
    assert result["provider"] == "custom"
    assert result["api_key"] == "test-api-key"
    assert result["source"] == "direct-alias"
