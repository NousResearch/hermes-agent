"""Tests for user-configured ``model.default_headers`` in the auxiliary client.

Companion to ``tests/run_agent/test_provider_attribution_headers.py`` (which
covers the main agent client). The main agent turn and the auxiliary client
(title generation, context compression, vision routing) build separate OpenAI
clients, so a ``custom`` endpoint behind a gateway/WAF that rejects the OpenAI
SDK's identifying headers needs the ``model.default_headers`` override applied
on BOTH paths — otherwise the main turn succeeds but auxiliary calls to the
same endpoint still fail with an opaque 4xx/502. (#40033)
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so load_config() reads our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


class TestApplyUserDefaultHeadersHelper:
    """Direct unit tests for the merge helper."""

    def test_user_headers_merged_and_win(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({"User-Agent": "OpenAI/Python 2.24.0"})
        assert merged["User-Agent"] == "curl/8.7.1"  # user wins
        assert merged["X-Extra"] == "1"

    def test_no_config_is_noop_returns_original(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _apply_user_default_headers
        original = {"User-Agent": "OpenAI/Python"}
        merged = _apply_user_default_headers(original)
        assert merged == original

    def test_none_headers_with_config_creates_dict(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1"}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers(None)
        assert merged == {"User-Agent": "curl/8.7.1"}

    def test_none_headers_no_config_returns_none(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "m"}})
        from agent.auxiliary_client import _apply_user_default_headers
        assert _apply_user_default_headers(None) is None

    def test_none_values_skipped(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Drop": None}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({})
        assert merged == {"User-Agent": "curl/8.7.1"}
        assert "X-Drop" not in merged


class TestAuxClientHonorsUserDefaultHeaders:
    """Integration: resolve_provider_client must pass overridden headers to OpenAI."""

    def test_custom_provider_overrides_sdk_user_agent(self, tmp_path):
        """The #40033 reproduction on the auxiliary path."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
                "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"},
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent") == "curl/8.7.1"
        assert headers.get("X-Extra") == "1"

    def test_custom_provider_no_override_sends_no_user_agent(self, tmp_path):
        """Without config, the aux client injects nothing — SDK defaults apply."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "User-Agent" not in headers

    def test_named_custom_provider_honors_override(self, tmp_path):
        """A `custom_providers:` entry's aux calls also honor model.default_headers.

        This is a distinct construction path (_extra2) from the config-level
        `model.provider: custom` path — both must apply the global override.
        """
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "curl/8.7.1"},
            },
            "custom_providers": [
                {"name": "my-gw", "base_url": "http://my-gw.local/v1", "api_key": "k"},
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"


class TestAnthropicAdapterHonorsUserDefaultHeaders:
    """SDK-mocked factory tests for ``model.default_headers`` on Anthropic-mode
    providers (api_mode: anthropic_messages).

    The static-key and bearer-hook factories both accept an optional
    ``user_default_headers`` dict that is merged on top of provider/OAuth
    defaults before SDK instantiation. Without that merge, users configuring
    ``model.default_headers`` for Kimi/DeepSeek/etc. see their headers
    ignored on Anthropic requests. (#9589)

    These tests follow the established factory-test pattern from
    ``tests/agent/test_anthropic_adapter.py`` (``TestBuildAnthropicClient``):
    patch ``agent.anthropic_adapter._anthropic_sdk``, call the factory, and
    assert on ``mock_sdk.Anthropic.call_args[1]`` — no simulation, no
    in-memory merge copy-paste.
    """

    def test_user_headers_merged_onto_kimi_defaults(self, tmp_path):
        """User-Agent override wins over Kimi's hardcoded claude-code/0.1.0."""
        from agent.anthropic_adapter import build_anthropic_client

        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant...kimi",
                base_url="https://api.kimi.com/coding",
                user_default_headers={"User-Agent": "my-gateway/1.0"},
            )

        kwargs = mock_sdk.Anthropic.call_args[1]
        # Provider/OAuth defaults preserved
        assert "anthropic-beta" in kwargs["default_headers"]
        # User override wins
        assert kwargs["default_headers"]["User-Agent"] == "my-gateway/1.0"

    def test_none_user_headers_preserves_provider_defaults(self):
        """When user_default_headers=None, provider defaults are untouched."""
        from agent.anthropic_adapter import build_anthropic_client

        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant...kimi", base_url="https://api.kimi.com/coding")

        kwargs = mock_sdk.Anthropic.call_args[1]
        # Provider-default User-Agent (claude-code/0.1.0 for kimi) preserved
        assert kwargs["default_headers"]["User-Agent"] == "claude-code/0.1.0"
        assert "anthropic-beta" in kwargs["default_headers"]

    def test_empty_user_headers_preserves_provider_defaults(self):
        """An empty dict is treated as 'no overrides' — same as None."""
        from agent.anthropic_adapter import build_anthropic_client

        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant...kimi", base_url="https://api.kimi.com/coding",
                user_default_headers={},
            )

        kwargs = mock_sdk.Anthropic.call_args[1]
        assert kwargs["default_headers"]["User-Agent"] == "claude-code/0.1.0"
        assert "anthropic-beta" in kwargs["default_headers"]

    def test_user_header_none_value_does_not_remove_provider_default(self):
        """A None value in user_default_headers means 'don't override', not 'delete'.

        Whether None should instead mean 'remove this key' is an open design
        question (deferred — see PR #53237 follow-up note).
        """
        from agent.anthropic_adapter import build_anthropic_client

        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant...kimi", base_url="https://api.kimi.com/coding",
                user_default_headers={"User-Agent": None, "X-Extra": "yes"},
            )

        kwargs = mock_sdk.Anthropic.call_args[1]
        # Provider default preserved (None means skip, not delete)
        assert kwargs["default_headers"]["User-Agent"] == "claude-code/0.1.0"
        # Non-None user value applied
        assert kwargs["default_headers"]["X-Extra"] == "yes"

    def test_bearer_hook_path_forwards_user_default_headers(self, tmp_path):
        """The Azure Foundry bearer-hook factory must also honour user_default_headers.

        ``_build_anthropic_client_with_bearer_hook`` constructs a custom
        ``httpx.Client`` whose request event hook mints a fresh Entra ID
        JWT per request — the SDK never sees ``Authorization`` directly.
        User-configured headers still need to flow through so an Azure-fronted
        Anthropic endpoint that requires e.g. ``X-Tenant-Id`` works for
        auxiliary calls, not just the main turn.
        """
        from agent.anthropic_adapter import _build_anthropic_client_with_bearer_hook

        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk, \
             patch("agent.azure_identity_adapter.build_bearer_http_client") as mock_bhc:
            mock_bhc.return_value = MagicMock()
            _build_anthropic_client_with_bearer_hook(
                lambda: "fake-token",
                base_url="https://example.services.ai.azure.com/models/anthropic",
                user_default_headers={"X-Tenant-Id": "tenant-42"},
            )

        kwargs = mock_sdk.Anthropic.call_args[1]
        # Provider-default beta header preserved
        assert "anthropic-beta" in kwargs["default_headers"]
        # User-configured header merged in
        assert kwargs["default_headers"]["X-Tenant-Id"] == "tenant-42"
        # Bearer hook plumbing intact (http_client was passed)
        assert "http_client" in kwargs


class TestAuxiliaryConstructionForwardsUserDefaultHeaders:
    """Integration: every native Anthropic auxiliary construction site in
    ``agent/auxiliary_client.py`` must forward ``model.default_headers`` to
    ``build_anthropic_client`` so title/compression/vision calls behave
    consistently with the main agent turn. (#9589)

    Each test patches ``agent.anthropic_adapter.build_anthropic_client`` at
    its SOURCE module — auxiliary_client.py does lazy ``from agent.anthropic_adapter
    import build_anthropic_client`` inside each function, so patching the
    source module's attribute intercepts the lazy import.

    Coverage matrix (per ``git grep build_anthropic_client agent/auxiliary_client.py``):

      Site 1 — ``_maybe_wrap_anthropic``     (URL-detected Anthropic endpoint)
      Site 2 — ``_try_custom_endpoint``      (anonymous custom, api_mode=anthropic_messages)
      Site 3 — ``_try_anthropic``            (native Anthropic provider)
      Site 4 — ``resolve_provider_client``   (named custom, api_mode=anthropic_messages)

    Sites 1, 2, 3 are exercised directly below. Site 4 is covered by
    ``test_resolve_provider_client_named_custom_anthropic_forwards_user_headers``
    which routes through ``resolve_provider_client`` (the public entry point).
    """

    def test_maybe_wrap_anthropic_forwards_user_headers(self, tmp_path):
        """Site 1: ``_maybe_wrap_anthropic`` (URL-detected Anthropic endpoint)."""
        _write_config(tmp_path, {
            "model": {"default": "kimi-k2.5", "default_headers": {"User-Agent": "ua-1"}},
        })
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_bac:
            mock_bac.return_value = MagicMock()
            from agent.auxiliary_client import _maybe_wrap_anthropic
            _maybe_wrap_anthropic(
                MagicMock(), "kimi-k2.5", "sk-ant...key", "https://api.kimi.com/coding",
                api_mode="anthropic_messages",
            )

        assert mock_bac.called
        assert mock_bac.call_args.kwargs.get("user_default_headers") == {"User-Agent": "ua-1"}

    def test_try_anthropic_forwards_user_headers(self, tmp_path):
        """Site 3: ``_try_anthropic`` (native Anthropic provider)."""
        _write_config(tmp_path, {
            "model": {"default": "claude-sonnet", "default_headers": {"X-Org": "acme"}},
        })
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_bac, \
             patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant...native"), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("agent.auxiliary_client._get_aux_model_for_provider", return_value="claude-haiku-4-5-20251001"):
            mock_bac.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()

        assert mock_bac.called
        assert mock_bac.call_args.kwargs.get("user_default_headers") == {"X-Org": "acme"}

    def test_resolve_provider_client_named_custom_anthropic_forwards_user_headers(self, tmp_path):
        """Site 4: ``resolve_provider_client`` named-custom with api_mode=anthropic_messages.

        This is the entry point auxiliary tasks actually use (title gen,
        compression, vision routing). The named-custom path inside it
        resolves to ``_try_custom_endpoint`` for anonymous custom or to
        the same fallback chain for named — both must propagate headers.
        """
        _write_config(tmp_path, {
            "model": {"default": "test", "default_headers": {"X-Custom": "yes"}},
            "custom_providers": [
                {"name": "my-anthropic-gw", "base_url": "http://my-anthropic.local/v1",
                 "api_key": "sk-custom", "api_mode": "anthropic_messages"},
            ],
        })
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_bac:
            mock_bac.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-anthropic-gw", "test")

        assert mock_bac.called
        assert mock_bac.call_args.kwargs.get("user_default_headers") == {"X-Custom": "yes"}

    def test_no_user_headers_configured_forwards_none(self, tmp_path):
        """When no ``model.default_headers`` are configured, the forwarded
        value must be ``None`` (NOT an empty dict) — that's the contract
        ``build_anthropic_client`` uses to skip the merge."""
        _write_config(tmp_path, {"model": {"default": "test"}})
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_bac, \
             patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant...native"), \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("agent.auxiliary_client._get_aux_model_for_provider", return_value="claude-haiku-4-5-20251001"):
            mock_bac.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            _try_anthropic()

        assert mock_bac.called
        assert mock_bac.call_args.kwargs.get("user_default_headers") is None

    def test_all_auxiliary_sites_forward_user_default_headers(self, tmp_path):
        """Negative control: if any of the 4 sites regresses to dropping
        user_default_headers, this grep-level invariant catches it.

        Together with the file-level inspection (see #9589), this guards
        against silent regressions when the auxiliary client is refactored.
        The check operates on the *full* call text (not a truncated snippet),
        so multiline calls where ``user_default_headers=`` appears on a
        continuation line still match.
        """
        _write_config(tmp_path, {
            "model": {"default": "x", "default_headers": {"X-Custom": "yes"}},
        })
        import re
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent.parent / "agent" / "auxiliary_client.py"
        content = src.read_text()
        # Find all build_anthropic_client(...) call sites. DOTALL so newlines
        # inside the call are captured; the regex matches the opening paren
        # through the matching close. We use a simple bracket-balancing walk
        # instead of a greedy regex to avoid eating past the matching `)`.
        call_sites = []
        i = 0
        needle = "build_anthropic_client("
        while True:
            j = content.find(needle, i)
            if j < 0:
                break
            # Walk forward to find the matching `)` accounting for parens and
            # string literals.
            depth = 0
            k = j + len(needle) - 1  # index of the opening `(`
            in_str = None
            while k < len(content):
                ch = content[k]
                if in_str:
                    if ch == "\\":
                        k += 2
                        continue
                    if ch == in_str:
                        in_str = None
                else:
                    if ch in ("'", '"'):
                        in_str = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                k += 1
            call_text = content[j + len(needle):k]  # text inside the parens
            call_sites.append(call_text)
            i = k + 1

        assert call_sites, "expected at least one build_anthropic_client call site"
        # Every call site must reference user_default_headers= somewhere in
        # its full text (handles multiline calls correctly).
        offenders = [
            (i, snippet[:120].replace("\n", " "))
            for i, snippet in enumerate(call_sites)
            if "user_default_headers=" not in snippet
        ]
        assert not offenders, (
            "auxiliary_client.py has build_anthropic_client call sites that "
            "do not forward user_default_headers (would re-introduce #9589): "
            + ", ".join(f"#{i}: {s!r}" for i, s in offenders)
        )
