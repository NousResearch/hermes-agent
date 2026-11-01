"""Tests for `hermes portal` CLI — status, open, tools subcommands."""

from __future__ import annotations

from argparse import Namespace

import pytest

from hermes_cli.nous_subscription import NousFeatureState, NousSubscriptionFeatures


# ---------------------------------------------------------------------------
# Helpers — build mock feature states compactly
# ---------------------------------------------------------------------------

def _mk_features(
    *,
    nous_auth_present: bool = True,
    provider_is_nous: bool = True,
    web_managed: bool = True,
    image_gen_managed: bool = True,
    tts_managed: bool = True,
    browser_managed: bool = True,
    modal_managed: bool = False,
) -> NousSubscriptionFeatures:
    """Build a NousSubscriptionFeatures with sensible defaults."""
    def _fs(key: str, label: str, managed: bool) -> NousFeatureState:
        return NousFeatureState(
            key=key,
            label=label,
            included_by_default=True,
            available=True,
            active=True,
            managed_by_nous=managed,
            direct_override=False,
            toolset_enabled=True,
        )

    return NousSubscriptionFeatures(
        subscribed=True,
        nous_auth_present=nous_auth_present,
        provider_is_nous=provider_is_nous,
        features={
            "web": _fs("web", "Web search & extract", web_managed),
            "image_gen": _fs("image_gen", "Image generation", image_gen_managed),
            "tts": _fs("tts", "Text-to-speech", tts_managed),
            "browser": _fs("browser", "Browser automation", browser_managed),
            "modal": _fs("modal", "Cloud terminal", modal_managed),
        },
    )


# ---------------------------------------------------------------------------
# _cmd_status
# ---------------------------------------------------------------------------

class TestPortalStatus:
    """Tests for `hermes portal status` output."""

    @pytest.fixture(autouse=True)
    def _mock_load_config(self, monkeypatch):
        """Isolate config so test doesn't read the real config.yaml."""
        monkeypatch.setattr(
            "hermes_cli.portal_cli.load_config",
            lambda: {},
        )

    def test_status_logged_in_nous_provider(self, monkeypatch, capsys):
        """Status when fully logged in with Nous as the provider."""
        auth = {
            "logged_in": True,
            "portal_base_url": "https://portal.nousresearch.com",
            "inference_base_url": "https://inference.nousresearch.com",
        }
        features = _mk_features()

        # Lazy imports in portal_cli go through the source modules
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: auth,
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )
        monkeypatch.setattr(
            "hermes_cli.portal_cli.load_config",
            lambda: {"model": {"provider": "nous"}},
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "Nous Portal" in out
        assert "logged in" in out
        assert "https://portal.nousresearch.com" in out
        assert "https://inference.nousresearch.com" in out
        assert "using Nous as inference provider" in out

    def test_status_not_logged_in(self, monkeypatch, capsys):
        """Status when not authenticated with Nous."""
        auth = {"logged_in": False}
        features = _mk_features(nous_auth_present=False)

        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: auth,
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "not logged in" in out
        assert "hermes auth add nous --type oauth" in out

    def test_status_logged_in_different_provider(self, monkeypatch, capsys):
        """Status when logged into Nous but using a different provider."""
        auth = {
            "logged_in": True,
            "portal_base_url": "https://portal.nousresearch.com",
        }
        features = _mk_features(provider_is_nous=False)

        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: auth,
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )
        # Make config return openrouter as the active provider
        monkeypatch.setattr(
            "hermes_cli.portal_cli.load_config",
            lambda: {"model": {"provider": "openrouter"}},
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "logged in" in out
        assert "currently openrouter" in out
        assert "switch with `hermes model`" in out

    def test_status_features_unavailable(self, monkeypatch, capsys):
        """Status when subscription features cannot be resolved."""
        auth = {"logged_in": False}

        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: auth,
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: None,
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "could not resolve subscription state" in out

    def test_status_mixed_gateway_routing(self, monkeypatch, capsys):
        """Status when some tools are managed by Nous, some not."""
        auth = {
            "logged_in": True,
            "portal_base_url": "https://portal.nousresearch.com",
        }
        features = NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            features={
                "web": NousFeatureState(
                    key="web", label="Web search & extract",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=True, direct_override=False, toolset_enabled=True,
                ),
                "image_gen": NousFeatureState(
                    key="image_gen", label="Image generation",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=True, current_provider="fal",
                ),
                "tts": NousFeatureState(
                    key="tts", label="Text-to-speech",
                    included_by_default=True, available=True, active=False,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=True,
                ),
                "browser": NousFeatureState(
                    key="browser", label="Browser automation",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=True, direct_override=False, toolset_enabled=True,
                ),
                "modal": NousFeatureState(
                    key="modal", label="Cloud terminal",
                    included_by_default=True, available=False, active=False,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=False,
                ),
            },
        )

        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: auth,
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "via Nous Portal" in out
        assert "fal" in out
        assert "not configured" in out

    def test_status_auth_exception_handled(self, monkeypatch, capsys):
        """Status when get_nous_auth_status raises an exception."""
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: (_ for _ in ()).throw(RuntimeError("auth store down")),
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: _mk_features(nous_auth_present=False),
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "not logged in" in out

    def test_status_features_exception_handled(self, monkeypatch, capsys):
        """Status when get_nous_subscription_features raises an exception."""
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: (_ for _ in ()).throw(RuntimeError("no features")),
        )

        from hermes_cli.portal_cli import _cmd_status

        assert _cmd_status(Namespace()) == 0
        out = capsys.readouterr().out

        assert "could not resolve subscription state" in out


# ---------------------------------------------------------------------------
# _cmd_open
# ---------------------------------------------------------------------------

class TestPortalOpen:
    """Tests for `hermes portal open`."""

    def test_open_success(self, monkeypatch, capsys):
        """Browser opens successfully — return 0."""
        opened = []

        def _fake_open(url):
            opened.append(url)
            return True

        monkeypatch.setattr("hermes_cli.portal_cli.webbrowser.open", _fake_open)

        from hermes_cli.portal_cli import _cmd_open

        assert _cmd_open(Namespace()) == 0
        out = capsys.readouterr().out

        assert "https://portal.nousresearch.com/manage-subscription" in out
        assert len(opened) == 1
        assert "manage-subscription" in opened[0]

    def test_open_browser_fails(self, monkeypatch, capsys):
        """Browser fails to open — return 1 and show manual URL message."""
        monkeypatch.setattr(
            "hermes_cli.portal_cli.webbrowser.open",
            lambda url: False,
        )

        from hermes_cli.portal_cli import _cmd_open

        assert _cmd_open(Namespace()) == 1
        out = capsys.readouterr().out

        assert "Could not launch a browser" in out
        assert "Visit the URL above manually" in out

    def test_open_browser_exception(self, monkeypatch, capsys):
        """Browser open raises an exception — return 1 gracefully."""
        monkeypatch.setattr(
            "hermes_cli.portal_cli.webbrowser.open",
            lambda url: (_ for _ in ()).throw(RuntimeError("no browser")),
        )

        from hermes_cli.portal_cli import _cmd_open

        assert _cmd_open(Namespace()) == 1
        out = capsys.readouterr().out

        # Still prints the URL before attempting
        assert "manage-subscription" in out
        assert "Could not launch a browser" in out


# ---------------------------------------------------------------------------
# _cmd_tools
# ---------------------------------------------------------------------------

class TestPortalTools:
    """Tests for `hermes portal tools`."""

    @pytest.fixture(autouse=True)
    def _mock_load_config(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.portal_cli.load_config",
            lambda: {},
        )

    def test_tools_full_catalog(self, monkeypatch, capsys):
        """Full catalog with all tools managed by Nous."""
        features = _mk_features()
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )

        from hermes_cli.portal_cli import _cmd_tools

        assert _cmd_tools(Namespace()) == 0
        out = capsys.readouterr().out

        assert "Tool Gateway catalog" in out
        assert "Firecrawl" in out
        assert "FAL" in out
        assert "OpenAI TTS" in out
        assert "Browser Use" in out
        assert "Modal" in out
        assert "via Nous Portal" in out
        assert "manage-subscription" in out

    def test_tools_not_logged_in(self, monkeypatch, capsys):
        """Tools catalog when not logged into Nous."""
        features = _mk_features(nous_auth_present=False)
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )

        from hermes_cli.portal_cli import _cmd_tools

        assert _cmd_tools(Namespace()) == 0
        out = capsys.readouterr().out

        assert "Not logged into Nous Portal" in out
        assert "hermes auth add nous --type oauth" in out

    def test_tools_resolution_failure(self, monkeypatch, capsys):
        """Tools when features cannot be resolved at all."""
        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: (_ for _ in ()).throw(RuntimeError("broken")),
        )

        from hermes_cli.portal_cli import _cmd_tools

        assert _cmd_tools(Namespace()) == 1
        err = capsys.readouterr().err

        assert "Could not resolve Tool Gateway state" in err

    def test_tools_custom_provider_shown(self, monkeypatch, capsys):
        """When a tool is active but not via Nous, show the current provider."""
        features = NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            features={
                "web": NousFeatureState(
                    key="web", label="Web search & extract",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=True, direct_override=False, toolset_enabled=True,
                ),
                "image_gen": NousFeatureState(
                    key="image_gen", label="Image generation",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=True, current_provider="fal",
                ),
                "tts": NousFeatureState(
                    key="tts", label="Text-to-speech",
                    included_by_default=True, available=True, active=False,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=True,
                ),
                "browser": NousFeatureState(
                    key="browser", label="Browser automation",
                    included_by_default=True, available=True, active=True,
                    managed_by_nous=True, direct_override=False, toolset_enabled=True,
                ),
                "modal": NousFeatureState(
                    key="modal", label="Cloud terminal",
                    included_by_default=True, available=False, active=False,
                    managed_by_nous=False, direct_override=False,
                    toolset_enabled=False,
                ),
            },
        )

        monkeypatch.setattr(
            "hermes_cli.nous_subscription.get_nous_subscription_features",
            lambda config: features,
        )

        from hermes_cli.portal_cli import _cmd_tools

        assert _cmd_tools(Namespace()) == 0
        out = capsys.readouterr().out

        assert "FAL" in out           # partner name
        assert "fal" in out           # current provider
        assert "not configured" in out  # tts inactive
        assert "via Nous Portal" in out  # web and browser


# ---------------------------------------------------------------------------
# portal_command dispatch
# ---------------------------------------------------------------------------

class TestPortalCommandDispatch:
    """Tests for top-level `hermes portal` subcommand dispatch."""

    def test_no_subcommand_defaults_to_status(self, monkeypatch):
        """`hermes portal` with no subcommand runs status."""
        called = []

        def _fake_status(args):
            called.append("status")
            return 0

        monkeypatch.setattr(
            "hermes_cli.portal_cli._cmd_status",
            _fake_status,
        )

        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command=None)) == 0
        assert called == ["status"]

    def test_empty_subcommand_defaults_to_status(self, monkeypatch):
        """`hermes portal ''` with empty subcommand runs status."""
        called = []

        def _fake_status(args):
            called.append("status")
            return 0

        monkeypatch.setattr(
            "hermes_cli.portal_cli._cmd_status",
            _fake_status,
        )

        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command="")) == 0
        assert called == ["status"]

    def test_explicit_status_subcommand(self, monkeypatch):
        """`hermes portal status` explicitly."""
        called = []

        def _fake_status(args):
            called.append("status")
            return 0

        monkeypatch.setattr(
            "hermes_cli.portal_cli._cmd_status",
            _fake_status,
        )

        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command="status")) == 0
        assert called == ["status"]

    def test_open_subcommand(self, monkeypatch):
        """`hermes portal open` routes to _cmd_open."""
        called = []

        def _fake_open(args):
            called.append("open")
            return 42

        monkeypatch.setattr(
            "hermes_cli.portal_cli._cmd_open",
            _fake_open,
        )

        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command="open")) == 42
        assert called == ["open"]

    def test_tools_subcommand(self, monkeypatch):
        """`hermes portal tools` routes to _cmd_tools."""
        called = []

        def _fake_tools(args):
            called.append("tools")
            return 7

        monkeypatch.setattr(
            "hermes_cli.portal_cli._cmd_tools",
            _fake_tools,
        )

        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command="tools")) == 7
        assert called == ["tools"]

    def test_unknown_subcommand(self, capsys):
        """Unknown subcommand returns 1 and prints to stderr."""
        from hermes_cli.portal_cli import portal_command

        assert portal_command(Namespace(portal_command="bogus")) == 1
        err = capsys.readouterr().err

        assert "Unknown portal subcommand: bogus" in err
        assert "hermes portal -h" in err