"""``/model`` with no configured ``model.provider`` must display the resolver's actual pick (#136801).

``_ModelSwitchContext.read_config`` leaves ``current_provider`` on its ``openrouter`` placeholder
when config.yaml has no ``model.provider``. The listing used to display that placeholder, so a
funded OAuth login next to a dry API key looked like the current route while the gateway billed
the dry env-key provider (#29285 ordering) — the reporter's "Insufficient Balance despite funded
providers". The listing must name the provider the runtime resolver actually routes through."""

import asyncio


class _NoPickerRunner:
    """Bare host for ``GatewayModelCommandsMixin``: no delivery adapter, so ``/model`` with no
    args falls through to the text listing reply."""

    def _delivery_adapter_for(self, source):
        return None


def _make_ctx(tmp_path):
    from gateway.slash_commands_model import _ModelSwitchContext

    return _ModelSwitchContext(
        session_key="s",
        source=None,
        config_path=tmp_path / "config.yaml",
        persist_global=False,
    )


def _listing(captured):
    def fake_listing(**kwargs):
        captured.update(kwargs)
        return []

    return fake_listing


def _host():
    from gateway.slash_commands_model import GatewayModelCommandsMixin

    class Host(GatewayModelCommandsMixin, _NoPickerRunner):
        pass

    return Host()


async def _listing_reply(ctx):
    return await _host()._model_listing_reply(None, ctx, None)


def test_unconfigured_provider_lists_the_resolved_route(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  default: some-model\n")
    ctx = _make_ctx(tmp_path)
    ctx.read_config()
    assert ctx.route_provider is None

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_provider", lambda requested=None, **kw: "deepseek"
    )
    captured: dict = {}
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", _listing(captured)
    )

    reply = asyncio.run(_listing_reply(ctx))

    assert captured["current_provider"] == "deepseek"
    assert "on DeepSeek" in reply
    assert "OpenRouter" not in reply


def test_configured_provider_bypasses_the_resolver(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-x\n"
    )
    ctx = _make_ctx(tmp_path)
    ctx.read_config()
    assert ctx.route_provider == "anthropic"

    def _fail(requested=None, **kw):
        raise AssertionError("resolver must not run for a configured provider")

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", _fail)
    captured: dict = {}
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", _listing(captured)
    )

    reply = asyncio.run(_listing_reply(ctx))

    assert captured["current_provider"] == "anthropic"
    assert "on Anthropic" in reply


def test_resolver_failure_keeps_the_placeholder(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  default: some-model\n")
    ctx = _make_ctx(tmp_path)
    ctx.read_config()

    def _boom(requested=None, **kw):
        raise RuntimeError("no provider configured")

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", _boom)
    captured: dict = {}
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", _listing(captured)
    )

    reply = asyncio.run(_listing_reply(ctx))

    assert captured["current_provider"] == "openrouter"
    assert "on OpenRouter" in reply
