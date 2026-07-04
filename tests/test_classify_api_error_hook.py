"""Tests for the ``classify_api_error`` plugin hook.

Covers the seam in ``agent.error_classifier.classify_api_error`` (step 0,
consulted before the built-in pipeline), the sanitization contract of
``hermes_cli.plugins.get_plugin_error_classification``, and the bundled
``openrouter-tool-use-404`` reference plugin.

Mirrors the ``transform_tool_result`` hook tests: patch the symbol the
call site actually imports (``hermes_cli.plugins.*``) rather than the
consuming module, because the import happens at call time.
"""

import importlib.util
from pathlib import Path

import hermes_cli.plugins as plugins_mod
from agent.error_classifier import FailoverReason, classify_api_error


class _FakeAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.body = body or {}


_TOOL_USE_404 = 'No endpoints found that support tool use. Try disabling "browser_back".'


def _classify_tool_use_404(**kwargs):
    return classify_api_error(
        _FakeAPIError(_TOOL_USE_404, status_code=404),
        provider="openrouter",
        model="deepseek/deepseek-chat",
        **kwargs,
    )


# ── Baseline: no plugins ────────────────────────────────────────────────


def test_no_hook_falls_through_to_builtin(monkeypatch):
    # Fresh manager so no stale plugin hooks pollute state.
    monkeypatch.setattr(plugins_mod, "_plugin_manager", plugins_mod.PluginManager())

    result = _classify_tool_use_404()
    # Documents the gap the demo plugin closes: a generic 404 with no
    # model-not-found signal classifies as unknown/retryable today.
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True


# ── Plugin classification wins over built-ins ───────────────────────────


def test_plugin_classification_wins(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [
            {"reason": "model_not_found", "retryable": False, "should_fallback": True}
        ],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False
    assert result.should_fallback is True
    # Extracted context is preserved on the ClassifiedError.
    assert result.provider == "openrouter"
    assert result.status_code == 404


def test_plugin_overrides_builtin_classification(monkeypatch):
    # A 429 classifies as rate_limit built-in; a plugin can reclassify it.
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": "overloaded"}],
    )

    result = classify_api_error(
        _FakeAPIError("too many requests", status_code=429),
        provider="zai",
    )
    assert result.reason == FailoverReason.overloaded


def test_enum_reason_accepted(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": FailoverReason.billing}],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.billing


def test_reason_only_dict_uses_dataclass_defaults(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": "server_error"}],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.server_error
    assert result.retryable is True
    assert result.should_compress is False
    assert result.should_rotate_credential is False
    assert result.should_fallback is False


# ── Invalid returns are ignored, first valid wins ───────────────────────


def test_invalid_reason_falls_through_to_builtin(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": "not_a_real_reason"}],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.unknown


def test_non_dict_results_ignored(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: ["model_not_found", 123, ["nope"], None],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.unknown


def test_first_valid_result_wins(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [
            {"reason": "bogus"},
            {"reason": "billing"},
            {"reason": "rate_limit"},
        ],
    )

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.billing


def test_helper_exception_never_breaks_classification(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("plugin infrastructure exploded")

    monkeypatch.setattr(plugins_mod, "get_plugin_error_classification", _boom)

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True


# ── Hook kwargs contract ────────────────────────────────────────────────


def test_hook_receives_parsed_error_context(monkeypatch):
    seen = {}

    def _capture(name, **kw):
        seen.update(kw, hook_name=name)
        return []

    monkeypatch.setattr(plugins_mod, "invoke_hook", _capture)

    _classify_tool_use_404(approx_tokens=1234, num_messages=7)

    assert seen["hook_name"] == "classify_api_error"
    assert seen["provider"] == "openrouter"
    assert seen["model"] == "deepseek/deepseek-chat"
    assert seen["status_code"] == 404
    assert seen["error_type"] == "_FakeAPIError"
    assert "no endpoints found that support tool use" in seen["error_message"]
    assert seen["approx_tokens"] == 1234
    assert seen["num_messages"] == 7
    assert isinstance(seen["error_body"], dict)
    assert isinstance(seen["error"], _FakeAPIError)


def test_message_override_and_error_context_sanitized(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{
            "reason": "model_not_found",
            "message": "  custom guidance  ",
            "error_context": {"upstream_provider": "DeepSeek"},
        }],
    )

    result = _classify_tool_use_404()
    assert result.message == "custom guidance"
    assert result.error_context == {"upstream_provider": "DeepSeek"}


# ── Bundled reference plugin ────────────────────────────────────────────


def _load_demo_plugin():
    plugin_init = (
        Path(__file__).resolve().parent.parent
        / "plugins" / "openrouter-tool-use-404" / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "openrouter_tool_use_404_demo", plugin_init,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_plugin_claims_openrouter_tool_use_404():
    demo = _load_demo_plugin()
    result = demo.classify(
        provider="openrouter",
        status_code=404,
        error_message=_TOOL_USE_404.lower(),
    )
    assert result == {
        "reason": "model_not_found",
        "retryable": False,
        "should_fallback": True,
    }


def test_demo_plugin_self_scopes():
    demo = _load_demo_plugin()
    # Different provider: pass.
    assert demo.classify(
        provider="anthropic", status_code=404,
        error_message=_TOOL_USE_404.lower(),
    ) is None
    # Different status: pass.
    assert demo.classify(
        provider="openrouter", status_code=400,
        error_message=_TOOL_USE_404.lower(),
    ) is None
    # Different message: pass.
    assert demo.classify(
        provider="openrouter", status_code=404,
        error_message="model not found",
    ) is None
    # Missing status but unambiguous phrase: claim.
    assert demo.classify(
        provider="openrouter", status_code=None,
        error_message=_TOOL_USE_404.lower(),
    ) is not None


def test_demo_plugin_end_to_end(monkeypatch):
    """register() + real invoke_hook + classify_api_error, no mocks."""
    demo = _load_demo_plugin()
    manager = plugins_mod.PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)

    class _Ctx:
        def register_hook(self, name, cb):
            manager._hooks.setdefault(name, []).append(cb)

    demo.register(_Ctx())

    result = _classify_tool_use_404()
    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False
    assert result.should_fallback is True

    # And the built-in pipeline is untouched for everything the plugin
    # doesn't claim.
    other = classify_api_error(
        _FakeAPIError("rate limit exceeded", status_code=429),
        provider="openrouter",
    )
    assert other.reason == FailoverReason.rate_limit
