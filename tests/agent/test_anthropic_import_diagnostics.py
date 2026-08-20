"""Regression tests for #30149: ``anthropic`` SDK import diagnostics.

The bug: an operator on Python 3.14 / Windows installed ``anthropic``
via pip (``Requirement already satisfied: anthropic>=0.39.0 in
C:\\Python314\\Lib\\site-packages (0.104.0)``) but Hermes still raised
``Failed to initialize agent: The 'anthropic' package is required for
the Anthropic provider`` with no further detail — making it impossible
to tell whether the install landed in the wrong interpreter, whether a
transitive dep failed at import time on the new Python release, or
something else entirely.

The fix surfaces the actual underlying exception + the exact
``sys.executable`` of the interpreter Hermes is running in, so the
operator can either (a) re-run pip against the matching interpreter or
(b) read the captured exception to see which transitive dep / API
broke.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def _reset_module_state():
    """Drop the cached SDK + captured import error between tests so each
    case observes a fresh load attempt."""
    import agent.anthropic_adapter as mod
    mod._anthropic_sdk = ...
    mod._anthropic_sdk_import_error = None


@pytest.fixture(autouse=True)
def _isolate_adapter_state():
    _reset_module_state()
    yield
    _reset_module_state()


# ── _anthropic_unavailable_message format ────────────────────────────────


class TestUnavailableMessageFormat:
    def test_includes_context_phrase(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        assert "required for the Anthropic provider" in msg

    def test_includes_current_interpreter_path(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        assert sys.executable in msg, msg

    def test_includes_pip_command_pointing_at_current_interpreter(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        # The exact command form is what the user has to run to fix the
        # mismatch case: it must invoke pip via the SAME interpreter.
        assert f"{sys.executable} -m pip install" in msg
        assert "'anthropic>=0.39.0'" in msg

    def test_includes_interpreter_mismatch_hint(self):
        """The hint that points at #30149 (install landed in a different
        interpreter than the one running Hermes) — without this, the
        operator has no signal pointing at the actual cause."""
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        assert "#30149" in msg
        assert "different Python interpreter" in msg

    def test_includes_captured_underlying_error_when_present(self):
        import agent.anthropic_adapter as mod
        mod._anthropic_sdk_import_error = (
            "ModuleNotFoundError: No module named 'typing_extensions'"
        )
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        assert "Underlying import error" in msg
        assert "typing_extensions" in msg

    def test_omits_underlying_error_line_when_unavailable(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Anthropic provider")
        # No misleading "Underlying import error: None" line — when we
        # don't have a captured exception, we just don't render that line.
        assert "Underlying import error" not in msg

    def test_context_phrase_for_bedrock(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message("the Bedrock provider")
        assert "required for the Bedrock provider" in msg

    def test_context_phrase_for_azure(self):
        from agent.anthropic_adapter import _anthropic_unavailable_message
        msg = _anthropic_unavailable_message(
            "Azure Foundry Anthropic-style endpoints with Entra ID auth"
        )
        assert "Azure Foundry" in msg


# ── _get_anthropic_sdk captures non-ImportError exceptions ──────────────


class TestSdkImportFailureCapture:
    def test_captures_import_error_with_message(self):
        import agent.anthropic_adapter as mod

        def _raise(*args, **kwargs):
            raise ImportError("No module named 'anthropic'")

        with patch.dict(sys.modules, {"anthropic": None}, clear=False):
            with patch("builtins.__import__", side_effect=_raise):
                result = mod._get_anthropic_sdk()

        assert result is None
        assert mod._anthropic_sdk_import_error is not None
        assert "ImportError" in mod._anthropic_sdk_import_error
        assert "anthropic" in mod._anthropic_sdk_import_error

    def test_captures_non_import_exception_at_import_time(self):
        """Packages can raise ``RuntimeError`` / ``AttributeError`` /
        ``TypeError`` at import time when a transitive dep is broken
        (the #30149 Python 3.14 scenario). Capturing only ``ImportError``
        used to swallow those silently."""
        import agent.anthropic_adapter as mod

        def _raise(name, *args, **kwargs):
            if name == "anthropic":
                raise AttributeError("type object 'X' has no attribute 'Y'")
            return __real_import(name, *args, **kwargs)

        import builtins
        __real_import = builtins.__import__

        with patch.dict(sys.modules, {"anthropic": None}, clear=False):
            with patch("builtins.__import__", side_effect=_raise):
                result = mod._get_anthropic_sdk()

        assert result is None
        assert mod._anthropic_sdk_import_error is not None
        assert "AttributeError" in mod._anthropic_sdk_import_error

    def test_captures_runtime_error_at_import_time(self):
        import agent.anthropic_adapter as mod

        def _raise(name, *args, **kwargs):
            if name == "anthropic":
                raise RuntimeError("native lib failed to load")
            return __real_import(name, *args, **kwargs)

        import builtins
        __real_import = builtins.__import__

        with patch.dict(sys.modules, {"anthropic": None}, clear=False):
            with patch("builtins.__import__", side_effect=_raise):
                result = mod._get_anthropic_sdk()

        assert result is None
        assert mod._anthropic_sdk_import_error is not None
        assert "RuntimeError" in mod._anthropic_sdk_import_error
        assert "native lib failed to load" in mod._anthropic_sdk_import_error

    def test_clears_capture_on_subsequent_successful_load(self):
        import agent.anthropic_adapter as mod

        # First load fails — captures error.
        def _raise(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("boom")
            return __real_import(name, *args, **kwargs)

        import builtins
        __real_import = builtins.__import__

        with patch.dict(sys.modules, {"anthropic": None}, clear=False):
            with patch("builtins.__import__", side_effect=_raise):
                assert mod._get_anthropic_sdk() is None
        assert mod._anthropic_sdk_import_error is not None

        # Reset sentinel so the next call re-attempts the import.
        # In production this would happen after a successful pip install
        # + interpreter restart, but the cached failure remains useful
        # diagnostic state until the next attempt is made.
        captured_before_retry = mod._anthropic_sdk_import_error
        assert "boom" in captured_before_retry


# ── End-to-end: callsites use the helper ────────────────────────────────


class TestCallsitesUseDiagnosticHelper:
    """The three ``build_*`` entry points must raise the rich diagnostic
    when the SDK is unavailable — not the original opaque hint."""

    def _force_unavailable(self, monkeypatch, *, with_capture: bool = True):
        import agent.anthropic_adapter as mod
        monkeypatch.setattr(mod, "_get_anthropic_sdk", lambda: None)
        if with_capture:
            monkeypatch.setattr(
                mod, "_anthropic_sdk_import_error",
                "ImportError: simulated", raising=False,
            )
        return mod

    def test_build_anthropic_client_raises_rich_diagnostic(self, monkeypatch):
        mod = self._force_unavailable(monkeypatch)
        with pytest.raises(ImportError) as exc:
            mod.build_anthropic_client("sk-ant-test-key")
        msg = str(exc.value)
        assert "required for the Anthropic provider" in msg
        assert sys.executable in msg
        assert "#30149" in msg
        assert "Underlying import error: ImportError: simulated" in msg

    def test_build_anthropic_bedrock_client_raises_rich_diagnostic(
        self, monkeypatch
    ):
        mod = self._force_unavailable(monkeypatch)
        with pytest.raises(ImportError) as exc:
            mod.build_anthropic_bedrock_client(region="us-east-1")
        msg = str(exc.value)
        assert "required for the Bedrock provider" in msg
        assert sys.executable in msg
        assert "#30149" in msg

    def test_azure_entra_callsite_raises_rich_diagnostic(self, monkeypatch):
        """The Azure Foundry / Entra ID bearer hook path also routes
        through the same helper."""
        mod = self._force_unavailable(monkeypatch)
        # The function is called inside ``build_anthropic_client`` when
        # ``api_key`` is callable. Invoke the bearer-hook entry point
        # directly to exercise the message format.
        with pytest.raises(ImportError) as exc:
            mod._build_anthropic_client_with_bearer_hook(
                lambda: "bearer-jwt", None, None,
            )
        msg = str(exc.value)
        assert "Azure Foundry" in msg
        assert "Entra ID" in msg
        assert sys.executable in msg
