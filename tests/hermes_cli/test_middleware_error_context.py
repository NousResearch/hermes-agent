"""Execution middleware must hand a downstream error back with its exception chain untouched.

Regression for #149091: ``_run_execution_chain`` unwrapped a downstream error with ``raise
exc.original`` inside ``except _DownstreamExecutionError``, which rebinds the original's
``__context__`` to the private wrapper. ``agent.error_classifier`` walks ``__cause__ or
__context__`` for the status code/body, so a pure pass-through middleware turned an error whose
429 lives in its implicit context into an unclassifiable one.
"""

import logging

import pytest

from agent.error_classifier import _extract_status_code
from hermes_cli import middleware as mw
from hermes_cli.plugins import PluginManager


class StatusError(Exception):
    """Simulates an SDK status error (``openai.APIStatusError`` shape)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _run(kind, terminal):
    if kind == mw.LLM_EXECUTION_MIDDLEWARE:
        return mw.run_llm_execution_middleware({"model": "m"}, terminal)
    return mw.run_tool_execution_middleware("read_file", {"path": "/p"}, terminal)


_KINDS = [mw.LLM_EXECUTION_MIDDLEWARE, mw.TOOL_EXECUTION_MIDDLEWARE]


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    manager = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)
    return manager


def _pass_through(**kwargs):
    return kwargs["next_call"]()


def _register(manager, kind, *callbacks):
    manager._middleware.setdefault(kind, []).extend(callbacks)


def _raise_with_implicit_context(_payload):
    try:
        raise StatusError(429)
    except StatusError:
        raise RuntimeError("stream failed")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("depth", [1, 2], ids=["one-frame", "two-frames"])
def test_pass_through_keeps_implicit_context(manager, kind, depth):
    _register(manager, kind, *[_pass_through] * depth)

    with pytest.raises(RuntimeError, match="stream failed") as info:
        _run(kind, _raise_with_implicit_context)

    assert isinstance(info.value.__context__, StatusError)
    assert info.value.__cause__ is None
    assert _extract_status_code(info.value) == 429


@pytest.mark.parametrize("kind", _KINDS)
def test_pass_through_keeps_explicit_cause(manager, kind):
    status = StatusError(503)

    def terminal(_payload):
        try:
            raise ValueError("decode")
        except ValueError:
            raise RuntimeError("provider down") from status

    _register(manager, kind, _pass_through)

    with pytest.raises(RuntimeError, match="provider down") as info:
        _run(kind, terminal)

    assert info.value.__cause__ is status
    assert info.value.__suppress_context__ is True
    assert isinstance(info.value.__context__, ValueError)
    assert _extract_status_code(info.value) == 503


@pytest.mark.parametrize("kind", _KINDS)
def test_pass_through_keeps_own_status_and_no_context(manager, kind):
    error = StatusError(401)

    def terminal(_payload):
        raise error

    _register(manager, kind, _pass_through)

    with pytest.raises(StatusError) as info:
        _run(kind, terminal)

    assert info.value is error
    # Nothing was being handled when the provider raised, so the chain stays empty.
    assert info.value.__context__ is None
    assert _extract_status_code(info.value) == 401


@pytest.mark.parametrize("kind", _KINDS)
def test_skipped_broken_middleware_does_not_become_the_context(manager, kind, caplog):
    """A middleware that fails before calling next_call is skipped; the downstream error raised by
    the fallback must not report the plugin's failure as the context it happened in."""

    def broken(**kwargs):
        raise TypeError("plugin bug")

    def terminal(_payload):
        raise RuntimeError("provider down")

    _register(manager, kind, broken)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="provider down") as info:
        _run(kind, terminal)

    assert info.value.__context__ is None
    warnings = [r for r in caplog.records if "broken" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("kind", _KINDS)
def test_middleware_raising_is_still_reported_and_skipped(manager, kind, caplog):
    """No behaviour change for a middleware that itself raises: reported once through the
    manager's warn-once reporter, then the frame is skipped and the terminal still runs."""

    def broken(**kwargs):
        raise TypeError("plugin bug")

    reported = []
    original_report = manager._report_hook_failure

    def spy(hook_name, cb, kwargs, exc, *, surface="Hook"):
        reported.append((hook_name, cb, type(exc), surface))
        return original_report(hook_name, cb, kwargs, exc, surface=surface)

    manager._report_hook_failure = spy
    _register(manager, kind, broken)

    assert _run(kind, lambda payload: "ran") == "ran"
    assert reported == [(kind, broken, TypeError, "Middleware")]


@pytest.mark.parametrize("kind", _KINDS)
def test_middleware_raising_after_downstream_error_is_reported(manager, kind):
    """A middleware that swallows the downstream error and raises its own is reported, and its
    own error (not the provider's) propagates, as before."""

    def translating(**kwargs):
        try:
            kwargs["next_call"]()
        except Exception:
            raise KeyError("translated")

    reported = []
    manager._report_hook_failure = lambda hook_name, cb, kwargs, exc, surface: reported.append(type(exc))
    _register(manager, kind, translating)

    with pytest.raises(KeyError, match="translated"):
        _run(kind, _raise_with_implicit_context)

    assert reported == [KeyError]
