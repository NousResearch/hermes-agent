"""One bad plugin CLI command must not take the others down with it (issue #130607).

``_register_plugin_cli_commands`` ran discovery and every ``add_parser`` call inside a single
``try``. The first descriptor that raised — a handler-only command registered with
``setup_fn=None``, a ``setup_fn`` that raises, or a name that collides with a built-in — unwound
the loop, so every later plugin's command vanished from the parser table and the CLI answered
``'<cmd>' is not a hermes command`` for all of them. The exception itself was recorded at DEBUG
only, which is why the symptom looked like "the plugin is not installed".

Invariants: a missing ``setup_fn`` keeps the command and its handler; a command that fails to
attach is skipped and named in a WARNING; the commands around it still attach.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_plugin_cli


def _subparsers() -> argparse._SubParsersAction:
    return argparse.ArgumentParser(prog="hermes").add_subparsers(dest="command")


def _descriptor(name, *, setup_fn=None, handler_fn=None) -> dict:
    return {
        "name": name,
        "help": f"{name} (test-only)",
        "description": "",
        "setup_fn": setup_fn,
        "handler_fn": handler_fn if handler_fn is not None else (lambda _args: None),
    }


def _raising_setup(_parser):
    raise RuntimeError("plugin argparse tree is broken")


# ── attach_plugin_cli_command ──────────────────────────────────────────────


def test_handler_only_command_attaches_without_a_setup_fn():
    """A plugin that dispatches on raw argv registers ``setup_fn=None``; keep its handler.

    These handlers take a token list (``args[0] if args else ""``), so the adapter must hand them
    the trailing argv rather than an argparse Namespace — the shape that raised
    ``TypeError: 'Namespace' object is not subscriptable`` inside the plugin.
    """
    subs = _subparsers()
    received = []

    main_plugin_cli.attach_plugin_cli_command(
        subs, _descriptor("handleronly", handler_fn=received.append)
    )

    assert "handleronly" in subs.choices
    parsed = subs.choices["handleronly"].parse_args(["status", "--json"])
    assert callable(getattr(parsed, "func", None))
    parsed.func(parsed)
    assert received == [["status", "--json"]], "the handler received something other than argv"


def test_setup_fn_still_shapes_the_command_parser():
    """The guarded call must still run a real ``setup_fn`` (negative control)."""
    subs = _subparsers()

    def _setup(parser):
        parser.add_argument("--ticket", required=True)

    main_plugin_cli.attach_plugin_cli_command(subs, _descriptor("withsetup", setup_fn=_setup))

    parsed = subs.choices["withsetup"].parse_args(["--ticket", "abc"])
    assert parsed.ticket == "abc"


# ── register_plugin_cli_commands isolation ────────────────────────────────


@pytest.fixture
def stub_plugin_sources(monkeypatch):
    """Point the discovery imports at stand-ins so the loop runs without real plugins."""

    def _install(discovered, registered, existing: tuple[str, ...] = ()):
        subs = _subparsers()
        for name in existing:
            subs.add_parser(name, description="built-in")

        memory = types.ModuleType("plugins.memory")
        memory.discover_plugin_cli_commands = lambda: list(discovered)

        plugins = types.ModuleType("hermes_cli.plugins")
        plugins.discover_plugins = lambda: None
        manager = types.SimpleNamespace(
            _cli_commands={info["name"]: info for info in registered}
        )
        plugins.get_plugin_manager = lambda: manager

        left_core = types.ModuleType("hermes_cli.left_core_migration")
        left_core.recover_for_command = lambda *_a, **_kw: None

        monkeypatch.setitem(sys.modules, "plugins.memory", memory)
        monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)
        monkeypatch.setitem(sys.modules, "hermes_cli.left_core_migration", left_core)
        monkeypatch.setattr(cli_main, "_plugin_cli_discovery_needed", lambda: True)
        monkeypatch.setattr(
            cli_main, "_resolve_deferred_platform_cli_command", lambda *_a, **_kw: None
        )
        return subs

    return _install


def test_raising_setup_fn_only_drops_that_command(stub_plugin_sources, caplog):
    subs = stub_plugin_sources(
        discovered=[_descriptor("first"), _descriptor("broken", setup_fn=_raising_setup)],
        registered=[_descriptor("later")],
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.main_plugin_cli"):
        main_plugin_cli.register_plugin_cli_commands(subs)

    assert "first" in subs.choices
    assert "later" in subs.choices, "the loop stopped at the broken command"
    # The parser is created before ``setup_fn`` runs, so it stays in the table; it must fail loudly
    # rather than fall through to the top-level help and exit 0 as if the command had worked.
    if "broken" in subs.choices:
        parsed = subs.choices["broken"].parse_args([])
        assert callable(getattr(parsed, "func", None)), "a skipped command must not exit 0 quietly"
        assert parsed.func(parsed) == main_plugin_cli._SKIPPED_EXIT_CODE
    assert any(
        record.levelno == logging.WARNING and "broken" in record.getMessage()
        for record in caplog.records
    ), "the skipped command must be named at WARNING, not swallowed at DEBUG"


def test_name_colliding_with_a_builtin_does_not_break_the_cli(
    stub_plugin_sources, caplog
):
    """``add_parser`` raises on a duplicate name; that must not escape the loop."""
    subs = stub_plugin_sources(
        discovered=[_descriptor("chat"), _descriptor("aftercollision")],
        registered=[_descriptor("registeredafter")],
        existing=("chat",),
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.main_plugin_cli"):
        main_plugin_cli.register_plugin_cli_commands(subs)

    # The built-in wins: the plugin's duplicate must not replace it.
    assert subs.choices["chat"].description == "built-in"
    assert "aftercollision" in subs.choices
    assert "registeredafter" in subs.choices
    assert any(
        record.levelno == logging.WARNING and "chat" in record.getMessage()
        for record in caplog.records
    )


def test_missing_setup_fn_does_not_stop_the_registered_commands(stub_plugin_sources):
    """End-to-end shape of the reported bug: dashclaw-style handler-only command first."""
    subs = stub_plugin_sources(
        discovered=[_descriptor("dashclawish"), _descriptor("noop")],
        registered=[_descriptor("tokenwatchish"), _descriptor("xchatish")],
    )

    main_plugin_cli.register_plugin_cli_commands(subs)

    for name in ("dashclawish", "noop", "tokenwatchish", "xchatish"):
        assert name in subs.choices, f"{name} disappeared from the plugin command table"
        # Membership alone cannot tell an attached command from a dead stub: the parser survives
        # ``add_parser`` even when the handler was never bound.
        parsed = subs.choices[name].parse_args([])
        assert callable(getattr(parsed, "func", None)), f"{name} attached without a handler"


def test_a_skipped_name_does_not_block_a_later_provider(stub_plugin_sources, caplog):
    """A deferred provider of the same command must still get its turn after a failure.

    The first descriptor fails before argparse builds a subparser (no ``help`` key), so the name is
    still free; recording it as seen would have hidden the working provider behind the broken one.
    """
    broken = {"name": "xchat", "description": "", "setup_fn": None, "handler_fn": lambda _a: None}
    later = _descriptor("xchat", setup_fn=lambda parser: parser.add_argument("--room"))
    subs = stub_plugin_sources(discovered=[broken], registered=[later])

    with caplog.at_level(logging.WARNING, logger="hermes_cli.main_plugin_cli"):
        main_plugin_cli.register_plugin_cli_commands(subs)

    assert "--room" in subs.choices["xchat"]._option_string_actions
    parsed = subs.choices["xchat"].parse_args([])
    assert parsed.func is later["handler_fn"], "the failed name was never retried by its provider"
    assert any(
        record.levelno == logging.WARNING and "xchat" in record.getMessage()
        for record in caplog.records
    )


def test_discovery_failure_is_reported_at_warning(monkeypatch, caplog):
    """Discovery failing removes every plugin command; that must not be DEBUG-only again."""
    subs = _subparsers()

    def _explode():
        raise RuntimeError("plugins.memory import blew up")

    memory = types.ModuleType("plugins.memory")
    memory.discover_plugin_cli_commands = _explode
    monkeypatch.setitem(sys.modules, "plugins.memory", memory)
    monkeypatch.setattr(cli_main, "_plugin_cli_discovery_needed", lambda: True)
    monkeypatch.setattr(cli_main, "_first_positional_argv", lambda: "xchat")
    monkeypatch.setattr(
        cli_main, "_resolve_deferred_platform_cli_command", lambda *_a, **_kw: None
    )
    left_core = types.ModuleType("hermes_cli.left_core_migration")
    left_core.recover_for_command = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "hermes_cli.left_core_migration", left_core)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.main_plugin_cli"):
        main_plugin_cli.register_plugin_cli_commands(subs)

    assert not subs.choices
    assert any(
        record.levelno == logging.WARNING and "discovery failed" in record.getMessage()
        for record in caplog.records
    ), "a failed discovery must be named at WARNING"


def test_plugin_registration_runs_after_every_builtin(monkeypatch):
    """Built-ins must be in the table before plugin commands are attached.

    A plugin that claims a built-in's name otherwise hits ``add_parser`` outside the guarded loop,
    raising ``argparse.ArgumentError: conflicting subparser`` out of ``_build_cli_parser`` and taking
    every ``hermes`` invocation on the host with it.
    """
    order: list[str] = []
    monkeypatch.setattr(cli_main, "_register_plugin_cli_commands", lambda _subs: order.append("plugins"))
    monkeypatch.setattr(cli_main, "build_webapp_parser", lambda *a, **k: order.append("webapp"))
    monkeypatch.setattr(
        cli_main, "build_prompt_size_parser", lambda *a, **k: order.append("prompt-size")
    )

    cli_main._build_cli_parser()

    assert order == ["webapp", "prompt-size", "plugins"], f"registration order was {order}"
