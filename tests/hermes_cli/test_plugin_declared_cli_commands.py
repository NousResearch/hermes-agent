"""Plugin CLI commands declared in ``plugin.yaml`` (``cli_commands:``), top-level or under a built-in.

A declared command attaches BEFORE plugin discovery and its stub imports only its own plugin, so
``hermes <declared>`` never pays ``discover_plugins()``. ``parent:`` (and ``register_cli_command(
parent=)``) attaches a sub-verb under a built-in — ``hermes auth <verb>`` — which discovery could
never reach, because built-in invocations skip it.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

import hermes_cli.main as main_mod
from hermes_cli import plugins as plugins_mod


def _write_plugin(root, name, marker_dir, parent=""):
    plugin = root / name
    plugin.mkdir()
    parent_row = f"\n            parent: {parent}" if parent else ""
    (plugin / "plugin.yaml").write_text(textwrap.dedent(f"""\
        name: {name}
        kind: backend
        cli_commands:
          - name: {name}
            help: the {name} command{parent_row}
        """), encoding="utf-8")
    marker = (marker_dir / f"{name}.registered").as_posix()
    (plugin / "__init__.py").write_text(textwrap.dedent(f"""\
        def _setup(parser):
            parser.add_argument("--flag", action="store_true")

        def _run(args):
            return {name!r}

        def register(ctx):
            open({marker!r}, "w").close()
            ctx.register_cli_command({name!r}, help="h", setup_fn=_setup, handler_fn=_run,
                                     parent={parent!r} or None)
        """), encoding="utf-8")


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    """``alpha`` declares top-level ``hermes alpha``; ``beta`` declares ``hermes auth beta``."""
    bundled, markers = tmp_path / "bundled", tmp_path / "markers"
    bundled.mkdir()
    markers.mkdir()
    _write_plugin(bundled, "alpha", markers)
    _write_plugin(bundled, "beta", markers, parent="auth")
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    discovered = []
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda *a, **k: discovered.append(1))
    plugins_mod._reset_plugin_managers_for_tests()
    yield markers, discovered
    plugins_mod._reset_plugin_managers_for_tests()


def _parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    plugins_mod._reset_plugin_managers_for_tests()
    parser, subparsers = main_mod._build_cli_parser()
    return main_mod._parse_cli_args(parser, subparsers, argv)


def test_a_declared_command_runs_importing_only_its_own_plugin_without_discovery(plugins, monkeypatch):
    markers, discovered = plugins

    args = _parse(["alpha", "--flag"], monkeypatch)

    assert args.flag is True and args.func(args) == "alpha"
    assert sorted(p.name for p in markers.iterdir()) == ["alpha.registered"]
    assert discovered == []
    # Positive control: an unknown, undeclared first token DOES reach discovery through the same patch.
    monkeypatch.setattr(sys, "argv", ["hermes", "no-such-command"])
    main_mod._build_cli_parser()
    assert discovered == [1]


def test_a_declared_parent_verb_attaches_under_the_builtin_and_known_verbs_scan_nothing(plugins, monkeypatch):
    markers, discovered = plugins

    args = _parse(["auth", "beta", "--flag"], monkeypatch)

    assert args.command == "auth" and args.auth_action == "beta"
    assert args.flag is True and args.func(args) == "beta"  # the plugin's handler, not cmd_auth
    assert sorted(p.name for p in markers.iterdir()) == ["beta.registered"] and discovered == []
    # A built-in verb of the same parent keeps its own handler and never reads a manifest.
    monkeypatch.setattr(plugins_mod, "discover_declared_cli_commands", lambda: pytest.fail("scanned"))
    builtin = _parse(["auth", "list"], monkeypatch)
    assert builtin.auth_action == "list" and builtin.func is not args.func
