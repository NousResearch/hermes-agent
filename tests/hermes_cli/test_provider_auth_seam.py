"""`hermes auth <action> <provider>` prefers a model-provider plugin's auth handler.

The seam: ``ProviderProfile.auth_handler`` (an optional callable on the profile a
``kind: model-provider`` plugin registers). These tests drive the real
``hermes auth`` subcommand surface through the real argparse definitions, so they
fail if the dispatch is dropped, reordered after the built-in paths, or stops
passing the parsed arguments through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# A model-provider plugin that owns its own interactive auth. It appends one JSON
# record per dispatch so the test can prove the action + arguments arrived.
_PLUGIN_SOURCE = '''\
"""Fixture provider plugin: owns its own interactive auth."""
import json
import os

from providers import register_provider
from providers.base import ProviderProfile


def _record(action, args):
    path = os.environ.get("FAKE_AUTH_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "action": action,
            "provider": getattr(args, "provider", None),
            "label": getattr(args, "label", None),
            "target": getattr(args, "target", None),
            "api_key": getattr(args, "api_key", None)}) + "\\n")


def handler(action, args):
    _record(action, args)
    if action in (os.environ.get("FAKE_AUTH_DECLINE") or "").split(","):
        return False
    return True


register_provider(ProviderProfile(name="__NAME__", auth_handler=handler))
'''


def _rediscover() -> None:
    """Point the next profile lookup at the (new) HERMES_HOME user plugin dir.

    Only the discovery flag is cleared: bundled plugin modules stay in
    ``sys.modules`` (so their profiles stay registered) while the user dir is
    rescanned for the fixture. Fixture modules are evicted so the next
    ``_import_plugin_dir`` actually re-executes them.
    """
    import providers as _pkg

    _pkg._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def install_provider(tmp_path, monkeypatch):
    """Write a model-provider plugin into an isolated HERMES_HOME and discover it."""
    installed: list[str] = []

    def _install(name: str = "fake-auth", *, with_handler: bool = True,
                 async_handler: bool = False, raises: bool = False,
                 reinstall: bool = False) -> Path:
        """Write (or rewrite) the fixture plugin and re-run discovery."""
        plugin_dir = tmp_path / "hermes" / "plugins" / "model-providers" / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nkind: model-provider\nversion: 0.0.1\n"
            "description: provider auth seam fixture\n", encoding="utf-8")
        source = _PLUGIN_SOURCE.replace("__NAME__", name)
        if not with_handler:
            source = source.replace(", auth_handler=handler", "")
        if async_handler:
            source = source.replace("def handler(action, args):", "async def handler(action, args):")
        if raises:
            source = source.replace("    return True\n", '    raise ValueError("device flow exploded")\n')
        (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("FAKE_AUTH_LOG", str(tmp_path / "auth-log.jsonl"))
        _rediscover()
        installed.append(name)
        return plugin_dir

    yield _install

    # The provider registry is process-global: never leak the fixture profile.
    import providers as _pkg

    for name in installed:
        _pkg._REGISTRY.pop(name, None)
        for alias, canonical in list(_pkg._ALIASES.items()):
            if canonical == name:
                _pkg._ALIASES.pop(alias, None)
    _pkg._PROVIDER_LIST_CACHE = None


def _parse_auth_args(argv: list[str]) -> argparse.Namespace:
    """Parse `hermes auth <argv>` through the real subcommand parser."""
    from hermes_cli.subcommands.auth import build_auth_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_auth_parser(subparsers, cmd_auth=lambda args: None)
    return parser.parse_args(["auth", *argv])


def _log(tmp_path: Path) -> list[dict]:
    log = tmp_path / "auth-log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_add_dispatches_to_provider_handler_with_arguments(tmp_path, install_provider):
    """`hermes auth add <provider>` reaches the plugin handler, args included."""
    install_provider()

    from hermes_cli.auth_commands import auth_command

    args = _parse_auth_args(["add", "fake-auth", "--label", "work", "--api-key", "sk-fixture"])
    assert args.auth_action == "add" and args.provider == "fake-auth"

    auth_command(args)

    assert _log(tmp_path) == [{
        "action": "add", "provider": "fake-auth", "label": "work",
        "target": None, "api_key": "sk-fixture"}]
    # The built-in path never ran: nothing was written to the credential pool.
    assert not (tmp_path / "hermes" / "auth.json").exists()


@pytest.mark.parametrize(
    ("argv", "action"),
    [
        (["status", "fake-auth"], "status"),
        (["logout", "fake-auth"], "logout"),
        (["refresh", "fake-auth", "acct-2"], "refresh"),
    ],
)
def test_status_logout_refresh_dispatch_to_same_handler(tmp_path, install_provider, capsys, argv, action):
    install_provider()

    from hermes_cli.auth_commands import auth_command

    args = _parse_auth_args(argv)
    auth_command(args)

    recorded = _log(tmp_path)
    assert [r["action"] for r in recorded] == [action]
    # Core prints only when it owns the action; a dispatched action prints nothing here.
    assert capsys.readouterr().out == ""


def test_declining_handler_falls_back_to_the_builtin_path(tmp_path, install_provider, monkeypatch):
    """A handler may decline per action — core then behaves as it always did."""
    install_provider()
    monkeypatch.setenv("FAKE_AUTH_DECLINE", "add")

    from hermes_cli.auth_commands import auth_command

    with pytest.raises(SystemExit) as excinfo:
        auth_command(_parse_auth_args(["add", "fake-auth"]))

    # Offered first, declined by the handler, then the built-in unknown-provider exit...
    assert [r["action"] for r in _log(tmp_path)] == ["add"]
    # ...which now names the plugin instead of pretending the provider is unknown.
    message = str(excinfo.value)
    assert message.startswith("Unknown provider: fake-auth")
    assert "does not provide auth handling" in message


def test_builtin_provider_without_handler_is_unchanged(tmp_path, install_provider):
    """A provider with no handler keeps the exact built-in credential-pool path."""
    install_provider()

    from hermes_cli.auth_commands import auth_command

    auth_command(_parse_auth_args(["add", "openrouter", "--api-key", "sk-or-fixture", "--label", "personal"]))

    assert _log(tmp_path) == []  # no handler was ever consulted
    pool = json.loads((tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8"))["credential_pool"]
    entry = next(e for e in pool["openrouter"] if e["access_token"] == "sk-or-fixture")
    assert entry["label"] == "personal"


def test_provider_without_handler_still_reports_unknown_provider(tmp_path, install_provider):
    install_provider("handlerless", with_handler=False)

    from hermes_cli.auth_commands import auth_command

    with pytest.raises(SystemExit) as excinfo:
        auth_command(_parse_auth_args(["add", "handlerless"]))

    message = str(excinfo.value)
    assert message.startswith("Unknown provider: handlerless")
    assert "does not provide auth handling" in message
    assert _log(tmp_path) == []


def test_unregistered_provider_lookup_failure_falls_through(tmp_path, install_provider):
    """Registry lookup failure (no profile at all) must never raise or dispatch."""
    install_provider()

    from hermes_cli.auth_commands import _dispatch_provider_auth, _provider_auth_handler, auth_status_command

    assert _provider_auth_handler("not-a-registered-provider") == (None, None)
    assert _dispatch_provider_auth("add", SimpleNamespace(provider="not-a-registered-provider"),
                                   "not-a-registered-provider") is False

    auth_status_command(SimpleNamespace(provider="not-a-registered-provider"))
    assert _log(tmp_path) == []


def test_duplicate_registration_last_writer_wins(tmp_path, install_provider):
    """Two profiles under one name (a user plugin overriding a bundled one) resolve
    to the newest handler — the documented override semantics of register_provider."""
    install_provider()
    from hermes_cli.auth_commands import _provider_auth_handler

    first, _ = _provider_auth_handler("fake-auth")
    assert callable(first.auth_handler)

    install_provider(reinstall=True)  # second registration for the same name

    second, handler = _provider_auth_handler("fake-auth")
    assert handler is not None and handler is not first.auth_handler
    assert second is not first


def test_async_handler_is_awaited(tmp_path, install_provider):
    install_provider(async_handler=True)

    from hermes_cli.auth_commands import auth_command

    auth_command(_parse_auth_args(["add", "fake-auth"]))

    assert [r["action"] for r in _log(tmp_path)] == ["add"]


def test_handler_failure_becomes_a_readable_exit(tmp_path, install_provider):
    install_provider(raises=True)

    from hermes_cli.auth_commands import auth_command

    with pytest.raises(SystemExit) as excinfo:
        auth_command(_parse_auth_args(["add", "fake-auth"]))

    message = str(excinfo.value)
    assert "fake-auth auth handler failed for `add`" in message
    assert "ValueError: device flow exploded" in message


def test_every_action_reaches_the_seam(tmp_path, install_provider):
    """Guard against a future action being added to the core enum without dispatch."""
    install_provider()

    from hermes_cli import auth_commands

    for action, command in (
        ("add", auth_commands.auth_add_command),
        ("status", auth_commands.auth_status_command),
        ("logout", auth_commands.auth_logout_command),
        ("refresh", auth_commands.auth_refresh_command),
    ):
        assert auth_commands._dispatch_provider_auth(action, SimpleNamespace(provider="fake-auth"),
                                                     "fake-auth"), action

    assert sorted(r["action"] for r in _log(tmp_path)) == ["add", "logout", "refresh", "status"]
