"""OAuth-shaped model-provider plugins are first-class in `hermes auth` (#116408).

The seam: ``ProviderProfile.auth_handler``. These tests drive the real ``hermes auth`` argparse
surface against a plugin discovered from an isolated HERMES_HOME, so they fail if the dispatch is
dropped, reordered after the built-in paths, or the registry stops admitting non-api-key profiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


register_provider(ProviderProfile(name="__NAME__", auth_type="oauth_external",
    base_url="https://example.invalid/v1", auth_handler=handler))
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

    def _install(name: str = "fake-auth", *, with_handler: bool = True) -> Path:
        """Write (or rewrite) the fixture plugin and re-run discovery."""
        plugin_dir = tmp_path / "hermes" / "plugins" / "model-providers" / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {name}\nkind: model-provider\nversion: 0.0.1\n"
            "description: provider auth seam fixture\n", encoding="utf-8")
        source = _PLUGIN_SOURCE.replace("__NAME__", name)
        if not with_handler:
            source = source.replace(", auth_handler=handler", "")
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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["add", "fake-auth", "--label", "work", "--api-key", "sk-fixture"],
         {"action": "add", "provider": "fake-auth", "label": "work", "target": None, "api_key": "sk-fixture"}),
        (["status", "fake-auth"],
         {"action": "status", "provider": "fake-auth", "label": None, "target": None, "api_key": None}),
        (["logout", "fake-auth"],
         {"action": "logout", "provider": "fake-auth", "label": None, "target": None, "api_key": None}),
        (["refresh", "fake-auth", "acct-2"],
         {"action": "refresh", "provider": "fake-auth", "label": None, "target": "acct-2", "api_key": None}),
    ],
)
def test_oauth_plugin_owns_every_auth_action(tmp_path, install_provider, capsys, argv, expected):
    """An oauth_external plugin registers and `hermes auth <action>` reaches its handler, args included,
    before any built-in path (nothing printed, nothing written to the pool)."""
    install_provider()

    import hermes_cli.auth as auth_mod
    from hermes_cli.auth_commands import auth_command

    assert auth_mod.resolve_provider("fake-auth") == "fake-auth"
    assert auth_mod.PROVIDER_REGISTRY["fake-auth"].auth_type == "oauth_external"

    auth_command(_parse_auth_args(argv))

    assert _log(tmp_path) == [expected]
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "hermes" / "auth.json").exists()


def test_oauth_plugin_without_handler_fails_loud_and_builtins_are_untouched(tmp_path, install_provider):
    """No handler on an oauth-shaped profile = a clear error naming the missing hook (never a silent
    api-key prompt or "Unknown provider"); a built-in provider never consults the seam."""
    install_provider("handlerless", with_handler=False)

    from hermes_cli.auth_commands import auth_command

    with pytest.raises(SystemExit) as excinfo:
        auth_command(_parse_auth_args(["add", "handlerless"]))
    assert "ships no auth_handler" in str(excinfo.value) and "oauth_external" in str(excinfo.value)

    auth_command(_parse_auth_args(["add", "openrouter", "--api-key", "sk-or-fixture", "--label", "personal"]))
    assert _log(tmp_path) == []
    pool = json.loads((tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8"))["credential_pool"]
    assert next(e for e in pool["openrouter"] if e["access_token"] == "sk-or-fixture")["label"] == "personal"
