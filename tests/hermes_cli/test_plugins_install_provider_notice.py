"""``hermes plugins install`` tells the truth about a provider plugin the operator did not enable.

Provider discovery loads a user plugin whose manifest declares ``kind: model-provider`` without a
``plugins.enabled`` entry (documented in the model-provider plugin guide), and ``hermes plugins list``
shows it as enabled. The install line for such a plugin says it is active and how to turn it off,
instead of "installed but not enabled". Every other plugin, and every enabled install, prints what it
printed before; nothing is written to ``plugins.disabled``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd
from tests.pm.test_plugin_survival_contract import _uv_available, admission_env

pytestmark = pytest.mark.skipif(not _uv_available(), reason="uv not on PATH")

_PROVIDER_SOURCE = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", aliases=(), base_url="acp://{name}",
                                      auth_type="external_process"))
    """
)


def _clear_provider_caches():
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


@pytest.fixture
def world(admission_env):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    _root, home = admission_env
    _clear_provider_caches()
    yield home
    _clear_provider_caches()


def _git_repo(path: Path, name: str, *, provider: bool) -> Path:
    path.mkdir(parents=True)
    kind = "kind: model-provider\n" if provider else ""
    (path / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\n{kind}", encoding="utf-8")
    source = _PROVIDER_SOURCE.format(name=name) if provider else "def register(ctx):\n    pass\n"
    (path / "__init__.py").write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "fixture")):
        subprocess.run(["git", *args], cwd=path, env=env, check=True, capture_output=True, timeout=30)
    return path


def _old_line(name: str) -> str:
    return f"Plugin installed but not enabled. Run `hermes plugins enable {name}` to activate."


def _new_line(name: str) -> str:
    return ("Plugin installed. Provider plugins are active without an entry in plugins.enabled; "
            f"`hermes plugins disable {name}` turns it off.")


def _install(repo: Path, capsys, **flags) -> str:
    plugins_cmd.cmd_install(repo.as_uri(), yes_deps=True, allow_removed=True, **flags)
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("flags", [{"enable": False}, {"enable": None}, {"enable": None, "no_deps": True}])
def test_a_provider_plugin_left_unenabled_is_reported_as_active(world, tmp_path, capsys, flags):
    repo = _git_repo(tmp_path / "src", "noticeprov", provider=True)
    config_before = (world / "config.yaml").read_bytes()

    out = _install(repo, capsys, **flags)

    assert _new_line("noticeprov") in out
    assert "installed but not enabled" not in out
    assert (world / "config.yaml").read_bytes() == config_before  # nothing enabled, nothing disabled
    _clear_provider_caches()
    import providers
    assert providers.get_provider_profile("noticeprov") is not None  # the line matches what loads


def test_an_ordinary_plugin_left_unenabled_prints_the_same_line_as_before(world, tmp_path, capsys):
    repo = _git_repo(tmp_path / "src", "noticeplain", provider=False)

    out = _install(repo, capsys, enable=False)

    assert _old_line("noticeplain") in out
    assert "Provider plugins are active" not in out


def test_a_provider_plugin_listed_in_plugins_disabled_keeps_the_not_enabled_line(world, tmp_path, capsys):
    import hermes_yaml as yaml
    config = yaml.safe_load((world / "config.yaml").read_text(encoding="utf-8")) or {}
    config.setdefault("plugins", {})["disabled"] = ["noticeprov"]
    (world / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    repo = _git_repo(tmp_path / "src", "noticeprov", provider=True)

    out = _install(repo, capsys, enable=False)

    assert _old_line("noticeprov") in out
    assert "Provider plugins are active" not in out


def test_an_enabled_provider_install_prints_what_it_printed_before(world, tmp_path, capsys):
    repo = _git_repo(tmp_path / "src", "noticeprov", provider=True)

    out = _install(repo, capsys, enable=True)

    assert "✓ Plugin noticeprov enabled." in out
    assert "installed but not enabled" not in out
    assert "Provider plugins are active" not in out
