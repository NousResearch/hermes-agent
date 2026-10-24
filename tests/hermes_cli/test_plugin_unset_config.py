"""``PluginContext.unset_config`` deletes one plugin setting through the comment-preserving writer."""

from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

_CONFIG = """\
# operator note: keep this header
plugins:
  entries:
    delegate-roles:
      settings:
        roles:
          reviewer:
            model: m1  # pinned on purpose
            reasoning_effort: high
          writer:
            model: m2
        endpoint: https://example.invalid
"""


@pytest.fixture
def config_path(tmp_path: Path):
    home = tmp_path / "profile"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    token = set_hermes_home_override(home)
    try:
        yield path
    finally:
        reset_hermes_home_override(token)


def _ctx() -> PluginContext:
    return PluginContext(PluginManifest(name="delegate-roles"), PluginManager())


def test_unset_removes_key_and_keeps_siblings_and_comments(config_path: Path) -> None:
    ctx = _ctx()

    assert ctx.unset_config("roles.reviewer.reasoning_effort") is True

    text = config_path.read_text(encoding="utf-8")
    settings = yaml.safe_load(text)["plugins"]["entries"]["delegate-roles"]["settings"]
    assert settings == {
        "roles": {"reviewer": {"model": "m1"}, "writer": {"model": "m2"}},
        "endpoint": "https://example.invalid",
    }
    assert "# operator note: keep this header" in text
    assert "# pinned on purpose" in text
    assert ctx.get_config("roles.reviewer.reasoning_effort", default="gone") == "gone"


def test_unset_of_absent_key_returns_false_without_rewriting(config_path: Path) -> None:
    before = config_path.stat().st_mtime_ns

    assert _ctx().unset_config("roles.missing") is False

    assert config_path.stat().st_mtime_ns == before
    assert config_path.read_text(encoding="utf-8") == _CONFIG
