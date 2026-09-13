"""Persisted env writes must remain usable within the same routed turn."""

import json
import os
import subprocess
import sys

import pytest

from agent import secret_scope as scopes
from hermes_cli import config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import build_subprocess_env


@pytest.mark.parametrize("value", ["new-target", "", None])
def test_persisted_write_refreshes_scope_before_same_turn_child(monkeypatch, tmp_path, value):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text("ACME_TOKEN=source\n", encoding="utf-8")
    (target / ".env").write_text("ACME_TOKEN=old-target\n", encoding="utf-8")
    (target / "config.yaml").write_text("terminal:\n  env_passthrough: [ACME_TOKEN]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("ACME_TOKEN", "source")
    monkeypatch.setattr(config, "is_managed", lambda: False)
    home_token = set_hermes_home_override(target)
    scope_token = scopes.set_secret_scope(scopes.build_profile_secret_scope(target))
    scopes.set_multiplex_active(True)
    try:
        before = scopes.current_secret_scope()
        if value is None:
            assert config.remove_env_value("ACME_TOKEN")
        else:
            config.save_env_value("ACME_TOKEN", value)
        after = scopes.current_secret_scope()
        assert before is not None and after is not None
        assert before["ACME_TOKEN"] == "old-target"
        assert after.generation != before.generation
        assert after.get("ACME_TOKEN") == value
        child_env = build_subprocess_env(profile_home=target, source_profile_home=source)
        child = subprocess.run(
            [sys.executable, "-c", "import json,os; print(json.dumps(os.getenv('ACME_TOKEN')))"],
            env=child_env, capture_output=True, text=True, timeout=10, check=True)
        assert json.loads(child.stdout) == value
        assert os.environ["ACME_TOKEN"] == "source"
        # An unrelated out-of-band write must STILL invalidate this refreshed scope.
        (target / ".env").write_text("ACME_TOKEN=unannounced\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="scope is stale"):
            scopes.build_profile_env_boundary(source, target)
    finally:
        scopes.set_multiplex_active(False)
        scopes.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
