"""Canonical OAuth owners import independently and reject retained settings in another profile."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from auth.providers.anthropic import read_claude_code_credentials, resolve_anthropic_token
from auth.providers.codex import resolve_codex_runtime_credentials
from auth.providers.nous import resolve_nous_runtime_credentials
from auth.providers.nous_status import get_nous_auth_status
from tests.auth.test_pool_environment import environment, profile_scope, multiplex_scope


@pytest.mark.parametrize("operation", [
    read_claude_code_credentials,
    resolve_anthropic_token,
    resolve_codex_runtime_credentials,
    resolve_nous_runtime_credentials,
    get_nous_auth_status,
])
def test_retained_oauth_settings_reject_another_profile_before_read(tmp_path, monkeypatch, operation):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    supplied = environment(a)
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            operation(environment=supplied)
    assert not (a / "auth.json").exists()
    assert not (b / "auth.json").exists()


def test_all_authentication_modules_import_without_cli(tmp_path):
    code = r'''
import importlib, importlib.abc, pkgutil, sys
class BlockCLI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"hermes_cli", "nous_cli"}:
            raise AssertionError("reverse CLI dependency: " + fullname)
sys.meta_path.insert(0, BlockCLI())
import auth
for info in pkgutil.walk_packages(auth.__path__, "auth."):
    importlib.import_module(info.name)
'''
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, HERMES_HOME=str(tmp_path)),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
