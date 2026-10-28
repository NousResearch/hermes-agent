"""Readonly Codex profiles refuse clone/mirror before credential-copy I/O."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("with_bom", [False, True])
@pytest.mark.parametrize("operation", ["clone", "mirror"])
@pytest.mark.parametrize("store_kind", ["openai-codex", "codex", "chatgpt", "chatgpt-codex", "invalid-json", "invalid-shape"])
def test_codex_profile_copy_is_refused_before_copy_io(tmp_path, operation, store_kind, with_bom):
    probe = r'''
import json
import os
from pathlib import Path
import shutil
import sys
home = Path(os.environ["HERMES_HOME"])
home.mkdir()
(home / "config.yaml").write_text("model:\n  provider: openai-codex\n  default: synthetic\n")
store_kind = sys.argv[2]
store = {"providers": {store_kind: {"tokens": {"access_token": "synthetic", "refresh_token": "synthetic-refresh"}}}}
content = "{broken synthetic-private-value" if store_kind == "invalid-json" else "[]" if store_kind == "invalid-shape" else json.dumps(store)
(home / "auth.json").write_bytes((b"\xef\xbb\xbf" if sys.argv[3] == "True" else b"") + content.encode())
(home / ".env").write_text("SYNTHETIC_KEY=synthetic\n")
before = (home / "auth.json").read_bytes()
from agent.codex_auth_policy import bind_codex_auth_policy
from hermes_cli.profiles import create_profile
from tui_gateway import server
bind_codex_auth_policy(read_only=True)
copies = []
def forbidden(*args, **kwargs):
    copies.append(True)
    raise AssertionError("copy must be refused before shutil I/O")
shutil.copy2 = forbidden
shutil.copytree = forbidden
try:
    if sys.argv[1] == "clone":
        from hermes_cli.profiles import create_profile
        create_profile("clone", clone_all=True, no_alias=True)
    else:
        from tui_gateway import server
        target = home.parent / "target"
        target.mkdir()
        server._mirror_launch_credentials(target, {})
except RuntimeError as exc:
    assert str(exc) == "Codex credential copying is disabled by --codex-auth-read-only"
    assert "synthetic-private-value" not in str(exc)
else:
    raise AssertionError("Codex credential copy not refused")
assert not copies, "copy I/O was reached"
assert (home / "auth.json").read_bytes() == before
assert not (home.parent / "target" / "auth.json").exists()
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, operation, store_kind, str(with_bom)],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")},
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("with_bom", [False, True])
def test_readonly_copy_guard_accepts_non_codex_store_with_optional_bom(tmp_path, with_bom):
    probe = r'''
import json
import os
from pathlib import Path
import sys
home = Path(os.environ["HERMES_HOME"])
home.mkdir()
content = json.dumps({"providers": {"openrouter": {"api_key": "synthetic"}}}).encode()
(home / "auth.json").write_bytes((b"\xef\xbb\xbf" if sys.argv[1] == "True" else b"") + content)
from agent.codex_auth_policy import bind_codex_auth_policy, refuse_codex_credential_copy
bind_codex_auth_policy(read_only=True)
refuse_codex_credential_copy(home)
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, str(with_bom)],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")},
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
