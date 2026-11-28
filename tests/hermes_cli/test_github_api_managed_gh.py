"""Update checks can reuse a PM-managed gh login outside the desktop's PATH."""
from io import BytesIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import github_api, source_releases

pytestmark = pytest.mark.real_release_channels


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for name in github_api.GITHUB_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(github_api, "_gh_cli_probed", False)
    monkeypatch.setattr(github_api, "_gh_cli_cache", None)


@pytest.mark.parametrize("path_result", [
    FileNotFoundError(),
    subprocess.TimeoutExpired("gh", 3),
    subprocess.CompletedProcess([], 1, ""),
    subprocess.CompletedProcess([], 0, " \n"),
])
def test_managed_login_authenticates_stable_lookup(monkeypatch, tmp_path, path_result, caplog):
    binary = tmp_path / "managed tools" / "gh"
    credential = "test-managed-credential"
    lookup = Mock(return_value=SimpleNamespace(binary=binary))
    monkeypatch.setattr("pm.install.installed_package", lookup)

    def run(command, **kwargs):
        assert command[1:] == ["auth", "token"]
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 3
        if command[0] == "gh":
            if isinstance(path_result, Exception):
                raise path_result
            return path_result
        assert command[0] == str(binary)
        return subprocess.CompletedProcess(command, 0, f" {credential}\n")

    cli = Mock(side_effect=run)
    monkeypatch.setattr(github_api.subprocess, "run", cli)
    seen = []
    commit = "a" * 40

    def urlopen(request, **kwargs):
        seen.append(request)
        assert request.full_url.startswith("https://api.github.com/repos/NousResearch/hermes-agent/")
        payload = ({"tag_name": "v1.2.3", "draft": False, "prerelease": False}
                   if request.full_url.endswith("/releases/latest") else {"sha": commit})
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    target = source_releases._resolve_stable("NousResearch/hermes-agent")
    assert (target.version, target.commit) == ("1.2.3", commit)
    assert len(seen) == 2
    assert all(request.get_header("Authorization") == f"Bearer {credential}" for request in seen)
    assert github_api.github_token() == credential
    assert cli.call_count == 2  # Both release requests reuse the cached login.
    lookup.assert_called_once_with("gh", allow_outdated=True)
    assert credential not in caplog.text


@pytest.mark.parametrize("managed_result", [
    None,
    SimpleNamespace(binary=None),
    SimpleNamespace(binary=Path("managed tools") / "gh"),
    OSError("unavailable"),
    ValueError("unavailable"),
    KeyError("unavailable"),
    ImportError("unavailable"),
])
def test_unavailable_managed_login_stays_anonymous_and_caches_miss(monkeypatch, managed_result):
    lookup = Mock(side_effect=managed_result) if isinstance(managed_result, Exception) else Mock(return_value=managed_result)
    monkeypatch.setattr("pm.install.installed_package", lookup)
    cli = Mock(side_effect=FileNotFoundError())
    monkeypatch.setattr(github_api.subprocess, "run", cli)
    assert github_api.github_token() is None
    assert github_api.github_token() is None
    lookup.assert_called_once_with("gh", allow_outdated=True)
    expected_calls = 2 if getattr(managed_result, "binary", None) is not None else 1
    assert cli.call_count == expected_calls
