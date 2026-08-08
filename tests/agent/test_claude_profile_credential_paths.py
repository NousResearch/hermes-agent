"""Claude Code credential lookups must follow the active profile.

Claude Code keeps one login per configuration directory. The directory comes
from ``CLAUDE_CONFIG_DIR``. The matching secret store comes from
``CLAUDE_SECURESTORAGE_CONFIG_DIR``. Two lookups in the Anthropic adapter
ignored both variables:

* the macOS Keychain read used one fixed service name, so it found the
  default profile's login, or nothing at all;
* the ``.credentials.json`` read and write used one fixed path under
  ``~/.claude``, so a run under a named profile could change a different
  profile's file.

These tests pin the corrected behaviour. No test reads or writes a real
Keychain entry: the ``security`` command is always a mock.
"""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_adapter import (
    _read_claude_code_credentials_from_file,
    _read_claude_code_credentials_from_keychain,
    _write_claude_code_credentials,
    claude_credentials_path,
    claude_keychain_service_candidates,
)

# Every Keychain test here supplies its own ``subprocess.run`` mock, so it
# never reaches the real store.
pytestmark = pytest.mark.allow_macos_keychain

BASE_SERVICE = "Claude Code-credentials"


def _suffixed(path_string: str) -> str:
    digest = hashlib.sha256(path_string.encode("utf-8")).hexdigest()[:8]
    return f"{BASE_SERVICE}-{digest}"


def _services_queried(mock_run) -> list[str]:
    """Return the ``-s <service>`` value of every ``security`` call."""
    services = []
    for call in mock_run.call_args_list:
        argv = call.args[0]
        services.append(argv[argv.index("-s") + 1])
    return services


class TestKeychainServiceName:
    def test_default_profile_uses_the_base_service_name(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        assert claude_keychain_service_candidates() == [BASE_SERVICE]

    def test_named_profile_uses_the_suffixed_service_name(self, monkeypatch, tmp_path):
        profile = tmp_path / "claude-hermes"
        profile.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        assert _suffixed(str(profile)) in claude_keychain_service_candidates()
        assert BASE_SERVICE not in claude_keychain_service_candidates()

    def test_config_dir_equal_to_the_default_stays_the_base_service(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
        assert claude_keychain_service_candidates() == [BASE_SERVICE]

    def test_explicit_directory_argument_wins_over_the_environment(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "from-env"))
        asked = tmp_path / "from-argument"
        assert _suffixed(str(asked)) in claude_keychain_service_candidates(asked)

    def test_keychain_read_asks_for_the_profile_service(self, monkeypatch, tmp_path):
        profile = tmp_path / "claude-hermes"
        profile.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            _read_claude_code_credentials_from_keychain()

        queried = _services_queried(mock_run)
        assert queried, "the reader made no security call"
        assert BASE_SERVICE not in queried
        assert _suffixed(str(profile)) in queried

    def test_keychain_read_returns_the_profile_entry(self, monkeypatch, tmp_path):
        profile = tmp_path / "claude-hermes"
        profile.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        payload = json.dumps(
            {"claudeAiOauth": {"accessToken": "profile-token", "expiresAt": 9999999999999}}
        )

        def fake_run(argv, **kwargs):
            service = argv[argv.index("-s") + 1]
            if service == _suffixed(str(profile)):
                return MagicMock(returncode=0, stdout=payload, stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run", side_effect=fake_run):
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "profile-token"


class TestCredentialsFilePath:
    def test_default_profile_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
        assert claude_credentials_path() == tmp_path / ".claude" / ".credentials.json"

    def test_config_dir_moves_the_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "spare"))
        assert claude_credentials_path() == tmp_path / "spare" / ".credentials.json"

    def test_securestorage_dir_wins_over_config_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "spare"))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(tmp_path / "vault"))
        assert claude_credentials_path() == tmp_path / "vault" / ".credentials.json"

    def test_file_read_follows_the_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        default_file = tmp_path / ".claude" / ".credentials.json"
        default_file.parent.mkdir(parents=True)
        default_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "default-token"}})
        )
        profile_file = tmp_path / "spare" / ".credentials.json"
        profile_file.parent.mkdir(parents=True)
        profile_file.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "spare-token"}})
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "spare"))

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "spare-token"

    def test_file_write_stays_inside_the_selected_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "spare"))

        _write_claude_code_credentials("new-token", "new-refresh", 9999999999999)

        written = tmp_path / "spare" / ".credentials.json"
        assert written.exists()
        assert json.loads(written.read_text())["claudeAiOauth"]["accessToken"] == "new-token"
        assert not (tmp_path / ".claude" / ".credentials.json").exists()

    def test_file_write_keeps_owner_only_permissions(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "spare"))

        _write_claude_code_credentials("new-token", "new-refresh", 9999999999999)

        written = tmp_path / "spare" / ".credentials.json"
        assert written.stat().st_mode & 0o077 == 0
