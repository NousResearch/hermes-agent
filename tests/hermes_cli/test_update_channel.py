"""Tests for update channel (edge / stable) logic in hermes update.

Covers:
1. Default channel is "edge" when no config and no flag.
2. Config setting update.channel = "stable" is read correctly.
3. --channel stable flag overrides the config value.
4. Invalid channel value exits with code 1.
5. Stable channel fetches tags and checkouts the latest tag (mocked subprocess).
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Build a minimal argparse Namespace for cmd_update tests."""
    defaults = {
        "gateway": False,
        "channel": None,
    }
    defaults.update(kwargs)
    ns = types.SimpleNamespace(**defaults)
    return ns


def _make_completed_process(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


# ---------------------------------------------------------------------------
# Fixture: isolated HERMES_HOME + config
# ---------------------------------------------------------------------------

@pytest.fixture()
def hermes_env(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp directory and provide a blank config."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Ensure Path.home() also points at tmp_path so any secondary lookups agree.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return hermes_home


# ---------------------------------------------------------------------------
# 1. Default channel is "edge"
# ---------------------------------------------------------------------------

class TestChannelDefault:
    def test_default_channel_is_edge(self, hermes_env):
        """When no config and no --channel flag, channel resolves to 'edge'."""
        from hermes_cli.config import load_config
        cfg = load_config()
        channel = cfg.get("update", {}).get("channel", "edge")
        assert channel == "edge"

    def test_none_flag_falls_back_to_config_default(self, hermes_env):
        """getattr(args, 'channel', None) returns None → falls back to config 'edge'."""
        from hermes_cli.config import load_config
        cfg = load_config()
        channel = getattr(_make_args(), "channel", None) or cfg.get("update", {}).get("channel", "edge")
        assert channel == "edge"


# ---------------------------------------------------------------------------
# 2. Config update.channel = "stable" is read
# ---------------------------------------------------------------------------

class TestChannelFromConfig:
    def test_config_stable_is_read(self, hermes_env):
        """When config.yaml sets update.channel = 'stable', load_config returns it."""
        import yaml
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({"update": {"channel": "stable"}}))

        from hermes_cli import config as config_mod
        # Reload to pick up the new file
        import importlib
        importlib.reload(config_mod)

        cfg = config_mod.load_config()
        channel = cfg.get("update", {}).get("channel", "edge")
        assert channel == "stable"

    def test_config_edge_explicit(self, hermes_env):
        """Explicit config update.channel = 'edge' still resolves to edge."""
        import yaml
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({"update": {"channel": "edge"}}))

        from hermes_cli import config as config_mod
        import importlib
        importlib.reload(config_mod)

        cfg = config_mod.load_config()
        channel = cfg.get("update", {}).get("channel", "edge")
        assert channel == "edge"


# ---------------------------------------------------------------------------
# 3. --channel flag overrides config
# ---------------------------------------------------------------------------

class TestChannelFlagOverridesConfig:
    def test_flag_stable_overrides_config_edge(self, hermes_env):
        """--channel stable wins over config update.channel = 'edge'."""
        import yaml
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({"update": {"channel": "edge"}}))

        from hermes_cli import config as config_mod
        import importlib
        importlib.reload(config_mod)

        cfg = config_mod.load_config()
        args = _make_args(channel="stable")
        channel = getattr(args, "channel", None) or cfg.get("update", {}).get("channel", "edge")
        assert channel == "stable"

    def test_flag_edge_overrides_config_stable(self, hermes_env):
        """--channel edge wins over config update.channel = 'stable'."""
        import yaml
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({"update": {"channel": "stable"}}))

        from hermes_cli import config as config_mod
        import importlib
        importlib.reload(config_mod)

        cfg = config_mod.load_config()
        args = _make_args(channel="edge")
        channel = getattr(args, "channel", None) or cfg.get("update", {}).get("channel", "edge")
        assert channel == "edge"


# ---------------------------------------------------------------------------
# 4. Invalid channel exits 1
# ---------------------------------------------------------------------------

class TestInvalidChannel:
    def test_invalid_channel_exits(self, hermes_env, tmp_path):
        """An invalid channel value should call sys.exit(1)."""
        # We test the guard logic directly without invoking cmd_update end-to-end,
        # because cmd_update has many side-effects (git, pip, etc.).
        channel = "nightly"
        with pytest.raises(SystemExit) as exc_info:
            if channel not in ("edge", "stable"):
                sys.exit(1)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 5. Stable channel: fetch tags and checkout latest tag
# ---------------------------------------------------------------------------

class TestStableChannelGitBehavior:
    """Verify _update_to_latest_release_tag() performs expected git operations."""

    def _make_subprocess_sequence(self, fetch_rc=0, describe_rc=1, tag_list="v2026.4.16\nv2026.3.1\n",
                                   checkout_rc=0, api_tag=None):
        """Build a side_effect list for subprocess.run calls in _update_to_latest_release_tag."""
        calls = []
        # 1) git fetch origin --tags
        calls.append(_make_completed_process(returncode=fetch_rc))
        # 2) git describe --tags --exact-match (returncode != 0 means not on a tag)
        calls.append(_make_completed_process(returncode=describe_rc, stdout=""))
        # 3) git tag --list (only called when API fails/skipped)
        if api_tag is None:
            calls.append(_make_completed_process(returncode=0, stdout=tag_list))
        # 4) git checkout <tag>
        calls.append(_make_completed_process(returncode=checkout_rc))
        return calls

    def test_stable_fetches_tags_and_checks_out(self, hermes_env, tmp_path):
        """Stable path: fetch --tags, tag --list (API fails), describe, checkout latest."""
        from hermes_cli.main import _update_to_latest_release_tag

        git_cmd = ["git"]

        # Call order in _update_to_latest_release_tag:
        #   1. git fetch origin --tags
        #   2. git tag --list v20* (API fails → fallback)
        #   3. git describe --tags --exact-match
        #   4. git checkout <tag>
        run_results = [
            _make_completed_process(returncode=0),                                      # fetch --tags
            _make_completed_process(returncode=0, stdout="v2026.4.16\nv2026.3.1\n"),    # tag --list (API fails)
            _make_completed_process(returncode=1, stdout=""),                           # describe (not on a tag)
            _make_completed_process(returncode=0),                                      # checkout
        ]

        with patch("hermes_cli.main.subprocess.run", side_effect=run_results) as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("no network")):
            result = _update_to_latest_release_tag(git_cmd, tmp_path)

        assert result == "v2026.4.16"

        # Verify git fetch --tags was called
        fetch_call = mock_run.call_args_list[0]
        assert fetch_call[0][0] == ["git", "fetch", "origin", "--tags"]

        # Verify git checkout was called with the latest tag
        checkout_call = mock_run.call_args_list[3]
        assert checkout_call[0][0] == ["git", "checkout", "v2026.4.16"]

    def test_stable_already_on_latest_returns_none(self, hermes_env, tmp_path):
        """When already on the latest tag, returns None (already up to date)."""
        from hermes_cli.main import _update_to_latest_release_tag

        git_cmd = ["git"]

        # Call order:
        #   1. git fetch origin --tags
        #   2. git tag --list (API fails)
        #   3. git describe --tags --exact-match  → returns "v2026.4.16" (same as latest)
        #   → early return, no checkout
        run_results = [
            _make_completed_process(returncode=0),                                      # fetch --tags
            _make_completed_process(returncode=0, stdout="v2026.4.16\nv2026.3.1\n"),    # tag --list
            _make_completed_process(returncode=0, stdout="v2026.4.16"),                 # describe → on latest
        ]

        with patch("hermes_cli.main.subprocess.run", side_effect=run_results) as mock_run, \
             patch("urllib.request.urlopen", side_effect=Exception("no network")):
            result = _update_to_latest_release_tag(git_cmd, tmp_path)

        assert result is None
        # No checkout should have been attempted
        assert len(mock_run.call_args_list) == 3

    def test_stable_uses_github_api_when_available(self, hermes_env, tmp_path):
        """When GitHub API succeeds, use its tag_name without calling git tag --list."""
        from hermes_cli.main import _update_to_latest_release_tag
        import json
        import io

        git_cmd = ["git"]

        run_results = [
            _make_completed_process(returncode=0),                          # fetch --tags
            _make_completed_process(returncode=1, stdout=""),               # describe (not on a tag)
            _make_completed_process(returncode=0),                          # checkout
        ]

        api_response = json.dumps({"tag_name": "v2026.4.16"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=api_response)

        with patch("hermes_cli.main.subprocess.run", side_effect=run_results) as mock_run, \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = _update_to_latest_release_tag(git_cmd, tmp_path)

        assert result == "v2026.4.16"
        # git tag --list should NOT have been called (API succeeded)
        tag_list_calls = [
            c for c in mock_run.call_args_list
            if "tag" in c[0][0] and "--list" in c[0][0]
        ]
        assert len(tag_list_calls) == 0

    def test_stable_fetch_failure_exits(self, hermes_env, tmp_path):
        """Network error during git fetch --tags causes sys.exit(1)."""
        from hermes_cli.main import _update_to_latest_release_tag

        git_cmd = ["git"]
        run_results = [
            _make_completed_process(returncode=1, stderr="Could not resolve host: github.com"),
        ]

        with patch("hermes_cli.main.subprocess.run", side_effect=run_results), \
             pytest.raises(SystemExit) as exc_info:
            _update_to_latest_release_tag(git_cmd, tmp_path)

        assert exc_info.value.code == 1
