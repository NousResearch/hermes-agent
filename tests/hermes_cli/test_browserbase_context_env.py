"""Tests for the BROWSERBASE_CONTEXT_ID optional env var registration.

BROWSERBASE_CONTEXT_ID points cloud browser sessions at a persistent
Browserbase Context so cookies and login state survive across sessions.
It is an identifier, not a credential — unlike BROWSERBASE_API_KEY it
must never be masked in the setup UI.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Point HERMES_HOME at an empty tmp dir and unset the Browserbase vars.

    get_env_value() reads os.environ and the profile's ~/.hermes/.env, so
    both must be isolated or the developer's real home pollutes the
    get_missing_env_vars() assertions.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSERBASE_CONTEXT_ID",
    ):
        monkeypatch.delenv(key, raising=False)


class TestBrowserbaseContextRegistry:
    def test_optional_env_vars_include_context_id(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "BROWSERBASE_CONTEXT_ID" in OPTIONAL_ENV_VARS
        entry = OPTIONAL_ENV_VARS["BROWSERBASE_CONTEXT_ID"]
        assert entry["url"] == "https://browserbase.com/"
        assert entry["description"]

    def test_context_id_is_not_masked_but_api_key_is(self):
        """The API key is a secret; the context ID is an identifier."""
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert OPTIONAL_ENV_VARS["BROWSERBASE_CONTEXT_ID"]["password"] is False
        assert OPTIONAL_ENV_VARS["BROWSERBASE_API_KEY"]["password"] is True

    def test_context_id_category_matches_browserbase_siblings(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert (
            OPTIONAL_ENV_VARS["BROWSERBASE_CONTEXT_ID"]["category"]
            == OPTIONAL_ENV_VARS["BROWSERBASE_API_KEY"]["category"]
            == OPTIONAL_ENV_VARS["BROWSERBASE_PROJECT_ID"]["category"]
            == "tool"
        )


class TestBrowserbaseContextMissingEnv:
    def test_surfaced_as_optional_when_unset(self):
        from hermes_cli.config import get_missing_env_vars

        missing = {v["name"]: v for v in get_missing_env_vars(required_only=False)}
        assert "BROWSERBASE_CONTEXT_ID" in missing
        assert missing["BROWSERBASE_CONTEXT_ID"]["is_required"] is False

    def test_not_surfaced_when_required_only(self):
        from hermes_cli.config import get_missing_env_vars

        missing_names = {v["name"] for v in get_missing_env_vars(required_only=True)}
        assert "BROWSERBASE_CONTEXT_ID" not in missing_names
