"""_auth_header() credential resolution for configured A2A peers."""

import pytest

from plugins.platforms.a2a.tools import _auth_header


class TestAuthHeader:
    def test_inline_token(self):
        assert _auth_header({"type": "bearer", "token": "tok"}) == {"Authorization": "Bearer tok"}

    def test_token_env_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("A2A_TEST_TOKEN", "from-env")
        assert _auth_header({"type": "bearer", "token_env": "A2A_TEST_TOKEN"}) == {
            "Authorization": "Bearer from-env"
        }

    def test_token_env_is_stripped(self, monkeypatch):
        monkeypatch.setenv("A2A_TEST_TOKEN", "  padded  \n")
        assert _auth_header({"type": "bearer", "token_env": "A2A_TEST_TOKEN"}) == {
            "Authorization": "Bearer padded"
        }

    def test_inline_token_wins_over_token_env(self, monkeypatch):
        monkeypatch.setenv("A2A_TEST_TOKEN", "from-env")
        assert _auth_header({"type": "bearer", "token": "inline", "token_env": "A2A_TEST_TOKEN"}) == {
            "Authorization": "Bearer inline"
        }

    @pytest.mark.parametrize(
        "auth",
        [
            {},
            None,
            {"type": "basic", "token": "tok"},
            {"type": "bearer"},
            {"type": "bearer", "token_env": "A2A_UNSET_VAR"},
            {"type": "bearer", "token_env": ""},
        ],
    )
    def test_no_header_without_a_usable_bearer_token(self, auth, monkeypatch):
        monkeypatch.delenv("A2A_UNSET_VAR", raising=False)
        assert _auth_header(auth) == {}
