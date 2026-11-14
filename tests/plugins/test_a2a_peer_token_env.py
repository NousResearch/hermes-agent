"""Outbound peer token from an environment variable (``token_env``).

The inbound side already reads ``A2A_PEER_TOKENS`` from the environment, so a
peer gets a named identity that drives audit, rate limiting and the allow-list.
The outbound side had no equivalent: ``auth.token`` was taken literally from
config.yaml. Installs that rewrite or back up that file could not put a secret
there, so they ran with no per-peer token at all and every caller collapsed to
``ip:<addr>``.

``token_env`` names the variable instead. Inline ``token`` keeps precedence so
nothing existing changes.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import plugins.platforms.a2a.tools as a2at


class TestA2APeerTokenEnv(unittest.TestCase):
    def test_inline_token_still_wins(self):
        self.assertEqual(
            a2at._auth_header({"type": "bearer", "token": "sk-inline"}),
            {"Authorization": "Bearer sk-inline"},
        )

    def test_token_env_is_read_from_the_environment(self):
        with patch.dict(os.environ, {"A2A_TOKEN_RESEARCHER": "sk-from-env"}):
            self.assertEqual(
                a2at._auth_header({"type": "bearer", "token_env": "A2A_TOKEN_RESEARCHER"}),
                {"Authorization": "Bearer sk-from-env"},
            )

    def test_inline_token_takes_precedence_over_token_env(self):
        with patch.dict(os.environ, {"A2A_TOKEN_RESEARCHER": "sk-from-env"}):
            self.assertEqual(
                a2at._auth_header(
                    {"type": "bearer", "token": "sk-inline", "token_env": "A2A_TOKEN_RESEARCHER"}
                ),
                {"Authorization": "Bearer sk-inline"},
            )

    def test_unset_variable_sends_no_header(self):
        # Fail closed: an empty Authorization header would look authenticated
        # to nothing and hide the misconfiguration behind a 401 at the peer.
        os.environ.pop("A2A_TOKEN_MISSING", None)
        self.assertEqual(
            a2at._auth_header({"type": "bearer", "token_env": "A2A_TOKEN_MISSING"}), {}
        )

    def test_no_auth_and_non_bearer_are_unchanged(self):
        self.assertEqual(a2at._auth_header({}), {})
        self.assertEqual(a2at._auth_header({"type": "basic", "token": "x"}), {})
        with patch.dict(os.environ, {"A2A_TOKEN_RESEARCHER": "sk-from-env"}):
            self.assertEqual(
                a2at._auth_header({"type": "basic", "token_env": "A2A_TOKEN_RESEARCHER"}), {}
            )


if __name__ == "__main__":
    unittest.main()
