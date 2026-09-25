"""Only authorized real gateway turns expose source identity to plugin tools."""

import asyncio
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from unittest.mock import patch

from gateway.session_context import (
    clear_session_vars,
    get_authenticated_gateway_source,
    reset_session_vars,
    set_session_vars,
)


class AuthenticatedSourceTests(unittest.TestCase):
    def tearDown(self):
        reset_session_vars()

    def test_authorized_external_turn_exposes_source(self):
        tokens = set_session_vars(
            platform="telegram", chat_id="chat-1", user_id="user-1",
            authorized_external=True,
        )
        try:
            self.assertEqual(
                get_authenticated_gateway_source(),
                ("telegram", "chat-1", "user-1"),
            )
        finally:
            clear_session_vars(tokens)
        self.assertIsNone(get_authenticated_gateway_source())

    def test_internal_unbound_and_environment_only_identity_are_rejected(self):
        self.assertIsNone(get_authenticated_gateway_source())
        tokens = set_session_vars(
            platform="telegram", chat_id="chat-1", user_id="user-1",
            authorized_external=False,
        )
        try:
            self.assertIsNone(get_authenticated_gateway_source())
        finally:
            clear_session_vars(tokens)
        with patch.dict(os.environ, {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "chat-1",
            "HERMES_SESSION_USER_ID": "user-1",
        }):
            self.assertIsNone(get_authenticated_gateway_source())

    def test_new_message_resets_inherited_authorization_before_binding(self):
        set_session_vars(
            platform="telegram", chat_id="chat-1", user_id="user-1",
            authorized_external=True,
        )

        async def child():
            reset_session_vars()
            return get_authenticated_gateway_source()

        self.assertIsNone(asyncio.run(child()))

    def test_authorization_follows_explicit_context_copy_to_tool_thread(self):
        set_session_vars(
            platform="telegram", chat_id="chat-1", user_id="user-1",
            authorized_external=True,
        )
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertIsNone(pool.submit(get_authenticated_gateway_source).result())
            self.assertEqual(
                pool.submit(context.run, get_authenticated_gateway_source).result(),
                ("telegram", "chat-1", "user-1"),
            )


if __name__ == "__main__":
    unittest.main()
