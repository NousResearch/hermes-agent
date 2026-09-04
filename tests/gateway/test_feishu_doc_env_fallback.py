"""Tests for the shared Feishu tool client factory.

Covers the get_client() env-credential fallback used by feishu_doc_* / feishu_drive_*
so those tools keep working outside a Feishu comment-event context (DM
conversations, CLI, agent-initiated sessions) by building a lark client from
FEISHU_APP_ID / FEISHU_APP_SECRET.
"""

import importlib
import os
import sys
import threading
import unittest
from unittest import mock


def _reset_factory():
    """Reload the shared factory so each test starts with clean thread-local state."""
    for name in ("tools.feishu_client_factory",):
        sys.modules.pop(name, None)
    return importlib.import_module("tools.feishu_client_factory")


class _EnvFallbackCase(unittest.TestCase):
    TOOL_MODULES = ("tools.feishu_doc_tool", "tools.feishu_drive_tool")

    def _make_fake_lark(self):
        fake_client = object()
        builder = mock.MagicMock()
        # lark's Client.builder() is a fluent chain: app_id().app_secret()
        # .log_level().build() — make every link return the builder itself.
        builder.app_id.return_value = builder
        builder.app_secret.return_value = builder
        builder.log_level.return_value = builder
        builder.build.return_value = fake_client
        client_cls = mock.MagicMock()
        client_cls.builder.return_value = builder
        fake_lark = mock.MagicMock()
        fake_lark.Client = client_cls
        fake_lark.LogLevel.WARNING = "warning"
        return fake_lark, client_cls, builder, fake_client

    def _clear_feishu_env(self):
        for var in ("FEISHU_APP_ID", "FEISHU_APP_SECRET"):
            os.environ.pop(var, None)

    def _set_feishu_env(self, app_id="cli_test_app_id", secret="test_secret"):
        os.environ["FEISHU_APP_ID"] = app_id
        os.environ["FEISHU_APP_SECRET"] = secret

    def setUp(self):
        self._clear_feishu_env()

    def tearDown(self):
        self._clear_feishu_env()


class TestGetClientEnvFallback(_EnvFallbackCase):
    def test_returns_none_without_injection_and_env(self):
        factory = _reset_factory()
        self.assertIsNone(factory.get_client())

    def test_builds_client_from_env_credentials(self):
        factory = _reset_factory()
        fake_lark, client_cls, builder, fake_client = self._make_fake_lark()
        self._set_feishu_env()
        try:
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                got = factory.get_client()
        finally:
            self._clear_feishu_env()
        self.assertIs(got, fake_client)
        builder.app_id.assert_called_once_with("cli_test_app_id")
        builder.app_secret.assert_called_once_with("test_secret")

    def test_builds_client_once_and_caches_per_thread(self):
        factory = _reset_factory()
        fake_lark, _, builder, fake_client = self._make_fake_lark()
        self._set_feishu_env()
        try:
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                first = factory.get_client()
                second = factory.get_client()
        finally:
            self._clear_feishu_env()
        self.assertIs(first, fake_client)
        self.assertIs(second, fake_client)
        self.assertEqual(builder.build.call_count, 1)

    def test_injected_client_takes_priority_over_env(self):
        factory = _reset_factory()
        injected = object()
        factory.set_client(injected)
        fake_lark, _, _, _ = self._make_fake_lark()
        self._set_feishu_env()
        try:
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                got = factory.get_client()
        finally:
            self._clear_feishu_env()
        self.assertIs(got, injected)

    def test_missing_one_env_var_returns_none(self):
        for keep in ("FEISHU_APP_ID", "FEISHU_APP_SECRET"):
            with self.subTest(keep=keep):
                factory = _reset_factory()
                os.environ[keep] = "partial"
                try:
                    self.assertIsNone(factory.get_client())
                finally:
                    os.environ.pop(keep, None)

    def test_cache_is_keyed_on_credentials(self):
        """A credential swap mid-process rebuilds instead of reusing a stale client."""
        factory = _reset_factory()
        fake_lark, _, builder, first_client = self._make_fake_lark()
        self._set_feishu_env("app-a", "secret-a")
        try:
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                self.assertIs(factory.get_client(), first_client)
                # Same credentials -> cache hit, no rebuild.
                self.assertIs(factory.get_client(), first_client)
                self.assertEqual(builder.build.call_count, 1)
                # Different credentials -> rebuild.
                second_client = object()
                builder.build.return_value = second_client
                self._set_feishu_env("app-b", "secret-b")
                self.assertIs(factory.get_client(), second_client)
                self.assertEqual(builder.build.call_count, 2)
        finally:
            self._clear_feishu_env()

    def test_set_client_none_clears_env_built_client(self):
        factory = _reset_factory()
        fake_lark, _, builder, fake_client = self._make_fake_lark()
        self._set_feishu_env()
        try:
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                self.assertIs(factory.get_client(), fake_client)
                factory.set_client(None)
                rebuilt = object()
                builder.build.return_value = rebuilt
                self.assertIs(factory.get_client(), rebuilt)
        finally:
            self._clear_feishu_env()

    def test_env_fallback_skipped_in_delegated_child_context(self):
        """delegate_task children must not gain tenant-wide doc/drive access."""
        factory = _reset_factory()
        fake_lark, _, builder, _ = self._make_fake_lark()
        fake_ctx = mock.MagicMock()
        fake_ctx.is_delegated_child_context.return_value = True
        self._set_feishu_env()
        try:
            with mock.patch.dict(
                sys.modules, {"lark_oapi": fake_lark, "agent.delegation_context": fake_ctx}
            ):
                self.assertIsNone(factory.get_client())
        finally:
            self._clear_feishu_env()
        builder.build.assert_not_called()

    def test_env_fallback_engages_outside_delegated_child(self):
        factory = _reset_factory()
        fake_lark, _, builder, fake_client = self._make_fake_lark()
        fake_ctx = mock.MagicMock()
        fake_ctx.is_delegated_child_context.return_value = False
        self._set_feishu_env()
        try:
            with mock.patch.dict(
                sys.modules, {"lark_oapi": fake_lark, "agent.delegation_context": fake_ctx}
            ):
                self.assertIs(factory.get_client(), fake_client)
        finally:
            self._clear_feishu_env()

    def test_fallback_survives_missing_delegation_module(self):
        """An import failure must not turn into a tool failure."""
        factory = _reset_factory()
        fake_lark, _, _, fake_client = self._make_fake_lark()
        self._set_feishu_env()
        try:
            real_get = sys.modules.get("agent.delegation_context")
            sys.modules["agent.delegation_context"] = None
            with mock.patch.dict(sys.modules, {"lark_oapi": fake_lark}):
                self.assertIs(factory.get_client(), fake_client)
        finally:
            if real_get is not None:
                sys.modules["agent.delegation_context"] = real_get
            else:
                sys.modules.pop("agent.delegation_context", None)
            self._clear_feishu_env()


class TestToolsShareFactory(_EnvFallbackCase):
    def test_both_tools_reexport_shared_factory(self):
        """doc/drive tools must delegate to one implementation, not duplicate it."""
        for name in self.TOOL_MODULES:
            with self.subTest(module=name):
                mod = importlib.import_module(name)
                factory = importlib.import_module("tools.feishu_client_factory")
                self.assertIs(mod.get_client, factory.get_client)
                self.assertIs(mod.set_client, factory.set_client)

    def test_thread_isolation(self):
        factory = _reset_factory()
        results = {}

        def worker(marker):
            factory.set_client(marker)
            results[marker] = factory.get_client()

        t1 = threading.Thread(target=worker, args=("thread-one",))
        t2 = threading.Thread(target=worker, args=("thread-two",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(results, {"thread-one": "thread-one", "thread-two": "thread-two"})


if __name__ == "__main__":
    unittest.main()
