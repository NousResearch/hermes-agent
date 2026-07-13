"""Profile isolation for Feishu SDK-thread callbacks."""

import asyncio
import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent.secret_scope import (
    get_secret,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.platforms.feishu.adapter import FeishuAdapter


class TestFeishuCallbackProfileScope(unittest.TestCase):
    def test_submit_restores_adapter_profile_home_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outer_home = root / "outer"
            profile_home = root / "worker"
            outer_home.mkdir()
            profile_home.mkdir()
            (profile_home / ".env").write_text(
                "PROFILE_CALLBACK_SECRET=worker-secret\n",
                encoding="utf-8",
            )

            adapter = FeishuAdapter.__new__(FeishuAdapter)
            adapter._profile_home = profile_home
            adapter._log_background_failure = Mock()
            result_future = concurrent.futures.Future()
            loop = asyncio.new_event_loop()
            loop_ready = threading.Event()

            def run_loop():
                asyncio.set_event_loop(loop)
                loop_ready.set()
                loop.run_forever()

            loop_thread = threading.Thread(target=run_loop, daemon=True)
            loop_thread.start()
            self.assertTrue(loop_ready.wait(timeout=5))

            async def probe_context():
                result_future.set_result(
                    (
                        get_hermes_home(),
                        get_secret("PROFILE_CALLBACK_SECRET"),
                    )
                )

            home_token = set_hermes_home_override(outer_home)
            secret_token = set_secret_scope(
                {"PROFILE_CALLBACK_SECRET": "outer-secret"}
            )
            try:
                submitted = adapter._submit_on_loop(loop, probe_context())

                self.assertTrue(submitted)
                self.assertEqual(
                    result_future.result(timeout=5),
                    (profile_home, "worker-secret"),
                )
                self.assertEqual(get_hermes_home(), outer_home)
                self.assertEqual(
                    get_secret("PROFILE_CALLBACK_SECRET"), "outer-secret"
                )
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
                loop.call_soon_threadsafe(loop.stop)
                loop_thread.join(timeout=5)
                loop.close()


if __name__ == "__main__":
    unittest.main()
