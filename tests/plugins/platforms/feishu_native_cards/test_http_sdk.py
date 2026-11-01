"""真实 lark SDK 对本机 HTTP 替身的序列化/响应闭环，不访问飞书。"""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from tests.plugins.platforms.feishu_native_cards.support import make_adapter, add_turn
import itertools


class HttpSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_create_reference_content_and_final_on_loopback(self):
        operations = []

        class Handler(BaseHTTPRequestHandler):
            def process(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                operations.append((self.command, self.path, body))
                if self.path == "/open-apis/cardkit/v1/cards":
                    data = {"card_id": "local_card"}
                elif self.path.startswith("/open-apis/im/v1/messages"):
                    data = {"message_id": "local_im"}
                elif self.path.startswith("/open-apis/cardkit/v1/cards/local_card"):
                    data = {}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = json.dumps({"code": 0, "data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_POST = process
            do_PUT = process
            do_PATCH = process

            def log_message(self, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        from lark_oapi import Client, RequestOption
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, _hub, _ = make_adapter()
        add_turn(adapter)
        client = (
            Client.builder()
            .app_id("unit-app")
            .app_secret("unit-secret")
            .domain(f"http://127.0.0.1:{server.server_port}")
            .enable_set_token(True)
            .timeout(3)
            .build()
        )
        option = RequestOption.builder().tenant_access_token("unit-token").build()
        for obj, method in [
            (client.cardkit.v1.card, "create"),
            (client.cardkit.v1.card, "update"),
            (client.cardkit.v1.card, "settings"),
            (client.cardkit.v1.card_element, "content"),
            (client.im.v1.message, "create"),
            (client.im.v1.message, "reply"),
        ]:
            original = getattr(obj, method)
            setattr(obj, method, lambda request, fn=original: fn(request, option))
        adapter._client = client
        adapter._run_blocking = FeishuAdapter._run_blocking.__get__(adapter)
        adapter._feishu_send_with_retry = type(adapter)._feishu_send_with_retry.__get__(adapter)
        try:
            result = await adapter.send_draft(
                "chat_a", 1, "真实 SDK 流式", {"reply_to_message_id": "user_a"}
            )
            self.assertTrue(result.success, result.error)
            final = await adapter.send("chat_a", "最终答案", reply_to="user_a")
            self.assertTrue(final.success, final.error)
            self.assertEqual(final.message_id, "local_im")
            self.assertEqual(sum("/im/" in path for _, path, _ in operations), 1)
            changes = [
                body
                for _, path, body in operations
                if path.startswith("/open-apis/cardkit/v1/cards/local_card")
            ]
            self.assertTrue(
                all(b["sequence"] < a["sequence"] for b, a in itertools.pairwise(changes))
            )
            self.assertFalse(json.loads(changes[-1]["card"]["data"])["config"]["streaming_mode"])
        finally:
            adapter._get_sdk_executor().shutdown(wait=True)
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            server_thread.join()
