import unittest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.LOCAL)
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", content))
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, **kwargs):
        self.calls.append(("image", image_path, caption, reply_to, kwargs))
        return SendResult(success=True, message_id="image")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, **kwargs):
        self.calls.append(("document", file_path, caption, file_name, reply_to, kwargs))
        return SendResult(success=True, message_id="document")


class TestBaseSendFileRouter(unittest.IsolatedAsyncioTestCase):
    async def test_send_file_routes_images_to_send_image_file(self):
        adapter = DummyAdapter()
        result = await adapter.send_file("chat", "/tmp/poster.png", caption="cap", reply_to="om_parent", metadata={"thread_id":"om_root"})
        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "image")
        self.assertEqual(adapter.calls[0][0], "image")
        self.assertEqual(adapter.calls[0][1], "/tmp/poster.png")
        self.assertEqual(adapter.calls[0][2], "cap")
        self.assertEqual(adapter.calls[0][3], "om_parent")

    async def test_send_file_routes_unknown_files_to_document(self):
        adapter = DummyAdapter()
        result = await adapter.send_file("chat", "/tmp/report.xlsx", file_name="report.xlsx")
        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "document")
        self.assertEqual(adapter.calls[0][0], "document")
        self.assertEqual(adapter.calls[0][1], "/tmp/report.xlsx")
        self.assertEqual(adapter.calls[0][3], "report.xlsx")
