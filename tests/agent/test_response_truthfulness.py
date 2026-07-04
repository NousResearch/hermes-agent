import unittest

from agent.response_truthfulness import (
    has_file_delivery_payload,
    looks_like_unbacked_file_delivery_claim,
)


class TestResponseTruthfulness(unittest.TestCase):
    def test_unbacked_file_delivery_claim_detects_short_attachment_claims(self):
        self.assertTrue(looks_like_unbacked_file_delivery_claim("文件已生成，请查收。"))
        self.assertTrue(looks_like_unbacked_file_delivery_claim("I've attached the Word document below."))
        self.assertTrue(looks_like_unbacked_file_delivery_claim("我再发一次，应该会以附件形式出现。"))

    def test_unbacked_file_delivery_claim_allows_real_delivery_evidence(self):
        self.assertFalse(looks_like_unbacked_file_delivery_claim("MEDIA:/tmp/report.docx"))
        self.assertFalse(looks_like_unbacked_file_delivery_claim("文件路径：/tmp/report.docx"))
        self.assertFalse(looks_like_unbacked_file_delivery_claim("下载链接：https://example.com/report.pdf"))
        self.assertFalse(looks_like_unbacked_file_delivery_claim("```text\ninline file content\n```"))

    def test_unbacked_file_delivery_claim_allows_honest_failure(self):
        self.assertFalse(looks_like_unbacked_file_delivery_claim("文件生成失败：没有写入权限。"))
        self.assertFalse(looks_like_unbacked_file_delivery_claim("I could not upload the attachment."))

    def test_has_file_delivery_payload_recognizes_common_payloads(self):
        self.assertTrue(has_file_delivery_payload("请打开 /tmp/a/b/report.pdf"))
        self.assertTrue(has_file_delivery_payload("[report](https://example.com/report.pdf)"))
        self.assertFalse(has_file_delivery_payload("文件已生成，请查收。"))


if __name__ == "__main__":
    unittest.main()
