"""Attachment handling in the email adapter: what the agent actually receives.

Three behaviours, each from an observed failure:

* A mail forwarded AS AN ATTACHMENT (``message/rfc822``) used to arrive as the forwarder's covering
  line plus an opaque file: ``_first_body_part`` reads only the outer message, so the original's own
  text was silently dropped and the agent answered "what does this invoice want?" from "see below".
* A picture reached the model's vision only for the five extensions the API accepts inline; a
  ``.heic`` (default iPhone camera format), ``.bmp`` or ``.tif`` went to the document cache and was
  never looked at.
* An undecodable payload must still degrade to a document rather than being dropped.

No network and no live mailbox: the parts are built in-process and the caching helpers write into a
temp ``HERMES_HOME``.
"""

import os
import tempfile
import unittest
from contextlib import contextmanager
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import patch

MODULE = "plugins.platforms.email.adapter"

try:  # the base install ships pillow-heif; a stripped environment may not
    from pillow_heif import register_heif_opener as _register_heif
    _HEIF_CODEC = True
except ImportError:
    _HEIF_CODEC = False


@contextmanager
def _temp_home():
    """Temp HERMES_HOME so the attachment caches never touch the operator's own."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
            yield Path(tmp)


def _loaded():
    import importlib
    return importlib.import_module(MODULE)


def _forward(inner, cover="Forwarding this to you - what does it want?\n\nThanks,\nNick\n"):
    """A client-style forward: the original carried as a ``message/rfc822`` attachment."""
    outer = MIMEMultipart()
    outer["From"] = "operator@test.com"
    outer["To"] = "hermes@test.com"
    outer["Subject"] = "Fwd: Vendor invoice"
    outer.attach(MIMEText(cover, "plain", "utf-8"))
    part = MIMEMessage(inner)
    part.add_header("Content-Disposition", "attachment", filename="Vendor invoice.eml")
    outer.attach(part)
    return outer


class TestForwardedMessageText(unittest.TestCase):
    """The original's text must reach the body, not just its attachments."""

    def test_forward_as_attachment_inlines_the_original_text(self):
        inner = MIMEText("Please find your invoice attached.\n\nAccount reference INNER-CODE-8823\n", "plain", "utf-8")
        inner["From"] = "Billing <billing@vendor.test>"
        inner["Subject"] = "Vendor invoice 8823"

        body = _loaded()._compose_body(_forward(inner))

        self.assertIn("INNER-CODE-8823", body)
        self.assertIn("Forwarding this to you", body, "the covering line must survive")
        self.assertIn("billing@vendor.test", body, "the original's sender must be visible")
        self.assertIn("Vendor invoice 8823", body, "the original's subject must be visible")

    def test_original_attachments_are_still_extracted(self):
        inner = MIMEMultipart()
        inner["Subject"] = "Vendor invoice 8823"
        inner.attach(MIMEText("invoice attached", "plain", "utf-8"))
        pdf = MIMEApplication(b"%PDF-1.4\n% fake invoice\n", _subtype="pdf")
        pdf.add_header("Content-Disposition", "attachment", filename="invoice-8823.pdf")
        inner.attach(pdf)

        with _temp_home():
            adapter = _loaded()
            attachments = adapter._extract_attachments(_forward(inner))
            cached = {att["filename"]: Path(att["path"]).is_file() for att in attachments}

        self.assertIn("invoice-8823.pdf", cached)
        self.assertEqual({att["type"] for att in attachments}, {"document"})
        self.assertTrue(all(cached.values()), "every attachment must be on disk while the home lives")

    def test_a_plain_message_body_is_unchanged(self):
        msg = MIMEText("just a normal mail", "plain", "utf-8")
        self.assertEqual(_loaded()._compose_body(msg).strip(), "just a normal mail")


class TestPictureFormats(unittest.TestCase):
    """Pictures a camera or scanner produces must reach the model as pictures."""

    @staticmethod
    def _png_bytes():
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (40, 24), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def _attach(self, data, filename, subtype):
        msg = MIMEMultipart()
        part = MIMEImage(data, _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)
        return msg

    def test_a_bmp_picture_is_cached_as_an_image_png(self):
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (40, 24), "white").save(buffer, format="BMP")

        with _temp_home():
            attachments = _loaded()._extract_attachments(self._attach(buffer.getvalue(), "scan.bmp", "bmp"))

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["type"], "image")
        self.assertEqual(attachments[0]["media_type"], "image/png")
        self.assertTrue(attachments[0]["path"].endswith(".png"))
        self.assertTrue(attachments[0]["filename"].endswith(".png"))

    def test_a_native_png_is_cached_untouched(self):
        with _temp_home():
            attachments = _loaded()._extract_attachments(self._attach(self._png_bytes(), "photo.png", "png"))

        self.assertEqual(attachments[0]["type"], "image")
        self.assertTrue(attachments[0]["path"].endswith(".png"))

    def test_an_undecodable_picture_degrades_to_a_document(self):
        """A picture with no available decoder must stay usable as a file, never be dropped."""
        with _temp_home():
            attachments = _loaded()._extract_attachments(self._attach(b"\x00\x00\x00 ftypheic broken", "IMG_1.heic", "heic"))
            on_disk = Path(attachments[0]["path"]).is_file() if attachments else False

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["type"], "document")
        self.assertTrue(on_disk)

    @unittest.skipUnless(_HEIF_CODEC, "pillow-heif not installed in this environment")
    def test_an_iphone_heic_picture_is_cached_as_an_image_png(self):
        """The default iPhone camera format must reach vision, not the document cache."""
        import io
        from PIL import Image
        from pillow_heif import register_heif_opener
        register_heif_opener()  # so Pillow can WRITE the sample the adapter must read
        buffer = io.BytesIO()
        Image.new("RGB", (48, 32), "white").save(buffer, format="HEIF")

        with _temp_home():
            attachments = _loaded()._extract_attachments(
                self._attach(buffer.getvalue(), "IMG_0002.HEIC", "heic"))

        self.assertEqual(attachments[0]["type"], "image")
        self.assertEqual(attachments[0]["media_type"], "image/png")
        self.assertTrue(attachments[0]["path"].endswith(".png"))

    def test_skip_attachments_still_caches_nothing(self):
        with _temp_home():
            attachments = _loaded()._extract_attachments(self._attach(self._png_bytes(), "photo.png", "png"),
                                                         skip_attachments=True)
        self.assertEqual(attachments, [])


if __name__ == "__main__":
    unittest.main()
