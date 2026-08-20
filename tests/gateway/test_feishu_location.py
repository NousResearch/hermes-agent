"""Hermes feishu share_location normalize tests.

8 scenarios + 2 edge cases + 1 regression. Run from the hermes-agent repo root:

    python -m pytest tests/gateway/test_feishu_location.py -v

Or directly (no pytest dependency on this file's structure):

    python tests/gateway/test_feishu_location.py
"""
import json
import unittest

from plugins.platforms.feishu.adapter import normalize_feishu_message


class TestLocationNormalize(unittest.TestCase):
    """Verify normalize_feishu_message() handles feishu location events."""

    # ========== 8 core scenarios ==========
    def test_1_standard_shape(self):
        payload = json.dumps({
            "location_name": "深圳湾万象城",
            "address": "广东省深圳市南山区科苑南路2888号",
            "longitude": 113.943,
            "latitude": 22.527,
            "precision": 50,
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        self.assertEqual(msg.raw_type, "location")
        self.assertIn("深圳湾万象城", msg.text_content)
        self.assertIn("113.943", msg.text_content)
        md = msg.metadata or {}
        self.assertEqual(md.get("longitude"), 113.943)
        self.assertEqual(md.get("latitude"), 22.527)
        self.assertEqual(md.get("location_name"), "深圳湾万象城")

    def test_2_nested_legacy(self):
        """Legacy clients wrap in {"share_location": {...}}."""
        payload = json.dumps({
            "share_location": {
                "location_name": "福田中心公园",
                "longitude": 114.058,
                "latitude": 22.541,
            }
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        md = msg.metadata or {}
        self.assertEqual(md.get("longitude"), 114.058)
        self.assertEqual(md.get("latitude"), 22.541)

    def test_3_name_only(self):
        payload = json.dumps({"location_name": "上海外滩"})
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        self.assertIn("上海外滩", msg.text_content)
        md = msg.metadata or {}
        self.assertEqual(md.get("location_name"), "上海外滩")
        self.assertIsNone(md.get("longitude"))
        self.assertIsNone(md.get("latitude"))

    def test_4_lng_lat_aliases_graceful(self):
        """lng/lat aliases intentionally dropped in v2.4.4 (Feishu SDK only
        emits longitude/latitude). Falls back to FALLBACK_SHARE_LOCATION_TEXT."""
        payload = json.dumps({"lng": 113.323, "lat": 23.106})
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        md = msg.metadata or {}
        # No coordinates extracted — confirms alias-drop decision
        self.assertIsNone(md.get("longitude"))
        self.assertIsNone(md.get("latitude"))
        self.assertIn("[User shared a location]", msg.text_content)

    def test_5_empty_payload(self):
        msg = normalize_feishu_message(message_type="location", raw_content="")
        self.assertEqual(msg.raw_type, "location")
        self.assertTrue(msg.text_content)

    def test_6_address_only(self):
        payload = json.dumps({
            "address": "北京市朝阳区某地",
            "longitude": 116.4,
            "latitude": 39.9,
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        self.assertIn("北京市朝阳区某地", msg.text_content)
        md = msg.metadata or {}
        self.assertEqual(md.get("longitude"), 116.4)
        self.assertEqual(md.get("latitude"), 39.9)

    def test_7_precision_only(self):
        payload = json.dumps({
            "location_name": "北京天安门",
            "longitude": 116.397,
            "latitude": 39.909,
            "precision": 100,
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        md = msg.metadata or {}
        self.assertEqual(md.get("precision"), 100)

    def test_8_legacy_message_type_name(self):
        """Old message_type was 'share_location', not 'location'."""
        payload = json.dumps({
            "location_name": "X",
            "longitude": 1.0,
            "latitude": 2.0,
        })
        msg = normalize_feishu_message(message_type="share_location", raw_content=payload)
        self.assertEqual(msg.raw_type, "location")
        md = msg.metadata or {}
        self.assertEqual(md.get("longitude"), 1.0)

    # ========== 2 edge cases ==========
    def test_9_zero_coordinate_preserved(self):
        """longitude=0 (Null Island) must NOT fallthrough to lng alias."""
        payload = json.dumps({
            "location_name": "Null Island",
            "longitude": 0.0,
            "latitude": 0.0,
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        md = msg.metadata or {}
        self.assertEqual(md.get("longitude"), 0.0)
        self.assertEqual(md.get("latitude"), 0.0)

    def test_10_non_numeric_coordinate(self):
        """Non-numeric lng/lat must not crash — caught by try/except."""
        payload = json.dumps({
            "location_name": "Dirty",
            "longitude": "abc",
            "latitude": None,
        })
        msg = normalize_feishu_message(message_type="location", raw_content=payload)
        md = msg.metadata or {}
        self.assertIsNone(md.get("longitude"))
        self.assertIsNone(md.get("latitude"))
        # Name should still surface in text
        self.assertIn("Dirty", msg.text_content)

    # ========== 1 regression ==========
    def test_regression_other_branches_unaffected(self):
        """Other normalize branches must still work."""
        msg_text = normalize_feishu_message(
            message_type="text", raw_content=json.dumps({"text": "hello"}))
        self.assertEqual(msg_text.raw_type, "text")
        self.assertIn("hello", msg_text.text_content)

        msg_img = normalize_feishu_message(
            message_type="image", raw_content=json.dumps({"image_key": "img_abc"}))
        self.assertEqual(msg_img.raw_type, "image")
        self.assertIn("img_abc", msg_img.image_keys or [])

        msg_share = normalize_feishu_message(
            message_type="share_chat", raw_content=json.dumps({"chat_name": "test"}))
        self.assertEqual(msg_share.raw_type, "share_chat")
        self.assertIn("test", msg_share.text_content)