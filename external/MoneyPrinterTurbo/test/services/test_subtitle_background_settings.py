import json
from pathlib import Path
import unittest

import numpy as np

from app.models.schema import VideoParams
from app.services import video


class TestSubtitleBackgroundSettings(unittest.TestCase):
    def test_all_locales_include_subtitle_background_labels(self):
        """
        WebUI 新增字幕背景开关和颜色选择器后，所有已有语言都必须包含对应
        翻译 key，避免某些语言界面直接显示英文内部 key。
        """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        required_keys = {
            "Enable Subtitle Background",
            "Subtitle Background Color",
            "No Voice",
        }

        for locale_file in i18n_dir.glob("*.json"):
            with self.subTest(locale=locale_file.name):
                data = json.loads(locale_file.read_text(encoding="utf-8"))
                translations = data.get("Translation", {})
                missing_keys = required_keys - translations.keys()

                self.assertEqual(missing_keys, set())

    def test_video_params_accepts_disabled_and_colored_subtitle_background(self):
        """
        UI 会根据开关向后端传递 False 或颜色字符串。这里验证 schema 仍然
        接受这两种值，避免后续依赖或类型调整破坏 WebUI 与合成逻辑的契约。
        """
        base_params = {
            "video_subject": "subtitle background smoke",
        }

        disabled_params = VideoParams(
            **base_params,
            text_background_color=False,
        )
        colored_params = VideoParams(
            **base_params,
            text_background_color="#123456",
        )

        self.assertFalse(disabled_params.text_background_color)
        self.assertEqual(colored_params.text_background_color, "#123456")

    def test_visible_text_position_centers_actual_mask_bounds(self):
        """
        TextClip 的画布会包含字体行高和 baseline 空白，直接居中画布会让
        字幕在背景里看起来偏下。这里用一个假 mask 模拟“可见文字像素
        在画布下半部分”的情况，验证 helper 会按真实可见区域重新计算 y。
        """

        class FakeMask:
            def get_frame(self, _):
                mask = np.zeros((46, 100), dtype=float)
                mask[12:46, 10:90] = 1.0
                return mask

        class FakeTextClip:
            w = 100
            h = 46
            mask = FakeMask()

        x, y = video._get_visible_center_position(
            FakeTextClip(), container_width=100, container_height=93
        )

        self.assertEqual(x, 0)
        # 可见像素高度为 34px，放在 93px 容器中应上下各约 29px；
        # 因为 mask 顶部从 12px 开始，所以 TextClip 本身需要向上移动到 18px。
        self.assertEqual(y, 18)

    def test_wrap_text_keeps_closing_punctuation_with_text(self):
        """
        中文长句按字符换行时，句号等闭合标点不能独占一行，否则字幕背景
        会被一个单独的小点撑高。这里复现大字号中文长句的边界情况。
        """
        font_path = (
            Path(__file__).parent.parent.parent
            / "resource"
            / "fonts"
            / "MicrosoftYaHeiBold.ttc"
        )

        wrapped_text, _ = video.wrap_text(
            "如果你调整字号，中文笔画也不能被黑色背景遮挡。",
            max_width=1642,
            font=str(font_path),
            fontsize=72,
        )

        self.assertNotIn("\n。", wrapped_text)
        self.assertIn("挡。", wrapped_text)
