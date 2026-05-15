import unittest

from ascii_pipeline.presets import PRESETS


class PresetTests(unittest.TestCase):
    def test_canonical_presets_exist(self):
        self.assertTrue({"stroke-clarity", "d30-dense", "braille-detail", "eikon-motion"}.issubset(PRESETS))

    def test_presets_share_contract_fields(self):
        for preset in PRESETS.values():
            self.assertEqual(len(preset.target), 2)
            self.assertIsInstance(preset.preprocess, tuple)
            self.assertGreater(preset.preview_font_size, 0)
            self.assertIsInstance(preset.quality_thresholds, dict)
            self.assertTrue(preset.notes)

    def test_published_still_image_presets_do_not_depend_on_chafa(self):
        for name in ("stroke-clarity", "d30-dense", "braille-detail"):
            self.assertNotIn("chafa", PRESETS[name].backend)


if __name__ == "__main__":
    unittest.main()
