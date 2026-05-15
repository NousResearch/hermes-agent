import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from ascii_pipeline.cli import main


class CliTests(unittest.TestCase):
    def test_presets_command_prints_known_preset(self):
        buf = io.StringIO()
        with patch("sys.argv", ["ascii-pipeline", "presets"]), contextlib.redirect_stdout(buf):
            main()
        self.assertIn("stroke-clarity", buf.getvalue())

    def test_render_preview_command_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sample.txt"
            out = Path(tmp) / "preview.png"
            src.write_text("@@@\n@ @\n@@@\n", encoding="utf-8")
            buf = io.StringIO()
            with patch("sys.argv", ["ascii-pipeline", "render-preview", "--input", str(src), "--out", str(out)]), contextlib.redirect_stdout(buf):
                main()
            self.assertTrue(out.exists())
            self.assertIn(str(out), buf.getvalue())

    def test_render_image_command_writes_text_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            out = Path(tmp) / "sample.txt"
            preview = Path(tmp) / "sample-preview.png"
            diagnostics = Path(tmp) / "sample-diagnostics.json"

            image = Image.new("RGB", (96, 96), (12, 12, 18))
            draw = ImageDraw.Draw(image)
            draw.rectangle((18, 18, 78, 78), fill=(240, 240, 240))
            draw.rectangle((32, 32, 64, 64), fill=(24, 24, 24))
            image.save(image_path)

            buf = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "ascii-pipeline",
                    "render-image",
                    "--input",
                    str(image_path),
                    "--out",
                    str(out),
                    "--preset",
                    "stroke-clarity",
                    "--preview-out",
                    str(preview),
                    "--diagnostics-out",
                    str(diagnostics),
                ],
            ), contextlib.redirect_stdout(buf):
                main()

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["preset"], "stroke-clarity")
            self.assertTrue(out.exists())
            self.assertTrue(preview.exists())
            self.assertTrue(diagnostics.exists())
            self.assertIn("@", out.read_text(encoding="utf-8"))

    def test_render_image_scale_flag_produces_correct_grid(self):
        """Verify --scale N produces the expected grid dimensions in output and diagnostics."""
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            out = Path(tmp) / "scaled.txt"
            diagnostics = Path(tmp) / "diag.json"

            # Create a simple grayscale test image: 100×100 pixels, vertical gradient
            image = Image.new("L", (100, 100))
            for y in range(100):
                for x in range(100):
                    image.putpixel((x, y), int(255 * y / 100))
            image.save(image_path)

            buf = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "ascii-pipeline",
                    "render-image",
                    "--input",
                    str(image_path),
                    "--out",
                    str(out),
                    "--preset",
                    "stroke-clarity",
                    "--diagnostics-out",
                    str(diagnostics),
                    "--scale",
                    "8",  # 48*8=384 wide, 24*8=192 tall
                ],
            ), contextlib.redirect_stdout(buf):
                main()

            # Check output dimensions: should be exactly 384×192
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 192)  # height
            self.assertTrue(all(len(line) == 384 for line in lines))  # width

            # Diagnostics should report expected dimensions
            diag = json.loads(diagnostics.read_text(encoding="utf-8"))
            self.assertEqual(diag["dimensions"]["expected"], [384, 192])
            self.assertTrue(diag["dimensions"]["all_match_expected"])

    def test_build_eikon_from_video_creates_valid_eikon(self):
        """End-to-end smoke test for build-eikon-from-video command."""
        video = Path('/Users/johann/Desktop/hooded-ouroboros-collection/owl_ascii.MP4')
        if not video.exists():
            self.skipTest("Reference video not available")

        with tempfile.TemporaryDirectory() as tmp:
            out_eikon = Path(tmp) / "owl-animated.eikon"
            buf = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "ascii-pipeline",
                    "build-eikon-from-video",
                    "--input", str(video),
                    "--out", str(out_eikon),
                    "--grid", "96x48",
                    "--charset", "dense-ref",
                    "--fps", "6",
                    "--pretty"
                ],
            ), contextlib.redirect_stdout(buf):
                main()

            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["grid"], [96, 48])
            self.assertGreaterEqual(payload["total_frames"], 70)
            self.assertIn("idle", payload["state_counts"])
            self.assertTrue(out_eikon.exists())

            # Verify eikon file format (header + first frame line)
            with out_eikon.open() as f:
                header = json.loads(f.readline())
                self.assertEqual(header["t"], "eikon")
                self.assertEqual(header["grid"]["w"], 96)



if __name__ == "__main__":
    unittest.main()
