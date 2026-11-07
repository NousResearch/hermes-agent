"""
Offline unit & integration tests for the `gradio-app-development` skill.

Validates:
  1. SKILL.md frontmatter compliance against Hermes Agent hardline authoring rules
  2. Portable paths (no machine-local `/home/` paths, no hardcoded `/tmp` in scripts)
  3. Linked references and scripts exist inside the skill directory
  4. `jit_gradio_space.py scaffold` + `verify` for all 4 templates
  5. `jit_gradio_space.py inspect-space` with mocked `gradio_client.Client`
  6. `jit_gradio_space.py deploy` with mocked `huggingface_hub` (`HfApi`, `whoami`)
"""
from __future__ import annotations

import ast
import importlib.util
import io
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch


def _resolve_skill_dir() -> Path:
    here = Path(__file__).resolve().parent
    # Works whether located at <repo>/tests/ or <hermes-agent>/tests/skills/
    candidates = [
        here.parent / "skills" / "gradio-app-development",
        here.parent.parent / "optional-skills" / "mlops" / "gradio-app-development",
        here.parent.parent / "skills" / "mlops" / "gradio-app-development",
    ]
    for candidate in candidates:
        if (candidate / "SKILL.md").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate gradio-app-development skill directory from {here}")


SKILL_DIR = _resolve_skill_dir()
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "jit_gradio_space.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("jit_gradio_space", str(SCRIPT_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGradioAppDevelopmentSkill(unittest.TestCase):
    def test_frontmatter_and_structure(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---\n"), "SKILL.md must start with --- at byte 0")
        match = re.search(r"\n---\s*\n", content[3:])
        self.assertIsNotNone(match, "SKILL.md must close YAML frontmatter with ---")

        fm_raw = content[4 : match.start() + 3]
        self.assertIn("name: gradio-app-development", fm_raw)
        self.assertIn("author: Jonathan Korstad (Jpalmer95), Hermes Agent", fm_raw)
        self.assertIn("platforms: [linux, macos, windows]", fm_raw)

        desc_match = re.search(r"^description:\s*(.+)$", fm_raw, re.MULTILINE)
        self.assertIsNotNone(desc_match)
        desc = desc_match.group(1).strip().strip('"').strip("'")
        self.assertLessEqual(len(desc), 60, f"Description exceeds 60-char hardline ({len(desc)} chars): {desc!r}")
        self.assertTrue(desc.endswith("."), "Description must end with a period")

    def test_no_machine_local_paths(self):
        for path in SKILL_DIR.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py"}:
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/home/", text, f"Machine-local /home/ path found in {path}")
                if path.suffix == ".py":
                    self.assertNotIn('"/tmp', text, f"Hardcoded /tmp path found in {path}; use tempfile.gettempdir()")
                    self.assertNotIn("'/tmp", text, f"Hardcoded /tmp path found in {path}; use tempfile.gettempdir()")

    def test_linked_files_exist(self):
        self.assertTrue((SKILL_DIR / "references" / "live-docs-and-api-cheatsheet.md").is_file())
        self.assertTrue((SKILL_DIR / "references" / "daggr-in-gradio-patterns.md").is_file())
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_scaffold_and_ast_check_all_templates(self):
        mod = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            for tmpl in ("blocks", "daggr-modular", "daggr-hybrid", "workflow"):
                out_dir = Path(tmp) / tmpl
                rc = mod.main([
                    "scaffold",
                    str(out_dir),
                    "--title",
                    f"Test {tmpl}",
                    "--template",
                    tmpl,
                ])
                self.assertEqual(rc, 0)
                app_py = out_dir / "app.py"
                readme = out_dir / "README.md"
                reqs = out_dir / "requirements.txt"
                self.assertTrue(app_py.is_file())
                self.assertTrue(readme.is_file())
                self.assertTrue(reqs.is_file())

                # AST parse must succeed on every generated app.py
                ast.parse(app_py.read_text(encoding="utf-8"), filename=str(app_py))

                # README must have valid HF Spaces YAML frontmatter
                readme_text = readme.read_text(encoding="utf-8")
                self.assertTrue(readme_text.startswith("---\n"))
                self.assertIn("sdk: gradio", readme_text)
                self.assertIn("hf_oauth: true", readme_text)

    def test_inspect_space_with_mocked_client(self):
        mod = _load_script_module()
        fake_client_mod = types.ModuleType("gradio_client")
        mock_client_instance = MagicMock()
        mock_client_instance.view_api.return_value = {
            "named_endpoints": {
                "/infer": {
                    "parameters": [
                        {
                            "parameter_name": "prompt",
                            "python_type": {"type": "str"},
                            "parameter_has_default": False,
                        },
                        {
                            "parameter_name": "steps",
                            "python_type": {"type": "int"},
                            "parameter_has_default": True,
                            "parameter_default": 4,
                        },
                    ],
                    "returns": [
                        {"label": "Result", "python_type": {"type": "filepath"}},
                    ],
                }
            }
        }
        fake_client_mod.Client = MagicMock(return_value=mock_client_instance)

        buf = io.StringIO()
        with patch.dict(sys.modules, {"gradio_client": fake_client_mod}), redirect_stdout(buf):
            rc = mod.main(["inspect-space", "black-forest-labs/FLUX.1-schnell", "--token", "hf_fake"])

        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("api_name='/infer'", out)
        self.assertIn("prompt: str (required)", out)
        self.assertIn("steps: int = 4", out)

    def test_deploy_with_mocked_hf_api(self):
        mod = _load_script_module()
        fake_hf_mod = types.ModuleType("huggingface_hub")
        mock_api = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.stage = "RUNNING"
        mock_runtime.hardware = "cpu-basic"
        mock_api.get_space_runtime.return_value = mock_runtime

        fake_hf_mod.HfApi = MagicMock(return_value=mock_api)
        fake_hf_mod.whoami = MagicMock(return_value={"name": "testuser"})

        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "app"
            vendor_file = Path(tmp) / "helper_mod.py"
            vendor_file.write_text("X = 1\n", encoding="utf-8")

            mod.main(["scaffold", str(app_dir), "--template", "blocks"])

            buf = io.StringIO()
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_mod}), redirect_stdout(buf):
                rc = mod.main([
                    "deploy",
                    str(app_dir),
                    "--repo-id",
                    "my-jit-space",
                    "--token",
                    "hf_test_secret",
                    "--forward-hf-token",
                    "--secret",
                    "CUSTOM_KEY=val123",
                    "--vendor",
                    str(vendor_file),
                ])

            self.assertEqual(rc, 0)
            mock_api.create_repo.assert_called_once()
            mock_api.upload_folder.assert_called_once()
            self.assertEqual(mock_api.add_space_secret.call_count, 2)
            self.assertIn("https://huggingface.co/spaces/testuser/my-jit-space", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
