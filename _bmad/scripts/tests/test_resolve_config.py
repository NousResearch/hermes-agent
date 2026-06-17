import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "resolve_config.py"


class ResolveConfigStdoutTests(unittest.TestCase):
    def test_writes_emoji_json_when_stdout_encoding_is_cp1252(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            bmad_dir = project_root / "_bmad"
            bmad_dir.mkdir()
            (bmad_dir / "config.toml").write_text(
                '[agents.emoji-agent]\nname = "Emoji Agent"\nicon = "🧭"\n',
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "cp1252"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--project-root",
                    str(project_root),
                    "--key",
                    "agents",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )

            stderr = result.stderr.decode("utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, msg=stderr)

            output = result.stdout.decode("utf-8")
            self.assertIn("🧭", output)
            resolved = json.loads(output)
            self.assertEqual(resolved["agents"]["emoji-agent"]["icon"], "🧭")


if __name__ == "__main__":
    unittest.main()
