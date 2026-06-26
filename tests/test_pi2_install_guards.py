from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def write_minimal_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
dependencies = [
  "fastapi>=0.104.0,<1",
  "uvicorn>=0.24.0,<1",
]

[project.optional-dependencies]
web = ["fastapi==0.133.1", "uvicorn==0.41.0"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "tools").mkdir()
    (root / "tools" / "lazy_deps.py").write_text(
        'LAZY_DEPS = {"tool.dashboard": ("fastapi==0.133.1", "uvicorn==0.41.0")}\n',
        encoding="utf-8",
    )
    (root / "hermes_cli").mkdir()
    (root / "hermes_cli" / "web_server.py").write_text(
        'config = uvicorn.Config(app, host="127.0.0.1", port=9119, loop="asyncio")\n',
        encoding="utf-8",
    )
    (root / "setup-pi2.sh").write_text(
        """
#!/usr/bin/env bash
pip install \
  openai \
  pypdf \
  beautifulsoup4

echo "pip install fastapi uvicorn"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "setup-pi2-minimal.sh").write_text(
        """
#!/usr/bin/env bash
MACHINE="$(python - <<'PY'
import platform
print(platform.machine().lower())
PY
)"
if [[ "${HERMES_PI2_TRY_SQLITE_VEC:-0}" == "1" ]]; then
  python -m pip install sqlite-vec
elif [[ "$MACHINE" == armv7l || "$MACHINE" == armv6l ]]; then
  echo "sqlite-vec wheels are unavailable"
else
  python -m pip install sqlite-vec
fi
""".strip()
        + "\n",
        encoding="utf-8",
    )


class Pi2InstallGuardTests(unittest.TestCase):
    def run_guard(self, repo: Path) -> subprocess.CompletedProcess[str]:
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_pi2_install_guards.py"
        return subprocess.run(
            [sys.executable, str(script), "--repo", str(repo)],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_pi2_install_guard_accepts_lightweight_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_minimal_repo(repo)

            result = self.run_guard(repo)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Pi2 install guard checks passed", result.stdout)

    def test_pi2_install_guard_blocks_uvicorn_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_minimal_repo(repo)
            pyproject = repo / "pyproject.toml"
            pyproject.write_text(
                pyproject.read_text(encoding="utf-8").replace("uvicorn==0.41.0", "uvicorn[standard]==0.41.0"),
                encoding="utf-8",
            )

            result = self.run_guard(repo)

        self.assertEqual(result.returncode, 1)
        self.assertIn("uvicorn[standard]", result.stdout)
        self.assertIn("pyproject.toml", result.stdout)

    def test_pi2_install_guard_blocks_heavy_default_rag_deps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_minimal_repo(repo)
            setup = repo / "setup-pi2.sh"
            setup.write_text(
                setup.read_text(encoding="utf-8").replace(
                    "beautifulsoup4",
                    "beautifulsoup4 \\\n  chromadb \\\n  sentence-transformers",
                ),
                encoding="utf-8",
            )

            result = self.run_guard(repo)

        self.assertEqual(result.returncode, 1)
        self.assertIn("chromadb", result.stdout)
        self.assertIn("sentence-transformers", result.stdout)

    def test_pi2_install_guard_requires_asyncio_loop_even_if_comment_mentions_uvloop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_minimal_repo(repo)
            web_server = repo / "hermes_cli" / "web_server.py"
            web_server.write_text(
                "# Avoid uvloop on Pi2.\nconfig = uvicorn.Config(app, host='127.0.0.1', port=9119)\n",
                encoding="utf-8",
            )

            result = self.run_guard(repo)

        self.assertEqual(result.returncode, 1)
        self.assertIn('loop="asyncio"', result.stdout)

    def test_pi2_install_guard_requires_sqlite_vec_opt_in_on_armv7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_minimal_repo(repo)
            (repo / "setup-pi2-minimal.sh").write_text(
                "#!/usr/bin/env bash\npython -m pip install sqlite-vec\n",
                encoding="utf-8",
            )

            result = self.run_guard(repo)

        self.assertEqual(result.returncode, 1)
        self.assertIn("HERMES_PI2_TRY_SQLITE_VEC", result.stdout)


if __name__ == "__main__":
    unittest.main()
