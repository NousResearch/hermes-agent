"""Plugin-owned generated paths do not invalidate replayable dependency inputs.

Standalone stdlib unittest lane: no pytest provisioning or network is required.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pm import workspace
from pm.environment import PythonEnvironment


BACKEND = '''from pathlib import Path
from zipfile import ZipFile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    assert Path("README.md").read_text() == "fixture"
    assert Path("build/backend.py").is_file()
    assert Path("dist/runtime.txt").is_file()
    name = "snapshot_fixture-1.0-py3-none-any.whl"
    dist = "snapshot_fixture-1.0.dist-info"
    entries = {
        "snapshot_fixture/__init__.py": Path("snapshot_fixture/__init__.py").read_bytes(),
        "snapshot_fixture/asset.txt": Path("reports/required.txt").read_bytes(),
        dist + "/METADATA": "Metadata-Version: 2.1\\nName: snapshot-fixture\\nVersion: 1.0\\n",
        dist + "/WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    }
    entries[dist + "/RECORD"] = "".join(path + ",,\\n" for path in entries)
    with ZipFile(Path(wheel_directory) / name, "w") as wheel:
        for path, data in entries.items():
            wheel.writestr(path, data)
    return name

build_editable = build_wheel
'''


class PluginSnapshotInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plugin = self.root / "plugin"
        self.plugin.mkdir()
        self.put("pyproject.toml", '[project]\nname="snapshot-fixture"\nversion="1.0"\n'
                 'requires-python=">=3.11"\n[build-system]\nrequires=[]\n'
                 'build-backend="backend"\nbackend-path=["build"]\n'
                 '[tool.hermes.build-inputs]\ngenerated=["reports", "logs", "dumps", "request_dump.json", "build", "dist"]\n'
                 'required=["reports/required.txt", "build/backend.py", "dist/runtime.txt"]\n')
        self.put("plugin.yaml", "name: snapshot-fixture\n")
        self.put("README.md", "fixture")
        self.put("build/backend.py", BACKEND)
        self.put("dist/runtime.txt", "shipped dist asset")
        self.put("snapshot_fixture/__init__.py", "VALUE = 'source-one'\n")
        self.put("reports/required.txt", "asset-one")
        for path in ("reports/report.json", "logs/run.log", "dumps/payload.json", "request_dump.json"):
            self.put(path, "output-one")

    def put(self, path, text):
        target = self.plugin / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def member(self, name):
        return workspace._workspace_member(self.plugin, self.root / name, identity=self.plugin)

    def tree(self, path):
        return {p.relative_to(path).as_posix(): p.read_bytes()
                for p in path.rglob("*") if p.is_file()}

    def test_generated_outputs_do_not_change_stamp_or_copy(self):
        before = workspace.members_stamp([self.plugin])
        first = self.tree(self.member("first"))
        for path in ("reports/report.json", "logs/run.log", "dumps/payload.json", "request_dump.json"):
            self.put(path, "changed disposable output")
        self.put("reports/new.json", "new disposable report")
        self.assertEqual(before, workspace.members_stamp([self.plugin]))
        second = self.tree(self.member("second"))
        self.assertEqual(first, second)
        for path in ("reports/report.json", "logs/run.log", "dumps/payload.json", "request_dump.json"):
            self.assertNotIn(path, second)
        for path in ("reports/required.txt", "build/backend.py", "dist/runtime.txt", "plugin.yaml", "pyproject.toml"):
            self.assertIn(path, second)
        for path in ("snapshot_fixture/__init__.py", "reports/required.txt", "dist/runtime.txt", "build/backend.py"):
            with self.subTest(functional=path):
                stamp = workspace.members_stamp([self.plugin])
                self.put(path, (self.plugin / path).read_text() + "\n# functional change\n")
                self.assertNotEqual(stamp, workspace.members_stamp([self.plugin]))
                copied = self.member("changed-" + path.replace("/", "-"))
                self.assertEqual((copied / path).read_bytes(), (self.plugin / path).read_bytes())

    def test_undeclared_legacy_assets_are_not_guessed_from_names(self):
        path = self.plugin / "pyproject.toml"
        path.write_text(path.read_text().split('[tool.hermes.build-inputs]')[0])
        before = workspace.members_stamp([self.plugin])
        self.put("reports/report.json", "could be a build input")
        self.assertNotEqual(before, workspace.members_stamp([self.plugin]))
        self.assertIn("reports/report.json", self.tree(self.member("legacy")))

    def test_policy_rejects_escapes_and_malformed_declarations(self):
        path = self.plugin / "pyproject.toml"
        prefix = path.read_text().split('[tool.hermes.build-inputs]')[0]
        invalid = ['generated=["../outside"]', 'generated=["/absolute"]', 'generated="logs"',
                   'required=[42]', 'generated=["."]', 'generated=["logs/../source"]',
                   'generated=["C:\\\\outside"]', 'unknown=[]', 'generated=["logs//run"]']
        for policy in invalid:
            with self.subTest(policy=policy):
                path.write_text(prefix + '[tool.hermes.build-inputs]\n' + policy + '\n')
                with self.assertRaisesRegex(ValueError, "build-inputs"):
                    workspace.members_stamp([self.plugin])

    def test_declarations_and_caches_cannot_be_overridden(self):
        self.put("__pycache__/required.pyc", "not an input")
        path = self.plugin / "pyproject.toml"
        prefix = path.read_text().split('[tool.hermes.build-inputs]')[0]
        path.write_text(prefix + '[tool.hermes.build-inputs]\ngenerated=["pyproject.toml", "plugin.yaml"]\n'
                        'required=["__pycache__"]\n')
        copied = self.tree(self.member("protected"))
        self.assertIn("pyproject.toml", copied)
        self.assertIn("plugin.yaml", copied)
        self.assertNotIn("__pycache__/required.pyc", copied)

    def test_offline_declared_editable_build_and_recorded_replay(self):
        uv = shutil.which("uv")
        self.assertIsNotNone(uv)
        core = self.root / "core"
        core.mkdir()
        (core / "pyproject.toml").write_text('[project]\nname="snapshot-core"\nversion="1"\n'
                                            'requires-python=">=3.11"\n[tool.uv]\npackage=false\nno-index=true\n')
        def build(name, replay=None):
            environment = PythonEnvironment(uv=Path(uv), python=Path(sys.executable),
                destination=self.root / (name + "-env"), cache=self.root / (name + "-cache"),
                env={"PATH": str(Path(uv).parent) + ":/usr/bin:/bin", "HOME": str(self.root),
                     "TMPDIR": str(self.root)}, offline=True)
            saved = self.root / name
            workspace.lock_and_sync([self.plugin], [], root=saved, source=core,
                                    seed_lock=None, environment=environment, replay=replay)
            process = subprocess.Popen([str(environment.executable), "-I", "-c",
                "import snapshot_fixture as p; from importlib.resources import files; "
                "print(p.VALUE, files(p).joinpath('asset.txt').read_text())"],
                cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={"HOME": str(self.root), "PATH": "/usr/bin:/bin"})
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
            print("IMPORT_PID_EVIDENCE", name, "pid", process.pid, stdout.strip(), "exit", process.returncode)
            return saved, stdout.strip()
        saved, actual = build("saved")
        self.assertEqual(actual, "source-one asset-one")
        self.assertNotIn(b"output-one", b"".join(self.tree(saved).values()))
        recorded_lock = (saved / "uv.lock").read_bytes()
        # Real functional input changes alter both the stamp and installed behavior.
        stamp = workspace.members_stamp([self.plugin])
        self.put("snapshot_fixture/__init__.py", "VALUE = 'source-two'\n")
        self.put("reports/required.txt", "asset-two")
        self.assertNotEqual(stamp, workspace.members_stamp([self.plugin]))
        _, actual = build("updated")
        self.assertEqual(actual, "source-two asset-two")
        # Repair uses the saved inputs, not live declarations, output rules or backend.
        self.put("pyproject.toml", "damaged [")
        self.put("build/backend.py", "raise RuntimeError('damaged live backend')\n")
        replay, actual = build("replayed", replay=saved)
        self.assertEqual(actual, "source-one asset-one")
        self.assertEqual((replay / "uv.lock").read_bytes(), recorded_lock)

    def test_tools_store_unreadable_facts_preserves_unknown_inputs(self):
        tools = self.root / "tools"
        (tools / "python-published").mkdir(parents=True)
        (tools / "fetch-abc").mkdir()
        (tools / "source").mkdir()
        (tools / "source/code.py").write_text("VALUE=1\n")
        (tools / "python-published/payload").write_text("binary fixture")
        (tools / "fetch-abc/payload").write_text("cache fixture")
        (self.root / "pyproject.toml").write_text(
            '[project]\nname="core"\nversion="1"\n'
            '[tool.setuptools.packages.find]\ninclude=["tools"]\n')
        def copied(name):
            target = self.root / name
            workspace._copy_core_inputs(self.root, target)
            self.assertTrue((target / "tools/source/code.py").is_file())
            return target / "tools"
        facts = tools / "facts.json"
        facts.write_text(json.dumps({"packages": {"python": {"entry": "python-published"}}}))
        self.assertEqual(workspace._published_store_entries(tools), {"python-published", "fetch-abc"})
        valid = copied("published-copy")
        self.assertFalse((valid / "python-published").exists())
        self.assertFalse((valid / "fetch-abc").exists())
        facts.unlink()
        self.assertEqual(workspace._published_store_entries(tools), set())
        self.assertTrue((copied("missing-copy") / "python-published/payload").is_file())
        facts.write_text("invalid JSON")
        self.assertEqual(workspace._published_store_entries(tools), set())
        original = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == facts:
                raise PermissionError("fixture unreadable facts")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", unreadable):
            self.assertEqual(workspace._published_store_entries(tools), set())
            self.assertTrue((copied("unreadable-copy") / "python-published/payload").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
