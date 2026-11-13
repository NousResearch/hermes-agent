"""Current/check paths must not expand collection without mapped-consumer protection."""
import mmap
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hermes_cli import runtime_state, venv_sync


class CurrentGenerationRetention(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1"\n')
        self.install = self.root / "install"
        self.generations = self.install / "environments"
        self.generations.mkdir(parents=True)
        self.selected = self.generation("selected")
        self.addCleanup(patch.stopall)
        patch.object(runtime_state, "install_state_dir", lambda project: self.install).start()
        patch("pm.environments.selected_venv", lambda project: self.selected / "venv").start()

    def generation(self, name, *, managed=True, young=False):
        path = self.generations / name
        (path / "venv").mkdir(parents=True)
        (path / "workspace").mkdir()
        (path / "workspace/asset").write_bytes(b"mapped interpreter/cwd/native closure fixture")
        if managed:
            marker = path / ".lease-managed"
            marker.touch()
            if not young:
                age = time.time() - 172800
                os.utime(marker, (age, age))
        return path

    def tree(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def test_existing_collector_only_removes_eligible_after_lease_release(self):
        eligible = self.generation("eligible")
        leased = self.generation("leased")
        young = self.generation("young", young=True)
        unmanaged = self.generation("unmanaged", managed=False)
        linked = self.generations / "linked"
        linked.symlink_to(unmanaged, target_is_directory=True)
        release_eligible = runtime_state.lease_directory(eligible)
        release_leased = runtime_state.lease_directory(leased)
        self.addCleanup(release_leased)
        self.assertEqual(runtime_state.collect_generations(self.project), [])
        release_eligible()
        self.assertEqual(runtime_state.collect_generations(self.project), [eligible])
        for path in (self.selected, leased, young, unmanaged, linked):
            self.assertTrue(path.is_dir(), str(path))

    def test_current_and_check_preserve_real_mapped_unleased_closure(self):
        mapped = self.generation("mapped-unleased")
        with (mapped / "workspace/asset").open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                before = self.tree()
                for check in (False, True):
                    with self.subTest(check=check):
                        with patch.object(venv_sync, "_is_sealed", return_value=False), \
                             patch("pm.venv_is_current", return_value=True), \
                             patch.object(venv_sync, "publish_launchers") as launchers, \
                             patch.object(venv_sync, "collect_superseded_generations") as collect, \
                             patch("pm.sync_venv") as sync:
                            self.assertEqual(venv_sync.sync(self.project, check=check),
                                             {"state": "current", "ok": True})
                            collect.assert_not_called()
                            sync.assert_not_called()
                            if check:
                                launchers.assert_not_called()
                        self.assertEqual(before, self.tree())
                        self.assertEqual(mapping[:], (mapped / "workspace/asset").read_bytes())

    def test_stale_check_is_inert(self):
        before = self.tree()
        with patch.object(venv_sync, "_is_sealed", return_value=False), \
             patch("pm.venv_is_current", return_value=False), \
             patch.object(venv_sync, "publish_launchers") as launchers, \
             patch.object(venv_sync, "collect_superseded_generations") as collect, \
             patch("pm.sync_venv") as sync:
            self.assertEqual(venv_sync.sync(self.project, check=True),
                             {"state": "would-sync", "ok": True})
            launchers.assert_not_called()
            collect.assert_not_called()
            sync.assert_not_called()
        self.assertEqual(before, self.tree())


if __name__ == "__main__":
    unittest.main(verbosity=2)
