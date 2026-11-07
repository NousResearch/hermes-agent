"""Deployment regressions; all build/install commands are isolated fakes."""
from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phone_refresh', ROOT / 'scripts/refresh_hermes_phone.py')
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)
IDENTITY = 'TESTTEAM01.com.local.hermes.companion'


class PhoneRenewalTests(unittest.TestCase):
    def test_signing_identity_uses_local_team_and_exact_hermes_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'Local.xcconfig'
            config.write_text('DEVELOPMENT_TEAM = TESTTEAM01\n')
            self.assertEqual(refresh.signing_identity(root), IDENTITY)
            config.write_text('DEVELOPMENT_TEAM = $(INHERITED)\n')
            with self.assertRaises(ValueError):
                refresh.signing_identity(root)

    def test_only_exact_free_hermes_profiles_move_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            fixtures = {
                'hermes': (True, IDENTITY),
                'other_app': (True, 'TESTTEAM01.com.example.othercompanion'),
                'other_team': (True, 'OTHERTEAM1.com.local.hermes.companion'),
                'paid': (False, IDENTITY),
            }
            originals = {}
            for name, (local, identity) in fixtures.items():
                path = cache / (name + '.mobileprovision')
                path.write_bytes(plistlib.dumps({'LocalProvision': local,
                    'Entitlements': {'application-identifier': identity}}))
                originals[path] = path.read_bytes()
            (cache / 'link.mobileprovision').symlink_to(cache / 'other_app.mobileprovision')
            staged = refresh.stage_profiles([cache], root / 'backups', IDENTITY,
                reader=lambda p: plistlib.loads(p.read_bytes()))
            self.assertEqual(len(staged), 1)
            self.assertFalse((cache / 'hermes.mobileprovision').exists())
            for path, data in originals.items():
                if path.name != 'hermes.mobileprovision':
                    self.assertEqual(path.read_bytes(), data)
            refresh.restore_profiles(staged)
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)
            self.assertTrue((cache / 'link.mobileprovision').is_symlink())

    def test_partial_staging_failure_restores_moved_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            cache.mkdir()
            path = cache / 'ours.mobileprovision'
            path.write_bytes(b'profile')
            symlink = root / 'symlink'
            symlink.symlink_to(cache, target_is_directory=True)
            with self.assertRaises(ValueError):
                refresh.stage_profiles([cache, symlink], root / 'backups', IDENTITY,
                    reader=lambda _: {'LocalProvision': True,
                                      'Entitlements': {'application-identifier': IDENTITY}})
            self.assertEqual(path.read_bytes(), b'profile')

    def test_restore_does_not_overwrite_new_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, backup = root / 'original', root / 'backup'
            original.write_bytes(b'new')
            backup.write_bytes(b'old')
            refresh.restore_profiles([(original, backup)])
            self.assertEqual(original.read_bytes(), b'new')

    def test_expiry_requires_advancement_and_six_days_remaining(self):
        now = datetime(2026, 8, 28, tzinfo=timezone.utc)
        old = now + timedelta(days=1)
        for before, after, expected in ((old, old, False),
                (old, now + timedelta(days=2), False),
                (old, now + timedelta(days=7), True),
                (None, now + timedelta(days=7), True)):
            with self.subTest(before=before, after=after):
                self.assertEqual(refresh.renewal_verified(before, after, now), expected)

    def test_wrong_profile_identity_is_rejected(self):
        profile = {'LocalProvision': True, 'TeamIdentifier': ['TESTTEAM01'],
                   'Entitlements': {'application-identifier': 'TESTTEAM01.com.example.othercompanion'}}
        with patch.object(refresh, 'read_profile', return_value=profile):
            with self.assertRaises(ValueError):
                refresh.profile_expiry(Path('unused'), IDENTITY)

    def test_unattended_clean_install_skips_launch_and_keeps_signature_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy2(ROOT / 'install-iphone.sh', root / 'install-iphone.sh')
            (root / 'Local.xcconfig').write_text('DEVELOPMENT_TEAM = TESTTEAM01\n')
            (root / 'LocalDevice.txt').write_text('test-phone\n')
            (root / 'rebuild.sh').write_text('#!/bin/bash\nexit 0\n')
            (root / 'rebuild.sh').chmod(0o700)
            commands = root / 'bin'
            commands.mkdir()
            calls = root / 'calls'
            for name in ('git', 'xcodebuild', 'codesign', 'xcrun', 'python3'):
                tool = commands / name
                tool.write_text('#!/bin/bash\nname="${0##*/}"\nprintf "%s %s\\n" "$name" "$*" >> "$TEST_CALLS"\n'
                                'if [[ "$name" == codesign && "${FAIL_SIGN:-0}" == 1 ]]; then exit 7; fi\n')
                tool.chmod(0o700)
            env = dict(os.environ, PATH=str(commands) + ':/usr/bin:/bin', TEST_CALLS=str(calls))
            script = ['/bin/bash', str(root / 'install-iphone.sh')]
            result = subprocess.run(script + ['--no-launch'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = calls.read_text()
            self.assertIn(' clean build\n', recorded)
            self.assertIn('codesign --verify --deep --strict', recorded)
            self.assertIn('refresh_hermes_phone.py --verify-built', recorded)
            self.assertIn('device install app --device test-phone', recorded)
            self.assertNotIn('process launch', recorded)
            calls.write_text('')
            result = subprocess.run(script + ['test-phone'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('process launch', calls.read_text())
            self.assertNotIn(' clean build', calls.read_text())
            calls.write_text('')
            result = subprocess.run(script + ['--no-launch'], env=dict(env, FAIL_SIGN='1'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('device install', calls.read_text())


if __name__ == '__main__':
    unittest.main()
