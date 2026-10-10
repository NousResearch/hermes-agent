"""Bounded Hermes phone signing renewal; never starts or restarts any backend."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
from uuid import uuid4

BUNDLE_ID = 'com.local.hermes.companion'


def signing_identity(app_root):
    # The machine's team and device remain in Git-ignored local configuration.
    config = (app_root / 'Local.xcconfig').read_text()
    teams = re.findall(r'^\s*DEVELOPMENT_TEAM\s*=\s*([A-Z0-9]{10})\s*;?\s*$', config, re.M)
    if len(teams) != 1:
        raise ValueError('Expected one explicit DEVELOPMENT_TEAM in Local.xcconfig')
    return teams[0] + '.' + BUNDLE_ID


def built_profile(app_root):
    return app_root / 'BuildEvidence/DeviceDerivedData/Build/Products/Debug-iphoneos/HermesCompanion.app/embedded.mobileprovision'


def verify_built(app_root):
    profile = built_profile(app_root)
    info = plistlib.loads((profile.parent / 'Info.plist').read_bytes())
    if info.get('CFBundleIdentifier') != BUNDLE_ID:
        raise ValueError('Unexpected app bundle identifier')
    return profile_expiry(profile, signing_identity(app_root))


def renewal_verified(before, after, now):
    return (before is None or after > before) and after >= now + timedelta(days=6)


def read_profile(path):
    result = subprocess.run(
        ['/usr/bin/openssl', 'smime', '-inform', 'der', '-verify', '-noverify', '-in', str(path)],
        capture_output=True, check=True, timeout=15,
    )
    return plistlib.loads(result.stdout)


def profile_expiry(path, identity):
    profile = read_profile(path)
    if (profile.get('LocalProvision') is not True
            or profile.get('Entitlements', {}).get('application-identifier') != identity
            or profile.get('TeamIdentifier') != [identity.split('.')[0]]):
        raise ValueError('Unexpected application identity in profile')
    value = profile['ExpirationDate']
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def restore_profiles(staged):
    for original, backup in staged:
        try:
            with os.fdopen(os.open(original, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as destination:
                destination.write(backup.read_bytes())
            original.chmod(0o600)
        except FileExistsError:
            # Xcode may already have written a new profile here. Never replace it.
            pass


def stage_profiles(cache_roots, backup_dir, identity, reader=read_profile):
    """Move only this app's free-development cache; preserve every other profile."""
    staged = []
    try:
        for index, cache in enumerate(cache_roots):
            if cache.is_symlink():
                raise ValueError('Refusing a symlinked provisioning cache')
            for path in cache.glob('*.mobileprovision'):
                if path.is_symlink() or not path.is_file():
                    continue
                original_bytes = path.read_bytes()
                profile = reader(path)
                if profile.get('LocalProvision') is not True or profile.get('Entitlements', {}).get('application-identifier') != identity:
                    continue
                if path.is_symlink() or path.read_bytes() != original_bytes:
                    raise ValueError('Provisioning cache changed during inspection')
                target_dir = backup_dir / str(index)
                target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                backup = target_dir / path.name
                if backup.exists():
                    raise FileExistsError('Provisioning backup already exists')
                path.rename(backup)
                staged.append((path, backup))
                backup.chmod(0o600)
        return staged
    except BaseException:
        restore_profiles(staged)
        raise


def refresh(app_root, check_only=False):
    output = app_root / 'BuildEvidence'
    identity = signing_identity(app_root)
    profile = built_profile(app_root)
    if check_only:
        before = verify_built(app_root) if profile.is_file() else None
        print(json.dumps({'mode': 'check_only', 'cached_profile_expires_at': before.isoformat() if before else None,
                          'renewal_attempted': False, 'device_install_verified': False}))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'phone-refresh.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Hermes refresh skipped: another refresh is still running.')
            return 1
        # Read baseline under the lock, before moving any profile.
        before = verify_built(app_root) if profile.is_file() else None
        device = (app_root / 'LocalDevice.txt').read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*', device):
            raise ValueError('Missing or invalid saved phone identifier')
        now = datetime.now(timezone.utc)
        attempt = now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
        log_path = output / ('phone-refresh-' + attempt + '.log')
        result = {'started_at': now.isoformat(), 'status': 'needs_attention',
                  'before_expires_at': before.isoformat() if before else None,
                  'renewal_verified': False, 'log_path': str(log_path)}
        cache_roots = [Path.home() / 'Library/MobileDevice/Provisioning Profiles',
                       Path.home() / 'Library/Developer/Xcode/UserData/Provisioning Profiles']
        staged = []
        try:
            staged = stage_profiles(cache_roots, output / 'provisioning-backups' / attempt, identity)
            result['cached_profiles_backed_up'] = len(staged)
            with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as log:
                env = dict(os.environ)
                env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'
                process = subprocess.Popen(['/bin/bash', str(app_root / 'install-iphone.sh'), '--no-launch'],
                    cwd=app_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
            result['installer_exit_code'] = code
            result['launch_attempted'] = False
            if code == 0:
                after = verify_built(app_root)
                result['after_expires_at'] = after.isoformat()
                result['renewal_verified'] = renewal_verified(before, after, datetime.now(timezone.utc))
                if result['renewal_verified']:
                    result['status'] = 'renewed'
                    result['message'] = 'Hermes installed with an extended signing expiry. Launch skipped; no screen unlock required for launch.'
                else:
                    result['message'] = 'Hermes reinstalled, but Apple did not extend expiry enough. Renew in Xcode before expiration; this was not a successful renewal.'
            else:
                result['message'] = 'Refresh could not complete. Connect the saved iPhone and check Xcode Apple-account access, then retry. No app data or pairing state was deleted.'
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, plistlib.InvalidFileException) as exc:
            result['message'] = 'Renewal failed: ' + type(exc).__name__ + '. No successful renewal is claimed.'
        finally:
            if not result['renewal_verified']:
                restore_profiles(staged)
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        result['receipt_path'] = str(output / ('phone-refresh-' + attempt + '.json'))
        encoded = json.dumps(result, indent=2) + '\n'
        with os.fdopen(os.open(result['receipt_path'], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as immutable:
            immutable.write(encoded)
        receipt = output / 'phone-refresh-latest.json'
        temp = receipt.with_suffix('.tmp')
        temp.write_text(encoded)
        temp.chmod(0o600)
        temp.replace(receipt)
        print(json.dumps(result, indent=2))
        return 0 if result['renewal_verified'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read cached profile expiry without rebuilding or installing')
    parser.add_argument('--verify-built', action='store_true', help='Verify app/profile identity before device installation')
    args = parser.parse_args()
    try:
        if args.verify_built:
            print('Hermes build identity verified; expiry: ' + verify_built(Path(__file__).resolve().parents[1]).isoformat())
            raise SystemExit(0)
        raise SystemExit(refresh(Path(__file__).resolve().parents[1], args.check))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, plistlib.InvalidFileException) as exc:
        print('Hermes refresh needs attention: ' + type(exc).__name__ + '. No renewal is claimed; inspect the local build log.')
        raise SystemExit(1)
