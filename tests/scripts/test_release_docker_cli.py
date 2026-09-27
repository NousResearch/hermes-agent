"""Docker receipt CLI validates identity and hashes actual local archive bytes."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_cli_manifest_and_verify(tmp_path):
    identity = ['--tag', 'v1.2.3', '--commit', 'a' * 40]
    digests = ['--digest-amd64', 'b' * 64, '--digest-arm64', 'c' * 64]
    archives = []
    expected = {}
    for arch, data in [('amd64', b'first archive'), ('arm64', b'second archive')]:
        path = tmp_path / f'{arch}.tar'
        path.write_bytes(data)
        archives.extend([f'--archive-{arch}', str(path)])
        expected[arch] = hashlib.sha256(data).hexdigest()

    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'scripts.releases.docker', *args],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)

    for extra in ([], archives):
        result = cli('manifest', *identity, *digests, *extra)
        assert result.returncode == 0, result.stderr
        manifest = json.loads(result.stdout)
        assert manifest['digests'] == {'amd64': 'b' * 64, 'arm64': 'c' * 64}
        assert manifest.get('archives') == (expected if extra else None)
        out = tmp_path / 'manifest.json'
        out.write_text(result.stdout, encoding='utf-8')
        assert cli('verify', *identity, str(out)).returncode == 0
    for extra in (archives[:2], ['--digest-arm64', 'z' * 64]):
        result = cli('manifest', *identity, *digests, *extra)
        assert result.returncode == 1 and '::error::' in result.stderr
    for change in [
        {'tag': 'v1.2.4'}, {'commit': 'b' * 40}, {'schema': 2},
        {'digests': {'amd64': 'b' * 64}}, {'digests': {'amd64': 'b' * 64, 'arm64': 'z' * 64}},
        {'archives': {'amd64': 'd' * 64}}, {'archives': {'amd64': 'd' * 64, 'riscv64': 'e' * 64}},
        {'list-digest': 'sha256:wrong'}, None,
    ]:
        bad = copy.deepcopy(manifest)
        if change:
            bad.update(change)
        out.write_text(json.dumps(bad) if change else 'not json', encoding='utf-8')
        result = cli('verify', *identity, str(out))
        assert result.returncode == 1 and '::error::' in result.stderr


def test_manifest_admits_the_attempt_ref_image_tag():
    from scripts.releases.docker import DockerReleaseError, build_manifest, parse_manifest

    manifest = build_manifest("rc.1-v1.2.3", "a" * 40, {"amd64": "b" * 64, "arm64": "c" * 64})
    assert parse_manifest(json.dumps(manifest).encode())["tag"] == "rc.1-v1.2.3"
    # The old suffix shape is dead; a recut reuses no image tag.
    with pytest.raises(DockerReleaseError):
        build_manifest("v1.2.3-rc", "a" * 40, {"amd64": "b" * 64, "arm64": "c" * 64})
    with pytest.raises(DockerReleaseError):
        parse_manifest(json.dumps(dict(manifest, tag="not-a-tag")).encode())


def test_promotion_reuses_the_receipt_digest_without_rebuilding():
    from scripts.releases.docker import DockerReleaseError, promote_stable

    digest = 'sha256:' + 'd' * 64
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[:4] == ['docker', 'buildx', 'imagetools', 'inspect']:
            return digest
        if argv[:4] == ['docker', 'buildx', 'imagetools', 'create']:
            return ''
        raise AssertionError(argv)

    promote_stable('v1.2.3', digest, run=run)
    create = next(argv for argv in calls if argv[3] == 'create')
    assert create[-1] == f'nousresearch/hermes-agent@{digest}'
    assert all('build' not in argv for argv in calls)

    with pytest.raises(DockerReleaseError, match='versioned tag'):
        promote_stable('v1.2.3', digest, run=lambda _argv: 'sha256:' + 'e' * 64)


def test_promotion_preserves_independent_desktop_digest():
    from scripts.releases.docker import promote_stable

    slim, desktop = ('sha256:' + c * 64 for c in 'de')
    inspected = []
    created = []

    def run(argv):
        if argv[3] == 'inspect':
            ref = argv[4]
            inspected.append(ref)
            return desktop if ref.endswith('-desktop') else slim
        if argv[3] == 'create':
            created.append(argv)
            return ''
        raise AssertionError(argv)

    promote_stable('v1.2.3', slim, run=run)
    assert f'nousresearch/hermes-agent:v1.2.3-desktop' in inspected
    assert {tuple(cmd[4:]) for cmd in created} == {
        ('-t', 'nousresearch/hermes-agent:stable', '-t', 'nousresearch/hermes-agent:latest',
         f'nousresearch/hermes-agent@{slim}'),
        ('-t', 'nousresearch/hermes-agent:stable-desktop', '-t',
         'nousresearch/hermes-agent:latest-desktop', f'nousresearch/hermes-agent@{desktop}'),
    }
    assert 'nousresearch/hermes-agent:stable-desktop' in inspected
    assert 'nousresearch/hermes-agent:latest-desktop' in inspected


def test_promotion_requires_desktop_version_before_moving_any_alias():
    from scripts.releases.docker import DockerReleaseError, promote_stable

    created = []

    def run(argv):
        if argv[3] == 'inspect':
            if argv[4].endswith('-desktop'):
                raise subprocess.CalledProcessError(1, argv)
            return 'sha256:' + 'd' * 64
        created.append(argv)
        return ''

    with pytest.raises(DockerReleaseError, match='desktop'):
        promote_stable('v1.2.3', 'sha256:' + 'd' * 64, run=run)
    assert not created


def test_published_digest_checks_desktop_tag():
    from scripts.releases.docker import DockerReleaseError, published_digest

    inspected = []

    def run(argv):
        inspected.append(argv[4])
        return 'sha256:' + ('e' if argv[4].endswith('-desktop') else 'd') * 64

    assert published_digest('v1.2.3', run=run) == 'sha256:' + 'd' * 64
    assert inspected == ['nousresearch/hermes-agent:v1.2.3',
                         'nousresearch/hermes-agent:v1.2.3-desktop']
    with pytest.raises(DockerReleaseError, match='desktop'):
        published_digest('v1.2.3', run=lambda argv: 'garbage' if argv[4].endswith('-desktop') else 'sha256:' + 'd' * 64)
