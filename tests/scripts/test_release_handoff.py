"""Release handoffs use exact R2 bytes without advancing a channel."""
import copy
import json
from urllib.parse import unquote

import pytest

from scripts.releases import handoff, r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


@pytest.mark.parametrize('tag', ['v1.2.3', 'v1.2.3+canary.20260908T232538Z', 'rc.1-v1.2.3', None])
def test_stage_and_fetch_bind_tag_commit_and_files_without_feed_writes(tmp_path, monkeypatch, r2_server, tag):
    commit = "a" * 40
    identity = ['--tag', tag, '--commit', commit] if tag else ['--commit-build', commit]
    prefix = f'releases/tag/{tag}/' if tag else f'releases/commit/{commit}/'
    monkeypatch.setenv('GITHUB_SHA', 'c' * 40)
    root = tmp_path / "built"
    root.mkdir()
    data = b"multi-chunk package transport fixture\n" * 90000
    (root / "app.msix").write_bytes(data)
    (root / "Store-app.msix").write_bytes(b"store transport fixture")
    (root / "metadata-windows-x64.json").write_text("{}", encoding="utf-8")
    nested = root / "apt" / "pool build" / "InRelease"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"APT index transport fixture")
    handoff.main(["stage", *identity, "--name", "win32-x64",
                  "--root", str(root), "--include", "*.msix", "--include", "metadata-*.json", "--include", "apt/**/*"])
    receipt_key = prefix + 'handoff-win32-x64.json'
    receipt = json.loads(r2_server.store[receipt_key][0])
    assert receipt['schema'] == (1 if tag else 2) and receipt['commit'] == commit
    assert receipt['tag'] == tag if tag else 'tag' not in receipt
    assert {row["path"] for row in receipt["files"]} == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert all(key.startswith(prefix) for key in r2_server.store)
    puts = [path for method, path, _ in r2_server.requests if method == "PUT"]
    assert unquote(puts[-1]).endswith(receipt_key)
    assert all(headers.get("If-None-Match") == "*" for method, _, headers in r2_server.requests if method == "PUT")

    downloaded = tmp_path / "downloaded"
    handoff.main(["fetch", *identity, "--name", "win32-x64",
                  "--root", str(downloaded), "--include", "*.msix"])
    assert (downloaded / "app.msix").read_bytes() == data
    assert (downloaded / "Store-app.msix").read_bytes() == b"store transport fixture"
    assert not (downloaded / "metadata-windows-x64.json").exists()
    handoff.main(['stage', *identity, *([] if tag else ['--commit', commit]), '--name', 'win32-x64',
                  '--root', str(root), '--include', '*.msix', '--include', 'metadata-*.json', '--include', 'apt/**/*'])
    handoff.main(['fetch', *identity, '--name', 'win32-x64', '--root', str(downloaded), '--include', 'apt/*'])
    assert (downloaded / nested.relative_to(root)).read_bytes() == nested.read_bytes()

    fetch = lambda target: handoff.main(['fetch', *identity, '--name', 'win32-x64', '--root', str(target)])
    for changed in ({'schema': 2 if tag else 1}, {'commit': 'b' * 40}, {'tag': 'v9.9.9'}, {'name': 'wrong'}):
        r2_server.store[receipt_key] = (json.dumps({**receipt, **changed}).encode(), '"e"')
        with pytest.raises(ValueError, match='identity') as error:
            fetch(tmp_path / 'wrong')
        assert not isinstance(error.value, handoff.MissingReceipt)

    for bad_path in ("../outside", "C:/outside", "dir\\outside", "a//b", "./a", "a%2fb", "a:b", "a?b", "a#b", '', '/abs.msix', 'NUL'):
        bad = copy.deepcopy(receipt)
        bad["files"][0]["path"] = bad_path
        r2_server.store[receipt_key] = (json.dumps(bad).encode(), '"e"')
        r2_server.requests.clear()
        with pytest.raises(ValueError, match="path"):
            fetch(tmp_path / 'unsafe')
        assert len(r2_server.requests) == 1
    bad = copy.deepcopy(receipt)
    bad['files'] += [{**bad['files'][0], 'path': bad['files'][0]['path'].upper()}]
    r2_server.store[receipt_key] = (json.dumps(bad).encode(), '"e"')
    with pytest.raises(ValueError, match='Duplicate'):
        fetch(tmp_path / 'duplicate')
    r2_server.store[receipt_key] = (json.dumps(receipt).encode(), '"e"')
    r2_server.store[prefix + 'app.msix'] = (b"different bytes", '"e"')
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch(downloaded)
    assert (downloaded / "app.msix").read_bytes() == data


@pytest.mark.parametrize('commit_only', [False, True])
def test_failed_stage_never_publishes_a_receipt_or_channel(tmp_path, r2_server, commit_only):
    tag, commit = "v1.2.3", "a" * 40
    (tmp_path / "one.msix").write_bytes(b"first")
    (tmp_path / "two.msix").write_bytes(b"second")
    prefix = f'releases/commit/{commit}/' if commit_only else f'releases/tag/{tag}/'
    identity = ['--commit-build', commit] if commit_only else ['--tag', tag, '--commit', commit]
    r2_server.fail_put = prefix + 'two.msix'
    with pytest.raises(r2.R2RequestError):
        handoff.main(['stage', *identity, '--name', 'win32-x64', '--root', str(tmp_path), '--include', '*.msix'])
    assert set(r2_server.store) == {prefix + 'one.msix'}
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="No files"):
        handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.zip"])
    assert r2_server.requests == []
    with pytest.raises(ValueError, match="identity"):
        handoff.stage("../bad", commit, "win32-x64", tmp_path, ["*.msix"])
    assert r2_server.requests == []


@pytest.mark.parametrize("commit_only", [False, True])
def test_shared_receipt_files_download_once_and_conflicts_leave_targets_intact(tmp_path, r2_server, commit_only):
    from scripts.releases import handoff

    commit, tag = "a" * 40, "v1.2.3"
    artifact = tmp_path / "shared.bin"
    artifact.write_bytes(b"same shared bytes")
    if commit_only:
        stage = lambda name: handoff.stage_commit_build(commit, name, tmp_path, ["*.bin"])
        fetch = lambda target: handoff.fetch_commit_build(commit, ["one", "two"], target)
        prefix = r2.commit_prefix_for(commit)
    else:
        stage = lambda name: handoff.stage(tag, commit, name, tmp_path, ["*.bin"])
        fetch = lambda target: handoff.fetch(tag, commit, ["one", "two"], target)
        prefix = r2.staging_key_for(tag, "")
    stage("one")
    stage("two")
    r2_server.requests.clear()
    target = tmp_path / "download"
    fetch(target)
    assert (target / "shared.bin").read_bytes() == artifact.read_bytes()
    gets = [url for method, url, _ in r2_server.requests if method == "GET"]
    assert sum(url.endswith(prefix + "shared.bin") for url in gets) == 1

    receipt_key = prefix + "handoff-two.json"
    receipt = json.loads(r2_server.store[receipt_key][0])
    receipt["files"][0]["sha256"] = "0" * 64
    r2_server.store[receipt_key] = (json.dumps(receipt).encode(), '"changed"')
    before = {file.name: file.read_bytes() for file in target.iterdir()}
    with pytest.raises(ValueError, match="Conflicting"):
        fetch(target)
    assert {file.name: file.read_bytes() for file in target.iterdir()} == before


def test_commit_identity_and_simultaneous_namespaces(tmp_path, r2_server):
    commit = 'a' * 40
    for bad in ('abc', commit[:39], 'g' * 40, '', None, ' ' * 40):
        with pytest.raises(ValueError):
            handoff.validate_commit_identity(bad, 'win32-x64')
    with pytest.raises(ValueError):
        handoff.validate_commit_identity(commit, 'Bad Name')
    (tmp_path / 'app.msix').write_bytes(b'commit bytes')
    args = ['stage', '--commit-build', commit, '--name', 'win32-x64', '--root', str(tmp_path), '--include', '*.msix']
    for conflict in (['--commit', 'b' * 40], ['--tag', 'v1.2.3']):
        with pytest.raises(SystemExit):
            handoff.main([*args, *conflict])
    assert r2_server.requests == []
    handoff.main(args)
    (tmp_path / 'app.msix').write_bytes(b'tag bytes')
    handoff.stage('v1.2.3', commit, 'win32-x64', tmp_path, ['*.msix'])
    for identity, data in ((['--commit-build', commit], b'commit bytes'),
                           (['--tag', 'v1.2.3', '--commit', commit], b'tag bytes')):
        handoff.main(['fetch', *identity, '--name', 'win32-x64', '--root', str(tmp_path / 'out')])
        assert (tmp_path / 'out/app.msix').read_bytes() == data
    with pytest.raises(handoff.MissingReceipt):
        handoff.read_commit_receipt('b' * 40, 'win32-x64')
    r2_server.store[f'releases/commit/{commit}/handoff-win32-x64.json'] = (b'not-json', '"e"')
    with pytest.raises(json.JSONDecodeError):
        handoff.read_commit_receipt(commit, 'win32-x64')


@pytest.mark.parametrize('tag', ['v1.2.3', None])
def test_public_handoff_downloads_exact_staged_bytes_without_credentials(tmp_path, monkeypatch, r2_server, tag):
    import os
    from pathlib import Path
    import subprocess
    import sys

    commit = 'a' * 40
    identity = ['--tag', tag, '--commit', commit] if tag else ['--commit-build', commit]
    prefix = f'releases/tag/{tag}/' if tag else f'releases/commit/{commit}/'
    built = tmp_path / 'built'
    built.mkdir()
    payload = b'public artifact transport fixture\n' * 90000
    (built / 'app x64.msix').write_bytes(payload)
    handoff.main(['stage', *identity, '--name', 'win32-x64', '--root', str(built), '--include', '*.msix'])
    for key in ('CLOUDFLARE_R2_ACCOUNT_ID', 'CLOUDFLARE_R2_ACCESS_KEY_ID',
                'CLOUDFLARE_R2_SECRET_ACCESS_KEY', 'CLOUDFLARE_R2_BUCKET'):
        monkeypatch.delenv(key)
    r2_server.requests.clear()
    download = tmp_path / 'downloaded'
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    args = ['fetch', *identity, '--public-base', base, '--name', 'win32-x64',
            '--root', str(download), '--include', '*.msix']
    handoff.main(args)
    assert (download / 'app x64.msix').read_bytes() == payload
    receipt_file = download / 'handoff-win32-x64.json'
    original_receipt = receipt_file.read_bytes()
    assert json.loads(original_receipt)['commit'] == commit
    assert all(method == 'GET' and not any(k.lower() == 'authorization' for k in headers)
               for method, _, headers in r2_server.requests)
    assert any(path.endswith('/app%20x64.msix') for _, path, _ in r2_server.requests)
    result = subprocess.run([sys.executable, '-m', 'scripts.releases.handoff', *args],
                            cwd=Path(__file__).resolve().parents[2], env=os.environ,
                            capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (download / 'app x64.msix').read_bytes() == payload
    # A bad CDN object cannot replace a previously verified local download.
    r2_server.store[prefix + 'app x64.msix'] = (b'corrupt', '"changed"')
    with pytest.raises(ValueError, match='checksum mismatch'):
        handoff.main(args)
    assert (download / 'app x64.msix').read_bytes() == payload
    assert receipt_file.read_bytes() == original_receipt
    assert sorted(p.name for p in download.iterdir()) == ['app x64.msix', receipt_file.name]
    bad = json.loads(original_receipt)
    bad['commit'] = 'b' * 40
    r2_server.store[prefix + receipt_file.name] = (json.dumps(bad).encode(), '"bad"')
    r2_server.requests.clear()
    with pytest.raises(ValueError, match='identity'):
        handoff.main(args)
    assert len(r2_server.requests) == 1
    bad = json.loads(original_receipt)
    bad['files'][0]['path'] = '../outside.msix'
    r2_server.store[prefix + receipt_file.name] = (json.dumps(bad).encode(), '"bad"')
    r2_server.requests.clear()
    with pytest.raises(ValueError, match='path'):
        handoff.main(args)
    assert len(r2_server.requests) == 1
    assert not (tmp_path / 'outside.msix').exists()


def test_public_handoff_rejects_unsafe_origins_paths_and_redirects(tmp_path, r2_server):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    commit = 'a' * 40
    args = ['fetch', '--commit-build', commit, '--name', 'win32-x64', '--root', str(tmp_path / 'out')]
    for base in ('http://cdn.example', 'https://user:password@cdn.example', 'file:///tmp',
                 'https://cdn.example/?token=private', 'https://cdn.example/#fragment',
                 'https://cdn.example/../escape', 'https://cdn.example/%2e%2e/escape'):
        with pytest.raises(ValueError, match='public'):
            handoff.main([*args, '--public-base', base])
    assert r2_server.requests == []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header('Location', f'http://127.0.0.1:{r2_server.server_port}/hermes-releases/unwanted')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(r2.R2RequestError) as error:
            handoff.main([*args, '--public-base', f'http://127.0.0.1:{server.server_port}'])
        assert error.value.status == 302
        assert r2_server.requests == []
        assert not (tmp_path / 'out').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
