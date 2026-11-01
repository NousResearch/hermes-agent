"""Release handoffs use exact R2 bytes without advancing a channel."""
import copy
import json
from urllib.parse import unquote

import pytest

from scripts.releases import handoff, r2
from tests.scripts.test_release_r2 import r2_server


@pytest.mark.parametrize('tag', ['v1.2.3', 'canary.20260908T232538Z-v1.2.3', 'rc.1-v1.2.3',
                                 'commit.20260908T232538Z-v1.2.3', 'channel.20260908T232538Z-v1.2.3'])
def test_stage_and_fetch_bind_tag_commit_and_files_without_feed_writes(tmp_path, monkeypatch, r2_server, tag):
    commit = "a" * 40
    identity = ['--tag', tag, '--commit', commit]
    prefix = f'releases/tag/{tag}/'
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
    assert receipt['schema'] == 1 and receipt['commit'] == commit
    assert receipt['tag'] == tag
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
    handoff.main(['stage', *identity, '--name', 'win32-x64',
                  '--root', str(root), '--include', '*.msix', '--include', 'metadata-*.json', '--include', 'apt/**/*'])
    handoff.main(['fetch', *identity, '--name', 'win32-x64', '--root', str(downloaded), '--include', 'apt/*'])
    assert (downloaded / nested.relative_to(root)).read_bytes() == nested.read_bytes()

    fetch = lambda target: handoff.main(['fetch', *identity, '--name', 'win32-x64', '--root', str(target)])
    for changed in ({'schema': 2}, {'commit': 'b' * 40}, {'tag': 'v9.9.9'}, {'name': 'wrong'}):
        r2_server.store[receipt_key] = (json.dumps({**receipt, **changed}).encode(), '"e"')
        with pytest.raises(ValueError, match='identity'):
            fetch(tmp_path / 'wrong')

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


def test_failed_stage_never_publishes_a_receipt_or_channel(tmp_path, request):
    server = request.getfixturevalue("r2_server")
    tag, commit = "v1.2.3", "a" * 40
    (tmp_path / "one.msix").write_bytes(b"first")
    (tmp_path / "two.msix").write_bytes(b"second")
    prefix = f'releases/tag/{tag}/'
    identity = ['--tag', tag, '--commit', commit]
    server.fail_put = prefix + 'two.msix'
    with pytest.raises(r2.R2RequestError):
        handoff.main(['stage', *identity, '--name', 'win32-x64', '--root', str(tmp_path), '--include', '*.msix'])
    assert set(server.store) == {prefix + 'one.msix'}
    server.requests.clear()
    with pytest.raises(ValueError, match="No files"):
        handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.zip"])
    assert server.requests == []
    with pytest.raises(ValueError, match="identity"):
        handoff.stage("../bad", commit, "win32-x64", tmp_path, ["*.msix"])
    assert server.requests == []


def test_shared_receipt_files_download_once_and_conflicts_leave_targets_intact(tmp_path, request):
    server = request.getfixturevalue("r2_server")
    from scripts.releases import handoff

    commit, tag = "a" * 40, "v1.2.3"
    artifact = tmp_path / "shared.bin"
    artifact.write_bytes(b"same shared bytes")
    stage = lambda name: handoff.stage(tag, commit, name, tmp_path, ["*.bin"])
    fetch = lambda target: handoff.fetch(tag, commit, ["one", "two"], target)
    prefix = r2.staging_key_for(tag, "")
    stage("one")
    stage("two")
    server.requests.clear()
    target = tmp_path / "download"
    fetch(target)
    assert (target / "shared.bin").read_bytes() == artifact.read_bytes()
    gets = [url for method, url, _ in server.requests if method == "GET"]
    assert sum(url.endswith(prefix + "shared.bin") for url in gets) == 1

    receipt_key = prefix + "handoff-two.json"
    receipt = json.loads(server.store[receipt_key][0])
    receipt["files"][0]["sha256"] = "0" * 64
    server.store[receipt_key] = (json.dumps(receipt).encode(), '"changed"')
    before = {file.name: file.read_bytes() for file in target.iterdir()}
    with pytest.raises(ValueError, match="Conflicting"):
        fetch(target)
    assert {file.name: file.read_bytes() for file in target.iterdir()} == before


def test_public_handoff_downloads_exact_staged_bytes_without_credentials(tmp_path, monkeypatch, request):
    server = request.getfixturevalue("r2_server")
    import os
    from pathlib import Path
    import subprocess
    import sys

    commit, tag = 'a' * 40, 'v1.2.3'
    identity = ['--tag', tag, '--commit', commit]
    prefix = f'releases/tag/{tag}/'
    built = tmp_path / 'built'
    built.mkdir()
    payload = b'public artifact transport fixture\n' * 90000
    (built / 'app x64.msix').write_bytes(payload)
    handoff.main(['stage', *identity, '--name', 'win32-x64', '--root', str(built), '--include', '*.msix'])
    for key in ('CLOUDFLARE_R2_ACCOUNT_ID', 'CLOUDFLARE_R2_ACCESS_KEY_ID',
                'CLOUDFLARE_R2_SECRET_ACCESS_KEY', 'CLOUDFLARE_R2_BUCKET'):
        monkeypatch.delenv(key)
    server.requests.clear()
    download = tmp_path / 'downloaded'
    base = f'http://127.0.0.1:{server.server_port}/hermes-releases'
    args = ['fetch', *identity, '--public-base', base, '--name', 'win32-x64',
            '--root', str(download), '--include', '*.msix']
    handoff.main(args)
    assert (download / 'app x64.msix').read_bytes() == payload
    receipt_file = download / 'handoff-win32-x64.json'
    original_receipt = receipt_file.read_bytes()
    assert json.loads(original_receipt)['commit'] == commit
    assert all(method == 'GET' and not any(k.lower() == 'authorization' for k in headers)
               for method, _, headers in server.requests)
    assert any(path.endswith('/app%20x64.msix') for _, path, _ in server.requests)
    result = subprocess.run([sys.executable, '-m', 'scripts.releases.handoff', *args],
                            cwd=Path(__file__).resolve().parents[2], env=os.environ,
                            capture_output=True, text=True, encoding='utf-8', timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (download / 'app x64.msix').read_bytes() == payload
    # A bad CDN object cannot replace a previously verified local download.
    server.store[prefix + 'app x64.msix'] = (b'corrupt', '"changed"')
    with pytest.raises(ValueError, match='checksum mismatch'):
        handoff.main(args)
    assert (download / 'app x64.msix').read_bytes() == payload
    assert receipt_file.read_bytes() == original_receipt
    assert sorted(p.name for p in download.iterdir()) == ['app x64.msix', receipt_file.name]
    bad = json.loads(original_receipt)
    bad['commit'] = 'b' * 40
    server.store[prefix + receipt_file.name] = (json.dumps(bad).encode(), '"bad"')
    server.requests.clear()
    with pytest.raises(ValueError, match='identity'):
        handoff.main(args)
    assert len(server.requests) == 1
    bad = json.loads(original_receipt)
    bad['files'][0]['path'] = '../outside.msix'
    server.store[prefix + receipt_file.name] = (json.dumps(bad).encode(), '"bad"')
    server.requests.clear()
    with pytest.raises(ValueError, match='path'):
        handoff.main(args)
    assert len(server.requests) == 1
    assert not (tmp_path / 'outside.msix').exists()


def test_public_handoff_rejects_unsafe_origins_paths_and_redirects(tmp_path, r2_server):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    commit = 'a' * 40
    args = ['fetch', '--tag', 'v1.2.3', '--commit', commit, '--name', 'win32-x64', '--root', str(tmp_path / 'out')]
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


def test_a_commit_claim_stages_under_its_claim_despite_its_commit_identity_env(tmp_path, monkeypatch):
    # A commit claim's legs export HERMES_BUILD_COMMIT (the Commit<sha7> identity) beside the claim
    # tag; the handoff CLI stages under the claim archive (there is no tagless commit mode).
    commit, tag = "a" * 40, "commit.20261026T180000Z-v1.2.3"
    monkeypatch.setenv("HERMES_BUILD_COMMIT", commit)
    calls = []
    monkeypatch.setattr(handoff, "stage", lambda *args: calls.append(("tag", *args)))
    handoff.main(["stage", "--tag", tag, "--commit", commit, "--name", "win32-x64",
                  "--root", str(tmp_path), "--include", "*.msix"])
    assert calls == [("tag", tag, commit, "win32-x64", tmp_path, ["*.msix"])]
