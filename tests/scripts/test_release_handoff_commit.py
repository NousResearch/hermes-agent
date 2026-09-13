"""Commit-only handoff receipts (schema 2) bind files to a bare commit."""
import copy
import json
import os
import unittest.mock as mock

import pytest

from scripts.releases import r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


COMMIT = "b" * 40


def test_commit_key_namespace_never_overlaps_tag_or_channel_dirs():
    assert r2.commit_key_for(COMMIT, "app.msix") == f"releases/commit/{COMMIT}/app.msix"
    assert r2.commit_prefix_for(COMMIT) == f"releases/commit/{COMMIT}/"
    for bad in ("abc", COMMIT[:39], "g" * 40, "", None, " " * 40):
        with pytest.raises(ValueError):
            r2.commit_key_for(bad, "app.msix")
    # Traversal / absolute / backslash paths stay forbidden...
    for bad_name in ("../outside", "a/../../b", "/abs.msix", "C:\\x.msix", "",
                     "a//b", "./a", "C:/b", "a%2fb", "a?b", "a#b", "NUL", "a/../b"):
        with pytest.raises(ValueError):
            r2.commit_key_for(COMMIT, bad_name)
    # ...but REAL nested artifact paths (the Termux receipt shape) fetch.
    assert (r2.commit_key_for(COMMIT, "deb/pool/hermes_0.28.0_aarch64.deb")
            == f"releases/commit/{COMMIT}/deb/pool/hermes_0.28.0_aarch64.deb")
    key = r2.commit_key_for(COMMIT, "app.msix")
    assert not key.startswith("releases/tag/")
    assert not key.startswith(("releases/win32/", "releases/darwin/", "releases/termux/"))


def test_commit_stage_fetch_roundtrip_is_receipt_last_and_tag_free(tmp_path, r2_server):
    from scripts.releases import handoff

    root = tmp_path / "built"
    root.mkdir()
    data = b"commit build transport fixture\n" * 50000
    (root / "HermesBundled-0.28.0-win-x64.msix").write_bytes(data)
    # No --commit flag: the commit-build identity comes ONLY from
    # --commit-build. --commit never defaults from ambient $GITHUB_SHA.
    handoff.main(["stage", "--commit-build", COMMIT, "--name", "win32-x64",
                  "--root", str(root), "--include", "*.msix"])
    receipt_key = f"releases/commit/{COMMIT}/handoff-win32-x64.json"
    receipt = json.loads(r2_server.store[receipt_key][0])
    assert receipt["schema"] == 2 and receipt["commit"] == COMMIT
    assert "tag" not in receipt
    assert receipt["files"][0]["sha256"] == r2.file_sha256(root / "HermesBundled-0.28.0-win-x64.msix")
    assert set(r2_server.store) <= {f"releases/commit/{COMMIT}/{name}" for name in
                                    ("HermesBundled-0.28.0-win-x64.msix", "handoff-win32-x64.json")}
    puts = [path for method, path, _ in r2_server.requests if method == "PUT"]
    assert puts[-1].endswith(receipt_key), "receipt must be the completion marker"
    assert all(headers.get("If-None-Match") == "*"
               for method, _, headers in r2_server.requests if method == "PUT")

    downloaded = tmp_path / "downloaded"
    handoff.main(["fetch", "--commit-build", COMMIT,
                  "--name", "win32-x64", "--root", str(downloaded)])
    assert (downloaded / "HermesBundled-0.28.0-win-x64.msix").read_bytes() == data

    # Same-SHA retry with identical bytes succeeds (verified-conflict path).
    handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix"])
    # Conflicting different bytes must fail, never delete the existing object.
    (root / "HermesBundled-0.28.0-win-x64.msix").write_bytes(b"different")
    with pytest.raises(Exception):
        handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix"])
    assert r2_server.store[f"releases/commit/{COMMIT}/HermesBundled-0.28.0-win-x64.msix"][0] == data


def test_commit_stage_fetch_nested_paths_roundtrip(tmp_path, r2_server):
    """The Termux leg stages deb/*.deb (nested receipt paths). The fetch
    must rebuild the same nested layout — the old commit_key_for forbade
    any slash, so the workflow's own stage shape was unfetchable."""
    from scripts.releases import handoff

    root = tmp_path / "termux-build"
    (root / "deb" / "pool").mkdir(parents=True)
    deb = root / "deb" / "pool" / "hermes_0.28.0_aarch64.deb"
    deb.write_bytes(b"!<arch>\ndeb-bytes")
    handoff.main(["stage", "--commit-build", COMMIT, "--name", "termux",
                  "--root", str(root), "--include", "deb/**/*.deb"])
    receipt = json.loads(r2_server.store[f"releases/commit/{COMMIT}/handoff-termux.json"][0])
    assert receipt["files"][0]["path"] == "deb/pool/hermes_0.28.0_aarch64.deb"

    downloaded = tmp_path / "out"
    handoff.main(["fetch", "--commit-build", COMMIT, "--name", "termux", "--root", str(downloaded)])
    assert (downloaded / "deb" / "pool" / "hermes_0.28.0_aarch64.deb").read_bytes() == b"!<arch>\ndeb-bytes"


def test_commit_cli_rejects_conflicting_commit_value(tmp_path, r2_server):
    """--commit is a SEPARATE explicit flag: it never defaults from ambient
    $GITHUB_SHA (a feature-branch run must not inherit a conflicting SHA
    that would parser.error against --commit-build), and when passed it
    must EQUAL --commit-build."""
    from scripts.releases import handoff

    root = tmp_path / "built"
    root.mkdir()
    (root / "a.msix").write_bytes(b"x")
    # Disagreeing explicit flags are a hard CLI error, never silently
    # resolved (the old draft accepted and dropped the override).
    with pytest.raises(SystemExit):
        handoff.main(["stage", "--commit-build", COMMIT, "--commit", "c" * 40,
                      "--name", "win32-x64", "--root", str(root), "--include", "*.msix"])
    # Without --commit, --commit-build alone stages fine even though the
    # inherited environment carries an unrelated GITHUB_SHA (the old draft
    # would parser.error here — every commit builder exports GITHUB_SHA).
    with mock.patch.dict(os.environ, {"GITHUB_SHA": "c" * 40}):
        handoff.main(["stage", "--commit-build", COMMIT,
                      "--name", "win32-x64", "--root", str(root), "--include", "*.msix"])
    # Equal values are accepted (they name the same commit).
    handoff.main(["stage", "--commit-build", COMMIT, "--commit", COMMIT,
                  "--name", "win32-x64", "--root", str(root), "--include", "*.msix"])


def test_commit_and_tag_namespaces_stay_isolated(tmp_path, r2_server):
    from scripts.releases import handoff

    # A tag staging and a commit staging of the same artifact name must
    # land in disjoint prefixes — neither can read or overwrite the other.
    root = tmp_path / "built"
    root.mkdir()
    (root / "a.msix").write_bytes(b"commit bytes")
    handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix"])
    tag_root = tmp_path / "tagged"
    tag_root.mkdir()
    (tag_root / "a.msix").write_bytes(b"tag bytes")
    handoff.stage("v0.28.0", COMMIT, "win32-x64", tag_root, ["*.msix"])
    assert r2_server.store[f"releases/commit/{COMMIT}/a.msix"][0] == b"commit bytes"
    assert r2_server.store["releases/tag/v0.28.0/a.msix"][0] == b"tag bytes"
    # The schema-1 tag receipt still carries tag+commit; the schema-2
    # commit receipt has no tag anywhere.
    tag_receipt = json.loads(r2_server.store["releases/tag/v0.28.0/handoff-win32-x64.json"][0])
    assert tag_receipt["schema"] == 1 and tag_receipt["tag"] == "v0.28.0" and tag_receipt["commit"] == COMMIT
    commit_receipt = json.loads(r2_server.store[f"releases/commit/{COMMIT}/handoff-win32-x64.json"][0])
    assert commit_receipt["schema"] == 2 and "tag" not in commit_receipt


def test_interrupted_upload_leaves_no_commit_receipt(tmp_path, r2_server):
    """Receipt-last means an interrupted build publishes NO receipt, so the
    summary can never mistake a half-staged leg for a completed one."""
    from scripts.releases import handoff

    root = tmp_path / "built"
    root.mkdir()
    (root / "a.msix").write_bytes(b"x")
    (root / "b.bin").write_bytes(b"y")
    # Second include pattern's upload dies mid-leg: the receipt PUT (last)
    # never happens, only artifacts exist.
    real_put = r2.put
    calls = {"n": 0}

    def flaky_put(**kwargs):
        calls["n"] += 1
        if "b.bin" in kwargs["key"]:
            raise OSError("connection reset mid-upload")
        return real_put(**kwargs)

    with mock.patch.object(r2, "put", side_effect=lambda **kw: flaky_put(**kw)):
        with pytest.raises(OSError):
            handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix", "*.bin"])
    assert f"releases/commit/{COMMIT}/a.msix" in r2_server.store
    assert f"releases/commit/{COMMIT}/b.bin" not in r2_server.store
    assert f"releases/commit/{COMMIT}/handoff-win32-x64.json" not in r2_server.store


def test_commit_receipt_identity_rejects_bad_sha_tag_sneak_and_traversal(tmp_path, r2_server):
    from scripts.releases import handoff

    for bad in ("abc", COMMIT[:39], "G" * 40, ""):
        with pytest.raises(ValueError):
            handoff.validate_commit_identity(bad, "win32-x64")
    with pytest.raises(ValueError):
        handoff.validate_commit_identity(COMMIT, "Bad Name")
    receipt = {"schema": 2, "commit": COMMIT, "name": "win32-x64",
               "files": [{"path": "a.msix", "size": 1, "sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="identity mismatch"):
        handoff.validate_commit_receipt({**receipt, "tag": "v1.2.3"}, COMMIT, "win32-x64")
    with pytest.raises(ValueError, match="identity mismatch"):
        handoff.validate_commit_receipt({**receipt, "schema": 1}, COMMIT, "win32-x64")
    bad_path = copy.deepcopy(receipt)
    bad_path["files"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="path"):
        handoff.validate_commit_receipt(bad_path, COMMIT, "win32-x64")

    # A fetch against a mismatched commit is refused.
    root = tmp_path / "built"
    root.mkdir()
    (root / "a.msix").write_bytes(b"x")
    handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix"])
    with pytest.raises(handoff.MissingReceipt):
        handoff.read_commit_receipt("c" * 40, "win32-x64")


def test_corrupt_commit_receipt_is_not_reported_as_missing(tmp_path, r2_server):
    """A present-but-invalid receipt is a distinct failure from a MISSING
    one: the summary renderer must crash loudly on a corrupt receipt, not
    render the leg as an ordinary "leg incomplete" row."""
    from scripts.releases import handoff

    root = tmp_path / "built"
    root.mkdir()
    (root / "a.msix").write_bytes(b"x")
    handoff.stage_commit_build(COMMIT, "win32-x64", root, ["*.msix"])

    # Corrupt the stored receipt (wrong schema: a validation failure).
    receipt_key = f"releases/commit/{COMMIT}/handoff-win32-x64.json"
    original = json.loads(r2_server.store[receipt_key][0])
    corrupt = copy.deepcopy(original)
    corrupt["schema"] = 1
    r2_server.store[receipt_key] = (json.dumps(corrupt).encode(), '"e"')
    with pytest.raises(ValueError) as excinfo:
        handoff.read_commit_receipt(COMMIT, "win32-x64")
    assert not isinstance(excinfo.value, handoff.MissingReceipt)

    # A mismatched-commit receipt is likewise NOT a MissingReceipt.
    corrupt2 = copy.deepcopy(original)
    corrupt2["commit"] = "c" * 40
    r2_server.store[receipt_key] = (json.dumps(corrupt2).encode(), '"e"')
    with pytest.raises(ValueError) as excinfo2:
        handoff.read_commit_receipt(COMMIT, "win32-x64")
    assert not isinstance(excinfo2.value, handoff.MissingReceipt)

    # Restore; then a genuinely absent receipt raises MissingReceipt.
    r2_server.store[receipt_key] = (json.dumps(original).encode(), '"e"')
    with pytest.raises(handoff.MissingReceipt):
        handoff.read_commit_receipt("d" * 40, "win32-x64")


def test_real_cli_stage_and_summary_run_as_subprocesses(tmp_path):
    """Run the actual CLI parsers and signed transport against loopback HTTP."""
    import re
    import subprocess
    import sys
    import threading
    from http.server import ThreadingHTTPServer
    from pathlib import Path
    from urllib.request import urlopen

    repo = Path(__file__).resolve().parents[2]
    from tests.scripts.test_release_r2 import _R2StubHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _R2StubHandler)
    server.store = {}
    server.requests = []

    def listing_xml():
        parts = ["<?xml version='1.0'?><ListBucketResult><IsTruncated>false</IsTruncated>"]
        for key, (body, _etag) in server.store.items():
            parts.append(f"<Contents><Key>{key}</Key><LastModified>2026-08-01T00:00:00Z</LastModified></Contents>")
        parts.append("</ListBucketResult>")
        return "".join(parts)

    server.listing_xml = listing_xml
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update({
            "CLOUDFLARE_R2_ACCOUNT_ID": "loopback",
            "CLOUDFLARE_R2_ACCESS_KEY_ID": "test-inert",
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY": "test-inert",
            "CLOUDFLARE_R2_BUCKET": "hermes-releases",
            "GITHUB_SHA": "c" * 40,  # ambient, unrelated — must be ignored
        })
        base = f"http://127.0.0.1:{server.server_port}"
        driver = tmp_path / "cli_driver.py"
        driver.write_text(
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            "from scripts.releases import r2\n"
            f"r2.s3_endpoint = lambda account: {base!r}\n"
            "mode = sys.argv.pop(1)\n"
            "if mode == 'handoff':\n"
            "    runpy.run_module('scripts.releases.handoff', run_name='__main__')\n"
            "else:\n"
            f"    runpy.run_path({str(repo / 'scripts/render-builds-table.py')!r}, run_name='__main__')\n",
            encoding="utf-8",
        )
        build_root = tmp_path / "built"
        build_root.mkdir()
        (build_root / "HermesBundled-0.28.0-win-x64.msix").write_bytes(b"cli-bytes")

        stage = subprocess.run(
            [sys.executable, str(driver), "handoff", "stage",
             "--commit-build", COMMIT, "--name", "win32-x64",
             "--root", str(build_root), "--include", "*.msix"],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert stage.returncode == 0, stage.stdout + stage.stderr

        summary = tmp_path / "summary.md"
        render = subprocess.run(
            [sys.executable, str(driver), "summary",
             "--summary-commit", COMMIT, "--summary-out", str(summary),
             "--r2-base-url", base + "/hermes-releases"],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert render.returncode == 0, render.stdout + render.stderr
        text = summary.read_text(encoding="utf-8")
        # The staged leg is receipt+object bound -> exactly one ✅ row with
        # a link; other enabled legs are Not built, and Linux is Disabled.
        built = [line for line in text.splitlines() if "✅ Built" in line]
        assert len(built) == 1, text
        assert "HermesBundled-0.28.0-win-x64.msix" in built[0]
        url = re.search(r"\]\((http[^)]+)\)", built[0]).group(1)
        assert f"/releases/commit/{COMMIT}/" in url
        with urlopen(url, timeout=5) as response:
            assert response.read() == b"cli-bytes"
        assert "Windows universal bundle (MSIXBUNDLE)" in text
        assert "Store" not in text
        assert "Linux x64" in text and "Linux ARM64" in text
        assert all(key.startswith(f"releases/commit/{COMMIT}/") for key in server.store)

        receipt_key = f"releases/commit/{COMMIT}/handoff-win32-x64.json"
        server.store[receipt_key] = (b"not-json", '"invalid"')
        failed = subprocess.run(
            [sys.executable, str(driver), "summary", "--summary-commit", COMMIT,
             "--summary-out", str(summary), "--r2-base-url", base + "/hermes-releases"],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert failed.returncode != 0
        assert summary.read_text(encoding="utf-8") == text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
