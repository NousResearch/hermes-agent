"""Behaviour of the p5js skill's headless exporter launch policy (issue #145361)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "skills" / "creative" / "p5js" / "scripts" / "export-frames.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(expr: str) -> object:
    code = f"const m = require({json.dumps(str(SCRIPT))}); process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(out.stdout)


def test_unrestricted_host_keeps_chrome_protections():
    args = _node("m.chromeArgs({isRoot: false, inDocker: false, apparmorUserns: false})")
    weakening = {"--no-sandbox", "--disable-setuid-sandbox", "--disable-web-security"}
    weakening.add("--allow-file-access-from-files")
    assert weakening.isdisjoint(args)


@pytest.mark.parametrize("fact", ["isRoot", "inDocker", "apparmorUserns"])
def test_sandbox_bypass_only_on_hosts_that_need_it(fact):
    facts = {"isRoot": False, "inDocker": False, "apparmorUserns": False, fact: True}
    args = _node(f"m.chromeArgs({json.dumps(facts)})")
    assert "--no-sandbox" in args
    assert "--disable-web-security" not in args


def test_local_assets_resolve_only_inside_sketch_dir(tmp_path):
    sketch_dir = tmp_path / "sketch"
    (sketch_dir / "assets").mkdir(parents=True)
    (sketch_dir / "sketch.html").write_text("<html></html>")
    (sketch_dir / "assets" / "a b.png").write_bytes(b"x")
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    (sketch_dir / "link.txt").symlink_to(outside)

    def resolve(url_path: str):
        return _node(f"m.resolveLocalFile({json.dumps(str(sketch_dir))}, m.SKETCH_ORIGIN + {json.dumps(url_path)})")

    assert resolve("/sketch.html") == str((sketch_dir / "sketch.html").resolve())
    assert resolve("/assets/a%20b.png") == str((sketch_dir / "assets" / "a b.png").resolve())
    for escaping in ("/%2e%2e/outside.txt", "/..%2foutside.txt", "/link.txt", "/missing.png", "/"):
        assert resolve(escaping) is None, escaping
    other = _node(f"m.resolveLocalFile({json.dumps(str(sketch_dir))}, 'http://example.com/sketch.html')")
    assert other is None


def test_other_origins_pass_through_untouched(tmp_path):
    out = _node(
        "(() => { const out = {}; m.sketchRequestHandler(" + json.dumps(str(tmp_path)) + ")({"
        " url: () => 'https://cdn.jsdelivr.net/npm/p5/lib/p5.min.js',"
        " continue: () => { out.continued = true; }, respond: (r) => { out.status = r.status; }}); return out; })()"
    )
    assert out == {"continued": True}


def _serve(sketch_dir: Path, url_path: str) -> dict:
    """Run the interception listener on one fake request; return what it answered."""
    req = (
        "(() => { const out = {}; const url = m.SKETCH_ORIGIN + " + json.dumps(url_path) + ";"
        " m.sketchRequestHandler(" + json.dumps(str(sketch_dir)) + ")({url: () => url,"
        " continue: () => { out.continued = true; },"
        " respond: (r) => { out.status = r.status; out.body = String(r.body); }}); return out; })()"
    )
    return _node(req)


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads mode-000 files")
def test_unreadable_asset_fails_only_its_own_request(tmp_path):
    sketch_dir = tmp_path / "sketch"
    sketch_dir.mkdir()
    (sketch_dir / "ok.txt").write_text("fine")
    locked = sketch_dir / "locked.png"
    locked.write_bytes(b"x")
    locked.chmod(0)
    try:
        assert _serve(sketch_dir, "/locked.png") == {"status": 404, "body": "Not found"}
        assert _serve(sketch_dir, "/ok.txt") == {"status": 200, "body": "fine"}
    finally:
        locked.chmod(0o600)
