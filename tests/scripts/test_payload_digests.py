"""Packaged facts describe final tool bytes without changing their identity."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from pm.lock import Facts, Lockfile
from pm.store import current_target, tree_digest
from scripts.bundles import payload


def _payload(tmp_path):
    root = tmp_path / "payload"
    store = root / "tools"
    entries = {"python": "python", "uv": "uv"}
    lock = Lockfile(tmp_path / "lock.json")
    for name, entry in entries.items():
        directory = store / entry
        directory.mkdir(parents=True)
        (directory / "tool").write_bytes(b"staging bytes")
        lock.set_pin(name, "1.0", {current_target(): {
            "url": "https://example.invalid/tool.zip", "sha256": "a" * 64,
        }})
    lock.save()
    payload.record_tools(root, lock.path, current_target(), entries)
    Facts(store / "facts.json").record_state("venv", "selection", ["web"])
    return root


def _rehash(root, cwd):
    return subprocess.run(
        [sys.executable, str(Path(payload.__file__)), "rehash", str(root)],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def test_rehash_records_all_changed_tools_and_preserves_identity(tmp_path):
    root = _payload(tmp_path)
    path = root / "tools" / "facts.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    (root / "tools/python/tool").write_bytes(b"final python bytes")
    (root / "tools/uv/tool").write_bytes(b"final uv bytes")
    result = _rehash(root, tmp_path)
    assert result.returncode == 0, result.stderr
    after = json.loads(path.read_text(encoding="utf-8"))
    for name in ("python", "uv"):
        expected = dict(before["packages"][name])
        expected["digest"] = tree_digest(root / "tools" / expected["entry"])
        assert after["packages"][name] == expected
        assert expected["digest"] != before["packages"][name]["digest"]
    assert after["packages"]["venv"] == before["packages"]["venv"]
    saved = path.read_bytes()
    assert _rehash(root, tmp_path).returncode == 0
    assert path.read_bytes() == saved


def test_invalid_tool_evidence_never_partially_rewrites_facts(tmp_path):
    root = _payload(tmp_path)
    path = root / "tools" / "facts.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    (root / "tools/python/tool").write_bytes(b"final python bytes")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept").write_bytes(b"foreign")
    for entry in ("missing", "../outside", str(outside)):
        invalid = copy.deepcopy(before)
        invalid["packages"]["uv"]["entry"] = entry
        path.write_text(json.dumps(invalid), encoding="utf-8")
        saved = path.read_bytes()
        assert _rehash(root, tmp_path).returncode != 0
        assert path.read_bytes() == saved
        assert (outside / "kept").read_bytes() == b"foreign"
    for raw in (b"not JSON", b'{"schema":1,"packages":{}}'):
        path.write_bytes(raw)
        assert _rehash(root, tmp_path).returncode != 0
        assert path.read_bytes() == raw
    path.unlink()
    assert _rehash(root, tmp_path).returncode != 0
    assert not path.exists()
