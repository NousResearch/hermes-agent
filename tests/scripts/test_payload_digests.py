"""Packaged facts describe final tool bytes without changing their identity."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

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
    before = json.loads(path.read_text(encoding="utf-8-sig"))
    (root / "tools/python/tool").write_bytes(b"final python bytes")
    (root / "tools/uv/tool").write_bytes(b"final uv bytes")
    result = _rehash(root, tmp_path)
    assert result.returncode == 0, result.stderr
    after = json.loads(path.read_text(encoding="utf-8-sig"))
    for name in ("python", "uv"):
        expected = dict(before["packages"][name])
        expected["digest"] = tree_digest(root / "tools" / expected["entry"])
        assert after["packages"][name] == expected
        assert expected["digest"] != before["packages"][name]["digest"]
    assert after["packages"]["venv"] == before["packages"]["venv"]
    saved = path.read_bytes()
    assert _rehash(root, tmp_path).returncode == 0
    assert path.read_bytes() == saved


@pytest.mark.platforms("posix")
def test_packaged_metadata_is_shared_but_mutable_writes_stay_private(tmp_path):
    root = _payload(tmp_path)
    path = root / "tools/facts.json"
    # The fixture's final mutable state write deliberately restores 0600.
    assert path.stat().st_mode & 0o777 == 0o600
    entries = {"python": "python", "uv": "uv"}
    payload.record_tools(root, tmp_path / "lock.json", current_target(), entries)
    assert path.stat().st_mode & 0o777 == 0o644
    Facts(path).record_state("venv", "selection", ["web"])
    assert path.stat().st_mode & 0o777 == 0o600
    assert payload.rehash_tools(root) == 2
    assert path.stat().st_mode & 0o777 == 0o644
    # Final native relocation follows application environment creation.
    venv = root / "venv"
    venv.mkdir()
    lock = venv / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o666)
    payload.relativize_links(root)
    assert not lock.exists()
    private = tmp_path / "private/facts.json"
    Facts(private).record_state("venv", "private", [])
    assert private.stat().st_mode & 0o777 == 0o600


@pytest.mark.platforms("posix")
def test_packaging_never_changes_external_metadata_or_build_locks(tmp_path):
    root = _payload(tmp_path)
    outside = tmp_path / "external"
    outside.mkdir()
    private = outside / "facts.json"
    Facts(private).record_state("venv", "private", [])
    before = private.read_bytes()
    facts = root / "tools/facts.json"
    facts.unlink()
    facts.symlink_to(private)
    for operation in (
        lambda: payload.record_tools(root, tmp_path / "lock.json", current_target(), {"uv": "uv"}),
        lambda: payload.rehash_tools(root),
    ):
        with pytest.raises(ValueError, match="symlinked"):
            operation()
        assert private.read_bytes() == before
        assert private.stat().st_mode & 0o777 == 0o600
    lock = outside / ".lock"
    lock.write_bytes(b"external lock")
    lock.chmod(0o600)
    (root / "venv").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        payload.relativize_links(root)
    assert lock.read_bytes() == b"external lock"
    assert lock.stat().st_mode & 0o777 == 0o600


def test_sealing_strips_bom_and_absolute_build_paths(tmp_path):
    root = tmp_path / "payload"
    python = root / "tools/pythön/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter")
    runtime = root / "pm-runtime"
    (runtime / "Lib/site-packages").mkdir(parents=True)
    (runtime / "bin").mkdir()
    (runtime / "Scripts").mkdir()
    cfg = runtime / "pyvenv.cfg"
    cfg.write_bytes("\ufeffhome = /private/builder\nexecutable = /private/python\nversion = 3.14\n".encode("utf-8"))
    payload.seal_pm_runtime(root, python)
    assert not cfg.read_bytes().startswith(b"\xef\xbb\xbf")
    text = cfg.read_text(encoding="utf-8-sig")
    assert text.splitlines() == [f"home = {Path('..') / 'tools' / 'pythön'}", "version = 3.14"]
    assert json.loads((runtime / "pm-runtime.json").read_text(encoding="utf-8-sig"))["python"] == "../tools/pythön/python"


def test_invalid_tool_evidence_never_partially_rewrites_facts(tmp_path):
    root = _payload(tmp_path)
    path = root / "tools" / "facts.json"
    before = json.loads(path.read_text(encoding="utf-8-sig"))
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
