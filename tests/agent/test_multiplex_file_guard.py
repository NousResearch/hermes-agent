"""File tools never cross served profile homes under multiplex (4d)."""
from __future__ import annotations

import json

import pytest

from agent import multiplex_file_guard as mfg
from agent import secret_scope as ss
from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_denied
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

FILES = ("config.yaml", "SOUL.md", "skills/x/SKILL.md", "memories/MEMORY.md", "cron/jobs.json")


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    edu = root / "profiles" / "educator"
    other = root / "profiles" / "unserved"
    for home in (root, edu, other):
        for rel in FILES:
            f = home / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            # config.yaml must be a valid mapping: an unparseable one now
            # turns agent_write_guard ON (agent/boundary_config.py).
            f.write_text("model: x\n" if rel == "config.yaml" else "x", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    mfg.set_served_profile_homes([root, edu])
    mfg._recent_audits.clear()
    ss.set_multiplex_active(True)
    yield root, edu, other
    ss.set_multiplex_active(False)
    mfg.set_served_profile_homes([])


class _As:
    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self.tok = set_hermes_home_override(str(self.home))

    def __exit__(self, *a):
        reset_hermes_home_override(self.tok)


@pytest.mark.parametrize("rel", FILES)
def test_educator_cannot_touch_default(homes, rel):
    root, edu, _ = homes
    with _As(edu):
        assert get_read_block_error(str(root / rel))
        assert is_write_denied(str(root / rel))
        assert "another profile" in get_write_denied_error(str(root / rel))
        assert get_read_block_error(str(edu / rel)) is None
        assert not is_write_denied(str(edu / rel))


@pytest.mark.parametrize("rel", FILES)
def test_default_cannot_touch_educator(homes, rel):
    root, edu, _ = homes
    with _As(root):
        assert get_read_block_error(str(edu / rel))
        assert is_write_denied(str(edu / rel))
        assert get_read_block_error(str(root / rel)) is None
        assert not is_write_denied(str(root / rel))


def test_unserved_profile_and_outside_paths_not_denied(homes, tmp_path):
    root, edu, other = homes
    outside = tmp_path / "work" / "notes.md"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")
    with _As(edu):
        assert get_read_block_error(str(other / "SOUL.md")) is None
        assert get_read_block_error(str(outside)) is None


def test_multiplex_off_never_denies(homes):
    root, edu, _ = homes
    ss.set_multiplex_active(False)
    with _As(edu):
        assert mfg.cross_profile_denial(str(root / "SOUL.md")) is None
        assert get_read_block_error(str(root / "SOUL.md")) is None


def test_denial_is_audited_once(homes):
    root, edu, _ = homes
    with _As(edu):
        get_read_block_error(str(root / "SOUL.md"))
        get_read_block_error(str(root / "SOUL.md"))
        log = edu / "audit" / "data-access.log"
        lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    hits = [l for l in lines if l["action"] == "cross_profile_denied"]
    assert len(hits) == 1
    assert hits[0]["target"].endswith("SOUL.md")


def test_read_file_tool_refuses(homes):
    root, edu, _ = homes
    from tools.file_tools import read_file_tool

    with _As(edu):
        out = json.loads(read_file_tool(str(root / "SOUL.md")))
    assert "another profile" in (out.get("error") or "")


# ── search_files: every hit is checked, not just the search root ───────────


def _search(root, **kw):
    from tools.file_tools import search_tool

    return json.loads(search_tool(pattern="MARKER", path=str(root), task_id="mx", **kw))


@pytest.fixture()
def marked(homes):
    root, edu, other = homes
    for home in (root, edu, other):
        for rel in FILES:
            (home / rel).write_text(f"MARKER in {home.name}/{rel}\n", encoding="utf-8")
    return homes


def _hit_paths(result):
    paths = [m.get("path", "") for m in result.get("matches", []) or []]
    paths += list(result.get("files", []) or [])
    paths += list((result.get("counts") or {}).keys())
    # content mode: path-grouped text — each path on its own line, then
    # indented "<line>: <content>" rows.
    for line in (result.get("matches_text") or "").splitlines():
        if line and not line[0].isspace():
            paths.append(line.strip())
    return paths


@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
def test_search_from_default_root_hides_educator_hits(marked, mode):
    root, edu, other = marked
    with _As(root):
        hits = _hit_paths(_search(root, output_mode=mode))
    assert hits, "default should still find its own files"
    assert not any(str(edu) in h for h in hits), hits
    assert any(str(other) in h for h in hits)  # unserved profile is not a boundary


@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
def test_search_from_educator_over_the_root_hides_default_hits(marked, mode):
    root, edu, _ = marked
    with _As(edu):
        hits = _hit_paths(_search(root.parent, output_mode=mode))
    assert any(str(edu) in h for h in hits), hits
    leaked = [h for h in hits if h.startswith(str(root)) and str(root / "profiles") not in h]
    assert not leaked, leaked


def test_filename_search_hides_other_profile(marked):
    root, edu, _ = marked
    from tools.file_tools import search_tool

    with _As(root):
        out = json.loads(search_tool(pattern="SOUL.md", target="files", path=str(root), task_id="mx"))
    files = out.get("files", []) or []
    assert files and not any(str(edu) in f for f in files), files


def test_guard_fails_closed_on_internal_error(homes, monkeypatch):
    root, edu, _ = homes
    import hermes_constants

    def boom():
        raise RuntimeError("injected")

    monkeypatch.setattr(hermes_constants, "get_hermes_home", boom)
    assert "could not verify" in mfg.cross_profile_denial(str(edu / "SOUL.md"))
    ss.set_multiplex_active(False)
    assert mfg.cross_profile_denial(str(edu / "SOUL.md")) is None
