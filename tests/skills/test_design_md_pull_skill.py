"""Tests for the optional skill ``design-md-pull``.

stdlib + pytest + unittest.mock only. No live network: every fetch is mocked
through ``urllib.request.urlopen``, so these run offline and never hit
raw.githubusercontent.com.
"""
from __future__ import annotations

import email.message
import importlib.util
import json
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "creative"
    / "design-md-pull"
    / "SKILL.md"
)
SCRIPT_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "creative"
    / "design-md-pull"
    / "scripts"
    / "fetch_design_md.py"
)

SAMPLE_DESIGN_MD = (
    "# Cursor DESIGN.md\n\n"
    "## Visual theme\n"
    "Dark window with gradient accents.\n"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("design_md_pull_fetch", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, str(SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_frontmatter():
    try:
        import hermes_yaml as yaml
    except ImportError:  # pragma: no cover - fallback for bare pytest runs
        import yaml  # type: ignore

    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n"), "SKILL.md must start with frontmatter"
    end = content.index("\n---\n", 3)
    return yaml.safe_load(content[3:end]), content


class FakeResponse:
    """Minimal stand-in for urllib's urlopen result (supports the context
    manager protocol the script uses)."""

    def __init__(self, body=b"", status=200, reason="OK", headers=None):
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _all_shipped_skill_names():
    names = set()
    for root in ("skills", "optional-skills"):
        names.update(p.parent.name for p in (REPO_ROOT / root).rglob("SKILL.md"))
    return names


# ── Frontmatter (authoring hardline) ──────────────────────────────────────

def test_frontmatter_required_fields():
    fm, _ = _load_frontmatter()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"


def test_name_matches_directory():
    fm, _ = _load_frontmatter()
    assert fm["name"] == "design-md-pull"


def test_description_hardline():
    fm, _ = _load_frontmatter()
    desc = str(fm["description"])
    assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
    assert desc.endswith("."), "description must end with a period"
    for marketing in ("powerful", "comprehensive", "seamless", "revolutionary", "cutting-edge", "state-of-the-art"):
        assert marketing not in desc.lower(), f"marketing word in description: {marketing}"


def test_author_credits_human_first():
    fm, _ = _load_frontmatter()
    author = str(fm["author"])
    assert "Hermes Agent" in author
    assert author.index(",") > 0, "author should credit the human before 'Hermes Agent'"


def test_license_platforms_tags():
    fm, _ = _load_frontmatter()
    assert fm["license"] == "MIT"
    assert {"linux", "macos", "windows"}.issubset(set(fm["platforms"]))
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags"), "metadata.hermes.tags required"


def test_related_skills_resolve_in_repo():
    fm, _ = _load_frontmatter()
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    shipped = _all_shipped_skill_names()
    for related in hermes.get("related_skills") or []:
        assert related in shipped, f"dangling related_skill: {related}"


def test_no_machine_local_paths():
    _, content = _load_frontmatter()
    assert "/home/" not in content, "SKILL.md must not bake in machine-local paths"


# ── Script behavior (all fetches mocked) ──────────────────────────────────

def test_fetch_returns_live_design_md(tmp_path):
    mod = _load_script_module()
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(body=SAMPLE_DESIGN_MD.encode("utf-8")),
    ) as mock:
        body, source = mod.fetch_design_md(
            "cursor", cache_dir=str(tmp_path / "cache")
        )
    assert body == SAMPLE_DESIGN_MD
    assert source == "fresh"
    assert "design-md/cursor/DESIGN.md" in mock.call_args.args[0].full_url


def test_fetch_404_raises_clear_error():
    mod = _load_script_module()
    error = urllib.error.HTTPError(
        "http://x", 404, "Not Found", email.message.Message(), None
    )
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(mod.FetchError) as exc_info:
            mod.fetch_design_md("no-such-site")
    message = str(exc_info.value)
    assert "no-such-site" in message
    assert "404" in message or "not found" in message.lower()


def test_fetch_rate_limit_falls_back_to_stale_cache(tmp_path):
    mod = _load_script_module()
    cache_dir = tmp_path / "cache"
    data_path, meta_path = mod._cache_paths(cache_dir, "vercel")
    data_path.parent.mkdir(parents=True)
    data_path.write_text("# cached vercel DESIGN.md\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps({"fetched_at": time.time() - 2 * mod.DEFAULT_TTL_SECONDS}),
        encoding="utf-8",
    )
    with patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("rate limited")
    ) as mock:
        body, source = mod.fetch_design_md(
            "vercel", cache_dir=str(cache_dir), ttl_seconds=mod.DEFAULT_TTL_SECONDS
        )
    assert body == "# cached vercel DESIGN.md\n"
    assert source.startswith("cache-stale")
    # The script does attempt the fetch (and fails) before serving the stale
    # cache; what matters is it never returns live content and degrades cleanly.
    mock.assert_called_once()


def test_fetch_fresh_cache_skips_network(tmp_path):
    mod = _load_script_module()
    cache_dir = tmp_path / "cache"
    data_path, meta_path = mod._cache_paths(cache_dir, "linear.app")
    data_path.parent.mkdir(parents=True)
    data_path.write_text("# fresh linear\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps({"fetched_at": time.time()}), encoding="utf-8"
    )
    with patch("urllib.request.urlopen") as mock:
        body, source = mod.fetch_design_md("linear.app", cache_dir=str(cache_dir))
    assert body == "# fresh linear\n"
    assert source == "cache-fresh"
    mock.assert_not_called()


def test_fetch_network_error_without_cache_raises_clear_message(tmp_path):
    mod = _load_script_module()
    with patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("Temporary failure")
    ):
        with pytest.raises(mod.FetchError) as exc_info:
            mod.fetch_design_md(
                "cursor", cache_dir=str(tmp_path / "empty"), ttl_seconds=mod.DEFAULT_TTL_SECONDS
            )
    message = str(exc_info.value)
    assert "cursor" in message
    assert "network" in message.lower() or "unreachable" in message.lower()


def test_main_prints_design_md_and_returns_zero(capsys, tmp_path):
    mod = _load_script_module()
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(body=SAMPLE_DESIGN_MD.encode("utf-8")),
    ):
        rc = mod.main(["cursor", "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0
    assert capsys.readouterr().out == SAMPLE_DESIGN_MD


def test_main_network_failure_prints_stderr_and_nonzero(capsys, tmp_path):
    mod = _load_script_module()
    with patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("down")
    ):
        rc = mod.main(
            ["cursor", "--cache-dir", str(tmp_path / "empty"), "--ttl-seconds", "0"]
        )
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "cursor" in stderr
    assert "design-md-pull" in stderr
