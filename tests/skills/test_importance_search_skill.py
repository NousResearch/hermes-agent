"""Hermetic tests for the importance-search optional skill.

The live pipeline needs Codex OAuth + network, so these tests stub
``agent.auxiliary_client`` / ``tools.url_safety`` / ``tools.website_policy``
before importing the script, then verify:
  - SKILL.md frontmatter conforms to the hardline format
  - config parses and matches what the script expects
  - view-velocity ranking actually uses views-per-day, not raw view totals
  - deep_fetch refuses unsafe / policy-blocked URLs and blocked redirects
  - yt-dlp is invoked with the approximate_date extractor arg
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "research" / "importance-search"
SCRIPT = SKILL_DIR / "scripts" / "importance_search.py"

DAY = 86400.0


def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture()
def mod():
    stubs = {
        "agent": _stub_module("agent"),
        "agent.auxiliary_client": _stub_module(
            "agent.auxiliary_client",
            call_llm=mock.Mock(return_value={}),
            extract_content_or_reasoning=mock.Mock(return_value=""),
        ),
        "tools": _stub_module("tools"),
        "tools.url_safety": _stub_module(
            "tools.url_safety", is_safe_url=mock.Mock(return_value=True)
        ),
        "tools.website_policy": _stub_module(
            "tools.website_policy", check_website_access=mock.Mock(return_value=None)
        ),
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("importance_search_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


# --- hardline frontmatter conventions -------------------------------------

def test_description_hardline(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "importance-search"


def test_author_credits_contributor_first(frontmatter) -> None:
    assert frontmatter["author"].startswith("Su Ham"), frontmatter["author"]


def test_platforms_cross_platform(frontmatter) -> None:
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_metadata_category_research(frontmatter) -> None:
    assert frontmatter["metadata"]["hermes"]["category"] == "research"


# --- script hygiene --------------------------------------------------------

def test_no_hardcoded_source_path() -> None:
    src = SCRIPT.read_text()
    assert "sys.path.insert" not in src, "script must not prepend a hard-coded checkout path"
    assert "~/hermes-agent" not in src


def test_config_matches_script_expectations() -> None:
    cfg = json.loads((SKILL_DIR / "search_domains.json").read_text())
    domains = cfg["domains"]
    assert "ai-tech" in domains
    for key, d in domains.items():
        assert d.get("label"), f"domain {key} missing label"
        assert d.get("keywords"), f"domain {key} missing keywords"


# --- view velocity ----------------------------------------------------------

def test_view_velocity_is_views_per_day(mod) -> None:
    now = 1_000_000_000.0
    item = {"views": 1000, "timestamp": now - 10 * DAY}
    assert mod.view_velocity(item, now=now) == pytest.approx(100.0)


def test_view_velocity_undated_fallback(mod) -> None:
    assert mod.view_velocity({"views": 300, "timestamp": 0}, now=0) == pytest.approx(10.0)


def test_view_velocity_age_floor_one_day(mod) -> None:
    now = 1_000_000_000.0
    item = {"views": 500, "timestamp": now - DAY / 24}
    assert mod.view_velocity(item, now=now) == pytest.approx(500.0)


def test_youtube_top_ranks_fresh_over_stale_viral(mod) -> None:
    now = 1_000_000_000.0
    stale_viral = {"title": "old", "url": "https://youtu.be/old", "views": 1_000_000,
                   "channel": "", "duration": 600, "timestamp": now - 1000 * DAY}
    fresh = {"title": "new", "url": "https://youtu.be/new", "views": 50_000,
             "channel": "", "duration": 600, "timestamp": now - 1 * DAY}
    short = {"title": "short", "url": "https://youtu.be/short", "views": 9_999_999,
             "channel": "", "duration": 45, "timestamp": now - 1 * DAY}
    with mock.patch.object(mod, "yt_flat_search", return_value=[stale_viral, fresh, short]), \
         mock.patch.object(mod.time, "time", return_value=now):
        top = mod.youtube_top({"youtube_queries": ["q"]}, k=3)
    urls = [t["url"] for t in top]
    assert urls[0] == "https://youtu.be/new", "views-per-day must outrank raw view total"
    assert "https://youtu.be/short" not in urls, "sub-90s items must be filtered"


def test_yt_flat_search_requests_approximate_date(mod) -> None:
    fake = mock.Mock(stdout=json.dumps(
        {"id": "abc", "title": "t", "view_count": 10, "duration": 100, "timestamp": 123}) + "\n")
    with mock.patch.object(mod.subprocess, "run", return_value=fake) as run:
        items = mod.yt_flat_search("query", n=2)
    cmd = run.call_args[0][0]
    assert "--extractor-args" in cmd
    assert "youtubetab:approximate_date" in cmd
    assert items == [{"title": "t", "url": "https://youtu.be/abc", "views": 10,
                      "channel": "", "duration": 100, "timestamp": 123}]


# --- deep fetch safety -------------------------------------------------------

def test_deep_fetch_refuses_unsafe_url(mod) -> None:
    mod.is_safe_url.return_value = False
    with mock.patch.object(mod.urllib.request, "build_opener") as build_opener:
        assert mod.deep_fetch("http://169.254.169.254/latest/meta-data/") == ""
    build_opener.assert_not_called()


def test_deep_fetch_refuses_policy_blocked_url(mod) -> None:
    mod.check_website_access.return_value = {"message": "host blocked by policy"}
    with mock.patch.object(mod.urllib.request, "build_opener") as build_opener:
        assert mod.deep_fetch("https://blocked.example.com/a") == ""
    build_opener.assert_not_called()


def test_redirect_handler_blocks_unsafe_target(mod) -> None:
    handler = mod._SafeRedirectHandler()
    req = urllib.request.Request("https://public.example.com/")
    mod.is_safe_url.side_effect = lambda url: "169.254" not in url
    with pytest.raises(urllib.error.HTTPError, match="redirect blocked"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://169.254.169.254/")


def test_redirect_handler_allows_safe_target(mod) -> None:
    handler = mod._SafeRedirectHandler()
    req = urllib.request.Request("https://public.example.com/")
    out = handler.redirect_request(req, None, 302, "Found", {}, "https://safe.example.com/next")
    assert out is not None and out.full_url == "https://safe.example.com/next"


# --- config / domain resolution ----------------------------------------------

def test_resolve_domain_falls_back_to_first(mod) -> None:
    cfg = {"domains": {"first": {"label": "First"}, "second": {"label": "Second"}}}
    assert mod.resolve_domain(cfg, "nope")["label"] == "First"
    assert mod.resolve_domain(cfg, "second")["label"] == "Second"


def test_load_config_missing_file_exits(mod) -> None:
    with mock.patch.object(mod, "CONFIG", str(SKILL_DIR / "does-not-exist.json")):
        with pytest.raises(SystemExit, match="cannot load config"):
            mod.load_config()


def test_resolve_domain_empty_config_exits(mod) -> None:
    with pytest.raises(SystemExit, match="no domains configured"):
        mod.resolve_domain({"domains": {}}, "any")
