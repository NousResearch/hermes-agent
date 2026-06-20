"""Invariants for scripts/build_skills_index.py's health-check guard.

Regression context (June 2026): a GitHub API rate limit zeroed every
api.github.com-backed source (github / claude-marketplace / well-known) at
once during the docs deploy crawl. The build's health check fired and exited
non-zero — but it had ALREADY written the degenerate index to disk, and
deploy-site.yml swallowed the exit code with ``|| echo non-fatal``. The
partial index (missing the OpenAI/Anthropic/HuggingFace/NVIDIA tabs) shipped
to the live Skills Hub.

These tests pin the two contracts that prevent a recurrence:
  1. A degenerate crawl exits non-zero AND does NOT write the output file
     (so extract-skills.py falls back instead of reading a broken index).
  2. A healthy crawl exits zero AND writes the file with every source present.
"""

import types

import pytest

import scripts.build_skills_index as build_mod


def _meta(name, src):
    return build_mod.SkillMeta(
        name=name, description="d", source=src,
        identifier=f"{src}/{name}", trust_level="community",
    )


class _FakeSource:
    def __init__(self, src, n, rate_limited=False):
        self._src = src
        self._n = n
        self.is_rate_limited = rate_limited

    def search(self, query, limit=10):
        return [_meta(f"{self._src}-{i}", self._src) for i in range(self._n)]


def _install_fake_sources(monkeypatch, *, github_count, claude_count=40,
                          well_known_count=10, github_rate_limited=False):
    monkeypatch.setattr(build_mod, "SkillsShSource", lambda auth: _FakeSource("skills.sh", 15000))
    monkeypatch.setattr(build_mod, "OptionalSkillSource", lambda: _FakeSource("official", 95))
    monkeypatch.setattr(build_mod, "WellKnownSkillSource", lambda: _FakeSource("well-known", well_known_count))
    monkeypatch.setattr(
        build_mod, "GitHubSource",
        lambda auth: _FakeSource("github", github_count, rate_limited=github_rate_limited),
    )
    monkeypatch.setattr(build_mod, "ClawHubSource", lambda: _FakeSource("clawhub", 69000))
    monkeypatch.setattr(
        build_mod, "ClaudeMarketplaceSource",
        lambda auth: _FakeSource("claude-marketplace", claude_count, rate_limited=github_rate_limited),
    )
    monkeypatch.setattr(build_mod, "LobeHubSource", lambda: _FakeSource("lobehub", 500))
    monkeypatch.setattr(build_mod, "BrowseShSource", lambda: _FakeSource("browse-sh", 380))
    monkeypatch.setattr(
        build_mod, "crawl_skills_sh",
        lambda source: [build_mod._meta_to_dict(m) for m in source.search("", 0)],
    )
    monkeypatch.setattr(build_mod, "batch_resolve_paths", lambda skills, auth: skills)
    monkeypatch.setattr(
        build_mod, "GitHubAuth",
        lambda: types.SimpleNamespace(auth_method=lambda: "token"),
    )


def test_degenerate_crawl_exits_nonzero_and_writes_no_file(tmp_path, monkeypatch):
    """A collapsed GitHub crawl must fail loud and leave OUTPUT_PATH unwritten."""
    out = tmp_path / "skills-index.json"
    monkeypatch.setattr(build_mod, "OUTPUT_PATH", str(out))
    _install_fake_sources(monkeypatch, github_count=0, claude_count=0,
                          well_known_count=0, github_rate_limited=True)

    with pytest.raises(SystemExit) as exc:
        build_mod.main()

    assert exc.value.code != 0
    # The degenerate index must NOT have been written — extract-skills.py
    # relies on the file's absence to fall back instead of reading garbage.
    assert not out.exists()


def test_healthy_crawl_writes_index_with_all_sources(tmp_path, monkeypatch):
    out = tmp_path / "skills-index.json"
    monkeypatch.setattr(build_mod, "OUTPUT_PATH", str(out))
    _install_fake_sources(monkeypatch, github_count=200)

    build_mod.main()  # exit 0 (no SystemExit)

    assert out.exists()
    import json
    data = json.loads(out.read_text())
    sources = {s["source"] for s in data["skills"]}
    # Every GitHub-API-backed source that vanished in the regression is present.
    assert {"github", "claude-marketplace", "well-known"} <= sources
    assert data["skill_count"] == len(data["skills"])
