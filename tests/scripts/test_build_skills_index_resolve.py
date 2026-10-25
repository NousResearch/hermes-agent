"""Invariants for scripts/build_skills_index.py's skills.sh path resolution.

Every build shipped ~1.9k skills.sh rows with no ``resolved_github_id``; most were rows whose
repo 404s or whose repo no longer has the skill, and each cost a client ~40 GitHub calls before it
failed as ``stale_index``. Tree payloads below are trimmed copies of the real GitHub responses
(google-deepmind/science-skills, expo/skills, PostHog/posthog, serpdownloaders/skills).
"""

import httpx

import scripts.build_skills_index as build_mod

_REPOS = {
    "google-deepmind/science-skills": {"tree": [
        {"path": "skills/pdb_database", "type": "tree"},
        {"path": "skills/pdb_database/SKILL.md", "type": "blob"},
        {"path": "skills/alphafold_database_fetch_and_analyze/SKILL.md", "type": "blob"},
    ], "truncated": False},
    "expo/skills": {"tree": [
        {"path": "plugins/expo/skills/eas-hosting/SKILL.md", "type": "blob"},
    ], "truncated": False},
    # 58k entries live; GitHub caps the recursive listing and flags it truncated.
    "posthog/posthog": {"tree": [{"path": "frontend/src/index.tsx", "type": "blob"}], "truncated": True},
    "flaky/repo": None,  # 502 on the repo lookup: transient, never a reason to drop
}
_CONTENTS = {"posthog/posthog/.agents/skills/react-doctor/SKILL.md"}


def _fake_get(url, **_kwargs):
    req = httpx.Request("GET", url)
    rest = url.removeprefix("https://api.github.com/repos/")
    repo = "/".join(rest.split("/")[:2])
    if repo not in _REPOS:
        return httpx.Response(404, json={"message": "Not Found"}, request=req)
    if _REPOS[repo] is None:
        return httpx.Response(502, request=req)
    tail = rest[len(repo):]
    if not tail:
        return httpx.Response(200, json={"default_branch": "main"}, request=req)
    if tail.startswith("/git/trees/"):
        return httpx.Response(200, json=_REPOS[repo], request=req)
    path = tail.removeprefix("/contents/")
    return httpx.Response(200 if f"{repo}/{path}" in _CONTENTS else 404, json={}, request=req)


class _Auth:
    def get_headers(self):
        return {}


def _row(identifier):
    repo = "/".join(identifier.split("/")[1:3])
    slug = identifier.rsplit("/", 1)[-1]
    return {"name": slug, "source": "skills.sh", "identifier": identifier, "repo": repo, "path": slug}


def _resolve(monkeypatch, *identifiers):
    monkeypatch.setattr(build_mod.httpx, "get", _fake_get)
    out = build_mod.batch_resolve_paths([_row(i) for i in identifiers], _Auth())
    return {s["identifier"]: s.get("resolved_github_id") for s in out}


def test_underscore_dirs_resolve_and_uninstallable_rows_are_not_shipped(monkeypatch):
    shipped = _resolve(
        monkeypatch,
        "skills-sh/google-deepmind/science-skills/pdb-database",
        "skills-sh/expo/skills/building-native-ui",      # tree read fine, skill gone upstream
        "skills-sh/serpdownloaders/skills/hulu-downloader",  # repo 404s
        "skills-sh/flaky/repo/some-skill",
    )
    assert shipped == {
        "skills-sh/google-deepmind/science-skills/pdb-database": "google-deepmind/science-skills/skills/pdb_database",
        "skills-sh/flaky/repo/some-skill": None,
    }


def test_truncated_tree_resolves_through_bounded_skill_md_probes(monkeypatch):
    shipped = _resolve(monkeypatch, "skills-sh/posthog/posthog/react-doctor", "skills-sh/posthog/posthog/gone-skill")
    # A partial tree proves nothing about absence: the unmatched row stays, unresolved.
    assert shipped == {
        "skills-sh/posthog/posthog/react-doctor": "posthog/posthog/.agents/skills/react-doctor",
        "skills-sh/posthog/posthog/gone-skill": None,
    }
