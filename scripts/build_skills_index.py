#!/usr/bin/env python3
"""Build the Hermes Skills Index — a centralized JSON catalog of all skills.

This script crawls every skill source (skills.sh, GitHub taps, official,
clawhub, lobehub, claude-marketplace) and writes a JSON index with resolved
GitHub paths. The index is served as a static file on the docs site so that
`hermes skills search/install` can use it without hitting the GitHub API.

Usage:
    # Local (uses gh CLI or GITHUB_TOKEN for auth)
    python scripts/build_skills_index.py

    # CI (set GITHUB_TOKEN as secret)
    GITHUB_TOKEN=ghp_... python scripts/build_skills_index.py

Output: website/static/api/skills-index.json
"""

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Allow importing from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Ensure HERMES_HOME is set (needed by tools/skills_hub.py imports)
os.environ.setdefault("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))

from tools.skills_hub import (
    GitHubAuth,
    GitHubSource,
    SkillsShSource,
    OptionalSkillSource,
    WellKnownSkillSource,
    ClawHubSource,
    ClaudeMarketplaceSource,
    LobeHubSource,
    BrowseShSource,
    SkillMeta,
)
import httpx

OUTPUT_PATH = os.path.join(REPO_ROOT, "website", "static", "api", "skills-index.json")
INDEX_VERSION = 1


def _meta_to_dict(meta: SkillMeta) -> dict:
    """Convert a SkillMeta to a serializable dict."""
    return {
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "identifier": meta.identifier,
        "trust_level": meta.trust_level,
        "repo": meta.repo or "",
        "path": meta.path or "",
        "tags": meta.tags or [],
        "extra": meta.extra or {},
    }


def crawl_source(source, source_name: str, limit: int) -> list:
    """Crawl a single source and return skill dicts."""
    print(f"  Crawling {source_name}...", flush=True)
    start = time.time()
    try:
        results = source.search("", limit=limit)
    except Exception as e:
        print(f"  Error crawling {source_name}: {e}", file=sys.stderr)
        return []
    skills = [_meta_to_dict(m) for m in results]
    elapsed = time.time() - start
    print(f"  {source_name}: {len(skills)} skills ({elapsed:.1f}s)", flush=True)
    return skills


def crawl_skills_sh(source: SkillsShSource) -> list:
    """Crawl skills.sh via its sitemap to enumerate the full catalog (~20k entries).

    Previously walked a hardcoded list of ~28 popular keywords (each capped at
    50 results) which yielded ~850 unique skills — about 4% of the real catalog.
    The SkillsShSource.search("") path now hits the sitemap directly, returning
    the full 20k-entry catalog deduplicated by canonical identifier.
    """
    print("  Crawling skills.sh (sitemap)...", flush=True)
    start = time.time()

    try:
        results = source.search("", limit=0)  # 0 = no cap, return the whole catalog
    except Exception as e:
        print(f"    Warning: skills.sh sitemap walk failed: {e}", file=sys.stderr)
        results = []

    all_skills: dict[str, dict] = {}
    for meta in results:
        entry = _meta_to_dict(meta)
        if entry["identifier"] not in all_skills:
            all_skills[entry["identifier"]] = entry

    elapsed = time.time() - start
    print(f"  skills.sh: {len(all_skills)} unique skills ({elapsed:.1f}s)",
          flush=True)
    return list(all_skills.values())


def _fetch_repo_tree(repo: str, auth: GitHubAuth) -> list:
    """Fetch the recursive tree for a repo. Returns list of tree entries."""
    headers = auth.get_headers()
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers, timeout=15, follow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        branch = resp.json().get("default_branch", "main")

        resp = httpx.get(
            f"https://api.github.com/repos/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
            headers=headers, timeout=30, follow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        if data.get("truncated"):
            return []
        return data.get("tree", [])
    except Exception:
        return []


def batch_resolve_paths(skills: list, auth: GitHubAuth) -> list:
    """Resolve GitHub paths for skills.sh entries using batch tree lookups.

    Instead of resolving each skill individually (N×M API calls), we:
    1. Group skills by repo
    2. Fetch one tree per repo (2 API calls per repo)
    3. Find all SKILL.md files in the tree
    4. Match skills to their resolved paths
    """
    # Filter to skills.sh entries that need resolution
    skills_sh = [s for s in skills if s["source"] in {"skills.sh", "skills-sh"}]
    if not skills_sh:
        return skills

    print(f"  Resolving paths for {len(skills_sh)} skills.sh entries...",
          flush=True)
    start = time.time()

    # Group by repo
    by_repo: dict[str, list] = defaultdict(list)
    for s in skills_sh:
        repo = s.get("repo", "")
        if repo:
            by_repo[repo].append(s)

    print(f"    {len(by_repo)} unique repos to scan", flush=True)

    resolved_count = 0

    # Fetch trees in parallel (up to 6 concurrent)
    def _resolve_repo(repo: str, entries: list):
        tree = _fetch_repo_tree(repo, auth)
        if not tree:
            return 0

        # Find all SKILL.md paths in this repo
        skill_paths = {}  # skill_dir_name -> full_path
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if path.endswith("/SKILL.md"):
                skill_dir = path[: -len("/SKILL.md")]
                dir_name = skill_dir.split("/")[-1]
                skill_paths[dir_name.lower()] = f"{repo}/{skill_dir}"

                # Also check SKILL.md frontmatter name if we can match by path
                # For now, just index by directory name
            elif path == "SKILL.md":
                # Root-level SKILL.md
                skill_paths["_root_"] = f"{repo}"

        count = 0
        for entry in entries:
            # Try to match the skill's name/path to a tree entry
            skill_name = entry.get("name", "").lower()
            skill_path = entry.get("path", "").lower()
            identifier = entry.get("identifier", "")

            # Extract the skill token from the identifier
            # e.g. "skills-sh/d4vinci/scrapling/scrapling-official" -> "scrapling-official"
            parts = identifier.replace("skills-sh/", "").replace("skills.sh/", "")
            skill_token = parts.split("/")[-1].lower() if "/" in parts else ""

            # Try matching in order of likelihood
            for candidate in [skill_token, skill_name, skill_path]:
                if not candidate:
                    continue
                matched = skill_paths.get(candidate)
                if matched:
                    entry["resolved_github_id"] = matched
                    count += 1
                    break
            else:
                # Try fuzzy: skill_token with common transformations
                for tree_name, tree_path in skill_paths.items():
                    if (skill_token and (
                        tree_name.replace("-", "") == skill_token.replace("-", "")
                        or skill_token in tree_name
                        or tree_name in skill_token
                    )):
                        entry["resolved_github_id"] = tree_path
                        count += 1
                        break

        return count

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_resolve_repo, repo, entries): repo
            for repo, entries in by_repo.items()
        }
        for future in as_completed(futures):
            try:
                resolved_count += future.result()
            except Exception as e:
                repo = futures[future]
                print(f"    Warning: {repo}: {e}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"  Resolved {resolved_count}/{len(skills_sh)} paths ({elapsed:.1f}s)",
          flush=True)
    return skills


def main():
    print("Building Hermes Skills Index...", flush=True)
    overall_start = time.time()

    auth = GitHubAuth()
    print(f"GitHub auth: {auth.auth_method()}")
    if auth.auth_method() == "anonymous":
        print("WARNING: No GitHub authentication — rate limit is 60/hr. "
              "Set GITHUB_TOKEN for better results.", file=sys.stderr)

    skills_sh_source = SkillsShSource(auth=auth)
    sources = {
        "official": OptionalSkillSource(),
        "well-known": WellKnownSkillSource(),
        "github": GitHubSource(auth=auth),
        "clawhub": ClawHubSource(),
        "claude-marketplace": ClaudeMarketplaceSource(auth=auth),
        "lobehub": LobeHubSource(),
        "browse-sh": BrowseShSource(),
    }

    all_skills: list[dict] = []

    # Crawl skills.sh
    all_skills.extend(crawl_skills_sh(skills_sh_source))

    # Crawl other sources in parallel.
    # Per-source soft caps — sources stop returning when they run out, so these
    # are ceilings, not targets.  ClawHub has 20k+ skills; bumping to 100k
    # (well above current catalog size) lets the full catalog land in the
    # index instead of being truncated at an arbitrary build-time limit.
    SOURCE_LIMITS = {
        # ClawHub had 49,698+ skills as of May 2026; 200k leaves headroom.
        "clawhub": 200_000,
        "lobehub": 100_000,
        "browse-sh": 5_000,
        "claude-marketplace": 5_000,
        "github": 5_000,
        "well-known": 5_000,
        "official": 5_000,
    }
    DEFAULT_SOURCE_LIMIT = 500

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for name, source in sources.items():
            limit = SOURCE_LIMITS.get(name, DEFAULT_SOURCE_LIMIT)
            futures[pool.submit(crawl_source, source, name, limit)] = name
        for future in as_completed(futures):
            try:
                all_skills.extend(future.result())
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

    # Batch resolve GitHub paths for skills.sh entries
    all_skills = batch_resolve_paths(all_skills, auth)

    # Collect which sources hit a GitHub API rate limit during the crawl.
    # github / claude-marketplace / well-known all read api.github.com, so a
    # rate-limited token zeroes all three at once — surfaced below so the
    # failure message names the real cause instead of "source returned 0".
    rate_limited_sources = {
        name for name, source in sources.items()
        if getattr(source, "is_rate_limited", False)
    }
    if rate_limited_sources:
        print(
            "  WARNING: GitHub API rate limit hit for: "
            + ", ".join(sorted(rate_limited_sources)),
            file=sys.stderr,
        )

    # Deduplicate by identifier
    seen: dict[str, dict] = {}
    for skill in all_skills:
        key = skill["identifier"]
        if key not in seen:
            seen[key] = skill
    deduped = list(seen.values())

    # Sort
    source_order = {"official": 0, "skills-sh": 1, "skills.sh": 1,
                    "github": 2, "well-known": 3, "clawhub": 4,
                    "browse-sh": 5, "claude-marketplace": 6, "lobehub": 7}
    deduped.sort(key=lambda s: (source_order.get(s["source"], 99), s["name"]))

    from collections import Counter
    by_source = Counter(s["source"] for s in deduped)
    print(f"\nCrawled {len(deduped)} skills in {time.time() - overall_start:.0f}s")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        resolved = sum(1 for s in deduped
                       if s["source"] == src and s.get("resolved_github_id"))
        extra = f" ({resolved} resolved)" if resolved else ""
        print(f"  {src}: {count}{extra}")

    # Health check: catch silent breakage early. Every source listed below
    # has historically returned at least `floor` entries; a zero (or near-
    # zero) result almost certainly means a tap path moved, an API changed,
    # or rate limiting kicked in.  Failing here forces a human look before
    # the broken index reaches the live docs.
    EXPECTED_FLOORS = {
        # skills.sh now uses the sitemap walker (~20k catalog as of May 2026).
        # Anything under 10k means the sitemap shape changed or fetches failed
        # — better to fail loudly than ship a regression to the 858-skill
        # popular-queries era.
        "skills.sh": 10000,
        "lobehub": 100,
        # ClawHub had 49,698+ skills as of May 2026 — anything under 20k means
        # pagination broke or the API surface changed.  Fail loudly rather
        # than ship a degenerate index (we shipped 200/50000 silently for
        # weeks because the floor was 50).
        "clawhub": 20000,
        "official": 50,
        "github": 30,        # collapsed across all GitHub taps
        "browse-sh": 50,
    }
    health_errors = []
    for src, floor in EXPECTED_FLOORS.items():
        # 'skills-sh' and 'skills.sh' are the same source; both labels exist.
        count = by_source.get(src, 0)
        if src == "skills.sh":
            count = by_source.get("skills.sh", 0) + by_source.get("skills-sh", 0)
        if count < floor:
            health_errors.append(f"  {src}: {count} < expected floor {floor}")

    MIN_TOTAL = 1500
    if len(deduped) < MIN_TOTAL:
        health_errors.append(
            f"  total: {len(deduped)} < expected floor {MIN_TOTAL}"
        )

    if health_errors:
        print(
            "\nERROR: skills index health check failed — refusing to ship "
            "a degenerate index. Investigate the following sources:",
            file=sys.stderr,
        )
        for line in health_errors:
            print(line, file=sys.stderr)
        if rate_limited_sources:
            print(
                "\nGitHub API rate limit was hit during this crawl for: "
                + ", ".join(sorted(rate_limited_sources))
                + ". This is the usual cause of an all-GitHub-tap collapse "
                "(github / claude-marketplace / well-known dropping to zero "
                "together). Re-run with a higher-quota GITHUB_TOKEN.",
                file=sys.stderr,
            )
        print(
            "\nIf the drop is expected (e.g. a hub is genuinely shutting "
            "down), lower the floor in scripts/build_skills_index.py "
            "EXPECTED_FLOORS in the same PR.",
            file=sys.stderr,
        )
        # IMPORTANT: do NOT write OUTPUT_PATH on failure. The index file is
        # gitignored, so a fresh deploy checkout has no copy on disk — leaving
        # it absent lets website/scripts/extract-skills.py fall back to the
        # legacy snapshot cache (or skip the unified index) instead of reading
        # a degenerate file. Writing-then-exiting-2 was the bug that shipped an
        # index with every GitHub-API source dropped to zero: deploy-site.yml
        # swallows the exit code with `|| echo non-fatal`, and the partial file
        # was already on disk for extract-skills to pick up.
        sys.exit(2)

    # Healthy — only now write the index out for the docs build to consume.
    index = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_count": len(deduped),
        "skills": deduped,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nDone! {len(deduped)} skills indexed in "
          f"{time.time() - overall_start:.0f}s")
    print(f"Output: {OUTPUT_PATH} ({file_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
