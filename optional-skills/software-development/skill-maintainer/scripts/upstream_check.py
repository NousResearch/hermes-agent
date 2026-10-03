#!/usr/bin/env python3
"""
Monthly upstream sync check — parses UPSTREAM_MANIFEST.md tables, fetches
upstream versions, compares content hashes for published skills, and reports
drift. Designed to run as a cron job on any agent platform.

Exit codes: 0 = all in sync (or unknown), 1 = drift detected

SETUP:
  1. Copy templates/upstream-manifest.md to your skills root as
     UPSTREAM_MANIFEST.md and fill in your skills
  2. Set SKILLS_ROOT and LAYER_2_REPO below (or via env vars)
  3. Optionally add ENGINE_DEPS entries for CLI tools / pip packages
  4. Schedule monthly via cron or your agent platform's scheduler

The script reads Layer 1 and Layer 2 tables directly from the manifest.
Engine deps are configured in this file (they require shell commands that
can't be safely parsed from markdown).

CUSTOMIZATION:
  - SKILLS_ROOT: where your skills + UPSTREAM_MANIFEST.md live
  - LAYER_2_REPO: path to your published repo (None to skip)
  - ENGINE_DEPS: list of dicts with name + command (as list, not shell string)
"""

import os
import re
import sys
import subprocess
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ─── CONFIGURATION — edit these to match your setup ───────────────────────

# Path to your skills directory (where UPSTREAM_MANIFEST.md lives)
# Derived from HERMES_HOME so named profiles are respected; override with
# SKILLS_ROOT env var if needed.
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
SKILLS_ROOT = HERMES_HOME / "skills"

# Path to your published repo (for Layer 2 checks). Set to None to skip.
# Override with LAYER_2_REPO env var if needed
LAYER_2_REPO = Path.home() / "agent_skills"

# Engine dependencies to check (CLI binaries and pip packages).
# Each entry: {"name": str, "command": [str, ...], "version_regex": r"..."}
# Commands are passed as lists — never shell=True — so untrusted config
# can't inject shell metacharacters. Do not populate ENGINE_DEPS from
# untrusted manifest input; keep it in this file under your control.
ENGINE_DEPS = [
    # Example:
    # {"name": "my-cli-tool", "command": ["my-cli", "--version"], "version_regex": r"(\d+\.\d+\.\d+)"},
    # {"name": "my-pip-pkg", "command": ["pip", "show", "my-pip-pkg"], "version_regex": r"(\d+\.\d+\.\d+)"},
]

# ─── END CONFIGURATION ────────────────────────────────────────────────────


def _resolve_config():
    """Allow env var overrides for configuration."""
    global SKILLS_ROOT, LAYER_2_REPO
    env_root = os.environ.get("SKILLS_ROOT")
    if env_root:
        SKILLS_ROOT = Path(env_root)
    env_repo = os.environ.get("LAYER_2_REPO")
    if env_repo:
        LAYER_2_REPO = Path(env_repo) if env_repo.lower() != "none" else None


MANIFEST_FILENAME = "UPSTREAM_MANIFEST.md"


# ─── Manifest parsing ─────────────────────────────────────────────────────


def parse_manifest(path):
    """Parse UPSTREAM_MANIFEST.md and return structured entries.

    Returns:
        {
            "layer1_skills": [{"skill": str, "source_repo": str, "repo_url": str,
                               "local_ver": str, "upstream_ver": str, ...}],
            "layer1_refs": [{...}],
            "layer1_engines": [{...}],
            "layer1_third_party": [{...}],
            "layer2_published": [{"skill": str, "repo_path": str, ...}],
        }
    """
    if not path.exists():
        return {"layer1_skills": [], "layer1_refs": [], "layer1_engines": [],
                "layer1_third_party": [], "layer2_published": []}

    text = path.read_text()
    sections = _split_sections(text)
    result = {
        "layer1_skills": [],
        "layer1_refs": [],
        "layer1_engines": [],
        "layer1_third_party": [],
        "layer2_published": [],
    }

    # Parse each table section
    for header, table_text in sections:
        rows = _parse_table(table_text)
        if not rows:
            continue

        header_lower = header.lower()
        if "whole skill" in header_lower:
            result["layer1_skills"] = rows
        elif "reference file" in header_lower:
            result["layer1_refs"] = rows
        elif "engine dep" in header_lower:
            result["layer1_engines"] = rows
        elif "third-party" in header_lower:
            result["layer1_third_party"] = rows
        elif "layer 2" in header_lower or "published repo" in header_lower:
            result["layer2_published"] = rows

    return result


def _split_sections(text):
    """Split manifest into (header, table_text) pairs by ## and ### headers."""
    sections = []
    current_header = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith("### ") or line.startswith("## "):
            if current_header and current_lines:
                sections.append((current_header, "\n".join(current_lines)))
            current_header = line.lstrip("# ").strip()
            current_lines = []
        elif current_header:
            current_lines.append(line)

    if current_header and current_lines:
        sections.append((current_header, "\n".join(current_lines)))

    return sections


def _parse_table(text):
    """Parse a markdown table into list of dicts using the header row as keys."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    # Find the header row (starts with |)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "|" in line[1:]:
            header_idx = i
            break

    if header_idx is None:
        return []

    header_cells = [c.strip().lower().replace(" ", "_") for c in
                    lines[header_idx].strip("|").split("|")]

    # Skip separator row (|---|---|...)
    data_start = header_idx + 1
    if data_start < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[data_start]):
        data_start += 1

    rows = []
    for line in lines[data_start:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        row = {}
        for key, val in zip(header_cells, cells):
            row[key] = val
        rows.append(row)

    return rows


def _extract_github_url(cell_text):
    """Extract GitHub repo owner/repo from a markdown link cell."""
    # Match [text](https://github.com/owner/repo)
    m = re.search(r"github\.com/([^/]+/[^/)\s]+)", cell_text)
    if m:
        repo = m.group(1).rstrip("/")
        # Remove trailing .git if present
        if repo.endswith(".git"):
            repo = repo[:-4]
        return repo
    return None


# ─── Fetching ─────────────────────────────────────────────────────────────


def fetch_url(url, timeout=15):
    """Fetch a URL via curl, return text or None."""
    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), url],
            capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout and "404: Not Found" not in r.stdout:
            return r.stdout
    except Exception:
        pass
    return None


def extract_version(text):
    """Extract version: field from YAML frontmatter."""
    m = re.search(r'^version:\s*(.+)$', text, re.M)
    if not m:
        return None
    val = m.group(1).strip()
    # Strip surrounding quotes if present
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    return val


def extract_name(text):
    """Extract name: field from YAML frontmatter."""
    m = re.search(r'^name:\s*(.+)$', text, re.M)
    return m.group(1).strip() if m else None


def get_last_commit_date(repo):
    """Get last commit date from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/commits?per_page=1"
    text = fetch_url(url)
    if not text:
        return None
    try:
        data = json.loads(text)
        if data and isinstance(data, list) and "commit" in data[0]:
            return data[0]["commit"]["committer"]["date"][:10]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def get_default_branch(repo):
    """Get default branch from GitHub API."""
    url = f"https://api.github.com/repos/{repo}"
    text = fetch_url(url)
    if not text:
        return "main"
    try:
        data = json.loads(text)
        return data.get("default_branch", "main")
    except (json.JSONDecodeError, KeyError):
        return "main"


def content_hash(text):
    """SHA-256 hash of text content for content-drift detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_local_skill(path):
    """Read a local SKILL.md file, return (name, version, hash) or None."""
    full = SKILLS_ROOT / path
    if not full.exists():
        return None
    text = full.read_text()
    return (extract_name(text), extract_version(text), content_hash(text))


# ─── Layer 1 checks ───────────────────────────────────────────────────────


def check_layer_1_skills(manifest_data):
    """Check Layer 1 whole-skills + reference-files tables."""
    results = []
    checked_repos = {}  # cache: repo -> default_branch

    all_entries = (manifest_data["layer1_skills"] +
                   manifest_data["layer1_refs"] +
                   manifest_data["layer1_third_party"])

    for row in all_entries:
        skill = row.get("skill", row.get("file", "?"))
        source_repo_cell = row.get("source_repo", row.get("source", ""))
        repo = _extract_github_url(source_repo_cell)

        if not repo:
            results.append(f"  ⚠️ {skill}: no GitHub URL in manifest")
            continue

        local_ver = row.get("local_ver", "")
        upstream_ver_field = row.get("upstream_ver", "")

        # Determine the path within the repo to fetch
        # For whole skills: skills/<skill-name>/SKILL.md
        # For reference files: the file path itself
        skill_path = row.get("skill", row.get("file", "")).strip("`")
        repo_skill_path = None
        if skill_path and not skill_path.startswith("http"):
            # If the skill path looks like a full path, use it
            if "/" in skill_path and skill_path.endswith(".md"):
                repo_skill_path = skill_path
            else:
                # Derive from skill name — try skills/<name>/SKILL.md
                skill_name = skill_path.split("/")[-1] if "/" in skill_path else skill_path
                repo_skill_path = f"skills/{skill_name}/SKILL.md"

        if repo not in checked_repos:
            checked_repos[repo] = get_default_branch(repo)
        branch = checked_repos[repo]

        if repo_skill_path:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{repo_skill_path}"
            upstream_text = fetch_url(url)
            if not upstream_text:
                # Try with 'master' branch
                url = f"https://raw.githubusercontent.com/{repo}/master/{repo_skill_path}"
                upstream_text = fetch_url(url)

            if not upstream_text:
                results.append(f"  ⚠️ {skill}: could not fetch upstream from {repo}")
                continue

            upstream_ver = extract_version(upstream_text) or "(no version)"

            if upstream_ver == "(no version)":
                last_commit = get_last_commit_date(repo)
                results.append(f"  ❓ {skill}: upstream has no version — last commit {last_commit or '?'}")
            elif upstream_ver == local_ver:
                results.append(f"  ✅ {skill}: v{local_ver} (in sync)")
            else:
                results.append(f"  🔄 {skill}: local v{local_ver} → upstream v{upstream_ver} **DRIFT**")
        else:
            # Can't determine path — check last commit
            last_commit = get_last_commit_date(repo)
            results.append(f"  📅 {skill}: last upstream commit {last_commit or '?'}")

    if not all_entries:
        results.append("  ℹ️ No Layer 1 entries in manifest")

    return results


# ─── Layer 2 checks ───────────────────────────────────────────────────────


def check_layer_2(manifest_data):
    """Check Layer 2: local skills vs published repo.

    Does TWO checks per skill:
    1. Version comparison (fast — catches version bumps)
    2. Content hash comparison (catches same-version content drift)
    """
    results = []

    if not LAYER_2_REPO or not LAYER_2_REPO.exists():
        results.append("  ℹ️ Published repo not found — set LAYER_2_REPO to enable")
        return results

    published = manifest_data.get("layer2_published", [])

    if not published:
        # Fallback: scan repo for all SKILL.md files
        repo_skills = [p for p in LAYER_2_REPO.rglob("SKILL.md")
                       if ".git" not in str(p)]
        for repo_path in sorted(repo_skills):
            results.extend(_check_single_layer2(repo_path, None))
    else:
        for row in published:
            skill_name = row.get("skill", "?").strip("`")
            repo_path_str = row.get("repo_path", "").strip("`")
            local_path_str = row.get("local_path", "").strip("`")

            repo_path = LAYER_2_REPO / repo_path_str / "SKILL.md" if repo_path_str else None
            local_path = SKILLS_ROOT / local_path_str / "SKILL.md" if local_path_str else None

            if not repo_path or not repo_path.exists():
                # Try finding by skill name
                repo_path = _find_skill_by_name(LAYER_2_REPO, skill_name)

            if not local_path or not local_path.exists():
                local_path = _find_skill_by_name(SKILLS_ROOT, skill_name)

            results.extend(_check_single_layer2(repo_path, local_path, skill_name))

    if not any("DRIFT" in r or "⚠️" in r for r in results):
        if not results:
            results.append("  ✅ No published skills to check")
        elif not any("✅" in r for r in results):
            results.append("  ✅ All published skills in sync")

    return results


def _find_skill_by_name(root, name):
    """Find a SKILL.md by skill name field in frontmatter."""
    if not root or not root.exists():
        return None
    for p in root.rglob("SKILL.md"):
        if ".git" in str(p):
            continue
        text = p.read_text()[:2000]
        if extract_name(text) == name:
            return p
    return None


def _check_single_layer2(repo_path, local_path, fallback_name=None):
    """Compare a single repo skill vs local. Returns list of result lines."""
    results = []

    if not repo_path or not repo_path.exists():
        name = fallback_name or "?"
        results.append(f"  ⚠️ {name}: repo SKILL.md not found")
        return results

    repo_text = repo_path.read_text()
    repo_name = extract_name(repo_text) or fallback_name or "?"
    repo_ver = extract_version(repo_text) or "(no version)"
    repo_hash = content_hash(repo_text)

    if not local_path or not local_path.exists():
        # Try to find by name
        local_path = _find_skill_by_name(SKILLS_ROOT, repo_name)

    if not local_path or not local_path.exists():
        results.append(f"  ⚠️ {repo_name}: no local counterpart found")
        return results

    local_text = local_path.read_text()
    local_ver = extract_version(local_text) or "(no version)"
    local_hash = content_hash(local_text)

    if local_ver != repo_ver:
        results.append(f"  🔄 {repo_name}: repo v{repo_ver} vs local v{local_ver} **VERSION DRIFT**")
    elif local_hash != repo_hash:
        results.append(f"  🔄 {repo_name}: same version (v{local_ver}) but content differs **CONTENT DRIFT**")
    # else: in sync — don't report

    return results


# ─── Engine dependency checks ─────────────────────────────────────────────


def check_engine_deps(manifest_data):
    """Check engine dependency versions.

    Engine deps are configured in ENGINE_DEPS in this file (not parsed from
    the manifest) because they require shell commands. This is a deliberate
    security boundary — manifest content is markdown and could be externally
    edited, so we don't execute commands derived from it.
    """
    results = []

    for dep in ENGINE_DEPS:
        name = dep["name"]
        cmd = dep["command"]
        try:
            r = subprocess.run(
                cmd,  # list form — no shell=True
                capture_output=True, text=True, timeout=10
            )
            output = (r.stdout + r.stderr).strip()
            regex = dep.get("version_regex", r"(\d+\.\d+\.\d+)")
            ver_m = re.search(regex, output)
            ver = ver_m.group(1) if ver_m else "unknown"
            results.append(f"  📦 {name}: v{ver}")
        except FileNotFoundError:
            results.append(f"  ⚠️ {name}: not installed (command not found)")
        except Exception as e:
            results.append(f"  ⚠️ {name}: check failed ({e})")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    _resolve_config()

    today = datetime.now().strftime("%Y-%m-%d")
    manifest_path = SKILLS_ROOT / MANIFEST_FILENAME

    lines = []
    lines.append(f"📋 Upstream Sync Check — {today}")
    lines.append("")

    if not manifest_path.exists():
        lines.append(f"⚠️ Manifest not found at {manifest_path}")
        lines.append("   Copy templates/upstream-manifest.md and fill in your skills.")
        print("\n".join(lines))
        return 1

    manifest_data = parse_manifest(manifest_path)

    skill_count = (len(manifest_data["layer1_skills"]) +
                   len(manifest_data["layer1_refs"]) +
                   len(manifest_data["layer1_third_party"]))
    published_count = len(manifest_data["layer2_published"])
    lines.append(f"Manifest: {skill_count} adapted skills, "
                 f"{published_count} published, "
                 f"{len(ENGINE_DEPS)} engine deps")
    lines.append("")

    lines.append("Layer 1 — External → Local:")
    lines.extend(check_layer_1_skills(manifest_data))
    lines.append("")

    lines.append("Layer 2 — Local → Published repo:")
    lines.extend(check_layer_2(manifest_data))
    lines.append("")

    engine_results = check_engine_deps(manifest_data)
    if engine_results:
        lines.append("Engine dependencies:")
        lines.extend(engine_results)
        lines.append("")

    output = "\n".join(lines)
    has_drift = "DRIFT" in output or "⚠️" in output

    if has_drift:
        print(output)
        return 1
    # Silent when in sync — no_agent cron delivers stdout only when there's
    # something to report. Empty stdout means "all good, nothing to say."
    return 0


if __name__ == "__main__":
    sys.exit(main())