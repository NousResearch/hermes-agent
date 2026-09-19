"""Regression tests for GitHub skill helper paths and unsafe auth guidance (#106671).

The split-skill helper tree `skills/github/github-auth/scripts/` was removed
when the GitHub skills merged. Canonical helpers live at
`skills/software-development/github/scripts/`. Docs must point there, and
auth.md must not recommend plaintext store / token-in-URL / empty-passphrase
SSH / hand-built hosts.yml as routine fallbacks.
"""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_SKILL = REPO_ROOT / "skills" / "software-development" / "github"
LEGACY_HELPER_INFIX = "skills/github/github-auth/scripts/"
SCAN_SUFFIXES = {".md", ".sh"}

# ${HERMES_HOME:-...}/skills/.../scripts/<file>  or  skills/.../scripts/<file>
HELPER_PATH_RE = re.compile(
    r"(?:\$\{HERMES_HOME:-[^}]+\}/)?skills/[\w./-]+/scripts/[\w.-]+"
)

# Website copies of the (formerly split) GitHub skills; update if they still
# ship the broken helper path.
_WEBSITE_GITHUB_TREES = (
    REPO_ROOT / "website" / "docs" / "user-guide" / "skills" / "bundled" / "github",
    REPO_ROOT
    / "website"
    / "i18n"
    / "zh-Hans"
    / "docusaurus-plugin-content-docs"
    / "current"
    / "user-guide"
    / "skills"
    / "bundled"
    / "github",
)


def _scan_trees() -> list[Path]:
    trees = [GITHUB_SKILL, *[t for t in _WEBSITE_GITHUB_TREES if t.is_dir()]]
    files: list[Path] = []
    for tree in trees:
        for path in tree.rglob("*"):
            if path.is_file() and path.suffix in SCAN_SUFFIXES:
                files.append(path)
    return files


def test_canonical_github_helpers_exist():
    scripts = GITHUB_SKILL / "scripts"
    assert (scripts / "gh-env.sh").is_file()
    assert (scripts / "git-credential-token.py").is_file()


def test_shipped_github_docs_do_not_reference_removed_split_skill_helpers():
    offenders = []
    for path in _scan_trees():
        text = path.read_text(encoding="utf-8")
        if LEGACY_HELPER_INFIX in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], (
        "docs still reference the removed split-skill helper path "
        f"{LEGACY_HELPER_INFIX!r}: {offenders}"
    )


def test_referenced_skill_helper_scripts_exist_in_repo():
    missing = []
    for path in _scan_trees():
        text = path.read_text(encoding="utf-8")
        for raw in HELPER_PATH_RE.findall(text):
            rel = re.sub(r"^\$\{HERMES_HOME:-[^}]+\}/", "", raw)
            target = REPO_ROOT / rel
            if not target.is_file():
                missing.append(f"{path.relative_to(REPO_ROOT)} -> {rel}")
    assert missing == [], f"helper path references do not resolve in-repo: {missing}"


def test_auth_md_prefers_supported_flows_not_unsafe_fallbacks():
    auth = (GITHUB_SKILL / "references" / "auth.md").read_text(encoding="utf-8")

    # CONTROL: supported gh login / status guidance may remain.
    assert "gh auth login" in auth
    assert "gh auth status" in auth

    forbidden_snippets = (
        "credential.helper store",
        "https://<username>:<token>@github.com",
        "https://<user>:<token>@github.com",
        '-N ""',
        "hosts.yml",
    )
    found = [snippet for snippet in forbidden_snippets if snippet in auth]
    assert found == [], (
        "auth.md still recommends unsafe agent-operated fallbacks as guidance: "
        f"{found}"
    )
