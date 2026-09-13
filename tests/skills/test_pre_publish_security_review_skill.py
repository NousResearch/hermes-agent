"""Contract and behavior tests for pre-publish-security-review."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = (
    REPO_ROOT
    / "optional-skills"
    / "security"
    / "pre-publish-security-review"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT = SKILL_DIR / "scripts" / "inventory_publish_surface.py"
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Review Lenses",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]
OWASP_2025_CATEGORIES = {
    "A01 Broken Access Control",
    "A02 Security Misconfiguration",
    "A03 Software Supply Chain Failures",
    "A04 Cryptographic Failures",
    "A05 Injection",
    "A06 Insecure Design",
    "A07 Authentication Failures",
    "A08 Software or Data Integrity Failures",
    "A09 Security Logging and Alerting Failures",
    "A10 Mishandling of Exceptional Conditions",
}


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter_and_body(skill_text: str) -> tuple[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", skill_text, re.DOTALL)
    assert match, "SKILL.md must begin with YAML frontmatter"
    return match.group(1), match.group(2)


def _frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(
        rf"^[ \t]*{re.escape(key)}:\s*(.+)$", frontmatter, re.MULTILINE
    )
    assert match, f"missing frontmatter field: {key}"
    return match.group(1).strip().strip('"')


def _run_inventory(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _load_inventory_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inventory_publish_surface", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frontmatter_meets_repo_standard(frontmatter_and_body: tuple[str, str]) -> None:
    frontmatter, body = frontmatter_and_body
    assert _frontmatter_value(frontmatter, "name") == "pre-publish-security-review"
    description = _frontmatter_value(frontmatter, "description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert _frontmatter_value(frontmatter, "version") == "0.1.0"
    assert _frontmatter_value(frontmatter, "author") == (
        "Mark S. (unsupportedpastels), Hermes Agent"
    )
    assert _frontmatter_value(frontmatter, "license") == "MIT"
    assert _frontmatter_value(frontmatter, "category") == "security"
    platforms = _frontmatter_value(frontmatter, "platforms").strip("[]").split(",")
    assert {platform.strip() for platform in platforms} == {"linux", "macos", "windows"}
    assert body.strip()


def test_modern_sections_are_in_order(skill_text: str) -> None:
    positions = [skill_text.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_related_skills_resolve_in_repo(frontmatter_and_body: tuple[str, str]) -> None:
    frontmatter, _ = frontmatter_and_body
    raw = _frontmatter_value(frontmatter, "related_skills").strip("[]")
    for name in (part.strip() for part in raw.split(",")):
        hits = list((REPO_ROOT / "skills").glob(f"**/{name}/SKILL.md"))
        hits += list((REPO_ROOT / "optional-skills").glob(f"**/{name}/SKILL.md"))
        assert hits, f"related skill does not ship: {name}"


def test_owasp_2025_changed_code_categories_are_complete(skill_text: str) -> None:
    assert "OWASP Top 10:2025" in skill_text
    for category in OWASP_2025_CATEGORIES:
        assert category in skill_text
    assert "SSRF" in skill_text


def test_a02_distinguishes_edge_https_from_local_http(skill_text: str) -> None:
    normalized = " ".join(skill_text.split())
    for phrase in (
        "HTTPS and edge termination (A02)",
        "Managed static site or SPA (the common path)",
        "`publish-site` sends a static output directory",
        "provider owns the public origin and HTTPS",
        "do not invent those findings",
        "Server-rendered app or API",
        "`http://localhost`",
        "terminates HTTPS at the edge",
        "public/share URL must use HTTPS",
        "redirects to the same host over HTTPS",
        "mixed content",
        "`ws://`",
        "directly reachable HTTP origin",
        "session cookies are still marked `Secure`",
        "managed static site with provider HTTPS",
        "Start with the common static path",
    ):
        assert phrase in normalized
    assert "Do not tell the user to add application-level TLS solely" in normalized


def test_pr_only_reviews_route_to_github_code_review(skill_text: str) -> None:
    normalized = " ".join(skill_text.split())
    assert "ordinary PR-only review with no pending publication" in normalized
    assert "use `github-code-review`" in normalized
    assert "PR's code or generated artifact is about to become public" in normalized
    assert "or upon the user's direct request" in normalized


def test_gate_distinguishes_blockers_from_coverage_gaps(skill_text: str) -> None:
    normalized = " ".join(skill_text.split())
    assert 'python "${HERMES_SKILL_DIR}/scripts/inventory_publish_surface.py"' in skill_text
    assert "review aid, not a security boundary" in normalized
    assert "OS-level isolation" in normalized
    for outcome in ("Ready to publish", "Fix before publishing", "Need your input"):
        assert outcome in skill_text
    assert "Scanner absence alone is a coverage warning" in normalized
    assert "Never publish merely because a scanner returned zero findings" in normalized
    assert "`review_candidates` entry" in normalized
    assert "every `skipped_files` entry" in normalized
    assert "Stop concurrent writers" in normalized
    assert "git diff --cached --stat" in skill_text


def test_default_report_is_plain_and_actionable(skill_text: str) -> None:
    normalized = " ".join(skill_text.split())
    for heading in ("## Publish safety check", "### Fixes I can make", "### What you need to do", "### Next step"):
        assert heading in skill_text
    assert "Do not lead with severity tables" in normalized
    assert "Group findings that share one fix" in normalized
    assert "one plain-language sentence" in normalized
    assert "Never include a matched secret value" in normalized
    assert "OWASP Changed-Code Pass" not in skill_text


def test_inventory_holds_sensitive_paths_without_reading_vcs_contents(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    (publish / "index.html").write_text("<h1>Safe</h1>\n", encoding="utf-8")
    (publish / ".env.production").write_text("PUBLIC_FLAG=true\n", encoding="utf-8")
    (publish / ".dev.vars").write_text("LOCAL_ONLY=true\n", encoding="utf-8")
    (publish / "production.env").write_text("LOCAL_ONLY=true\n", encoding="utf-8")
    (publish / "secrets.yaml").write_text("placeholder: true\n", encoding="utf-8")
    (publish / "client.pem").write_text("public-or-private-pem\n", encoding="utf-8")
    git_dir = publish / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("secret-value-must-not-appear\n", encoding="utf-8")
    ssh_dir = publish / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_rsa").write_text("another-value-must-not-appear\n", encoding="utf-8")

    result = _run_inventory(publish)
    report = json.loads(result.stdout)

    assert result.returncode == 1
    assert report["decision"] == "HOLD"
    assert {(item["path"], item["rule"]) for item in report["sensitive_paths"]} == {
        (".dev.vars", "credential-file"),
        (".env.production", "environment-file"),
        (".git/", "version-control-metadata"),
        (".ssh/", "credential-directory"),
        ("client.pem", "private-key-or-keystore"),
        ("production.env", "environment-file"),
        ("secrets.yaml", "credential-file"),
    }
    assert "secret-value-must-not-appear" not in result.stdout
    assert "another-value-must-not-appear" not in result.stdout


def test_sensitive_file_name_and_sensitive_root_hold(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    (publish / ".git").write_text("gitdir: hidden-location\n", encoding="utf-8")

    result = _run_inventory(publish)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert {item["rule"] for item in report["sensitive_paths"]} == {
        "version-control-metadata"
    }

    sensitive_root = tmp_path / ".ssh"
    sensitive_root.mkdir()
    (sensitive_root / "notice.txt").write_text("no credential here\n", encoding="utf-8")
    root_result = _run_inventory(sensitive_root)
    root_report = json.loads(root_result.stdout)
    assert root_result.returncode == 1
    assert {item["path"]: item["rule"] for item in root_report["sensitive_paths"]}["."] == (
        "credential-directory"
    )


def test_non_regular_entries_and_walk_errors_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_inventory_module()
    publish = tmp_path / "dist"
    publish.mkdir()
    special = publish / "special-entry"
    special.write_text("placeholder\n", encoding="utf-8")
    original_is_file = Path.is_file

    def fake_is_file(path: Path) -> bool:
        return False if path == special else original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    report = module.inventory(publish, module.DEFAULT_MAX_FILE_BYTES)
    assert report["decision"] == "HOLD"
    assert report["sensitive_paths"] == [
        {"path": "special-entry", "rule": "non-regular-file"}
    ]

    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()

    def fake_walk(root: Path, **kwargs: object):
        onerror = kwargs["onerror"]
        assert callable(onerror)
        onerror(PermissionError(13, "denied", str(Path(root) / "blocked")))
        return iter(())

    monkeypatch.setattr(module.os, "walk", fake_walk)
    error_report = module.inventory(unreadable, module.DEFAULT_MAX_FILE_BYTES)
    assert error_report["decision"] == "HOLD"
    assert error_report["sensitive_paths"] == [
        {"path": "blocked/", "rule": "directory-read-error"}
    ]


def test_inventory_reports_secret_locations_but_never_values(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    token = "gh" + "p_" + ("A" * 32)
    source = publish / "app.js"
    source.write_text(
        "const harmless = true;\n"
        f"const deployedToken = '{token}';\n"
        "const password = 'review-only-value';\n",
        encoding="utf-8",
    )

    result = _run_inventory(publish)
    report = json.loads(result.stdout)

    assert result.returncode == 1
    assert report["secret_candidates"] == [
        {"line": 2, "path": "app.js", "rule": "github-token"}
    ]
    assert {item["rule"] for item in report["review_candidates"]} == {
        "literal-credential-assignment"
    }
    assert token not in result.stdout
    assert "review-only-value" not in result.stdout


def test_clean_artifact_returns_review_not_security_claim(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    (publish / "index.html").write_text("<h1>Hello</h1>\n", encoding="utf-8")
    assets = publish / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('hello');\n", encoding="utf-8")

    result = _run_inventory(publish)
    report = json.loads(result.stdout)

    assert result.returncode == 0
    assert report["decision"] == "REVIEW"
    assert report["summary"]["blocking_findings"] == 0
    assert "not proof" in report["note"]


def test_oversize_text_file_is_a_review_gap(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    (publish / "large.js").write_text("0123456789", encoding="utf-8")

    result = _run_inventory(publish, "--max-file-bytes", "5")
    report = json.loads(result.stdout)

    assert result.returncode == 0
    assert report["skipped_files"] == [{"path": "large.js", "reason": "size-limit"}]
    assert report["review_candidates"] == [
        {"path": "large.js", "rule": "unscanned-oversize-file"}
    ]


def test_binary_file_is_an_explicit_coverage_gap(tmp_path: Path) -> None:
    publish = tmp_path / "dist"
    publish.mkdir()
    token = "gh" + "p_" + ("B" * 32)
    (publish / "opaque.bin").write_bytes(b"\x00" + token.encode("ascii"))

    result = _run_inventory(publish)
    report = json.loads(result.stdout)

    assert result.returncode == 0
    assert report["decision"] == "REVIEW"
    assert report["skipped_files"] == [{"path": "opaque.bin", "reason": "binary"}]
    assert token not in result.stdout


def test_inventory_module_imports_without_third_party_dependencies() -> None:
    module = _load_inventory_module()
    assert module.DEFAULT_MAX_FILE_BYTES > 0
