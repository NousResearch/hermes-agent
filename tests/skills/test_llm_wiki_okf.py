"""Tests for the llm-wiki bundled skill's OKF v0.2 tooling.

Covers the SKILL.md authoring standards and the behavior of
``scripts/okf.py`` — the conformance checker and migrator that backs the
skill's optional Open Knowledge Format support.

The load-bearing properties, in the order a reviewer should care about them:

* ``check`` separates OKF §11 hard failures from soft issues, because the
  spec says consumers MUST NOT reject a bundle over the soft ones.
* ``migrate`` turns a default Karpathy-style wiki into a conformant bundle —
  the claim the skill makes — verified by running ``check`` on the result.
* Both subcommands are dry-run by default and idempotent, so a second
  ``--write`` is a no-op rather than compounding edits.
* Frontmatter edits are text surgery: unrelated keys, comments, and ordering
  survive, and a legacy ISO 8601 ``timestamp`` is not mangled by a YAML
  round-trip.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "research" / "llm-wiki"
SCRIPT = SKILL_DIR / "scripts" / "okf.py"


@pytest.fixture(scope="module")
def skill_source() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_source: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def okf():
    spec = importlib.util.spec_from_file_location("llm_wiki_okf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # register before exec: @dataclass resolves annotations via sys.modules
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - defensive cleanup
        sys.modules.pop(spec.name, None)
        raise
    yield module
    sys.modules.pop(spec.name, None)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    """A default Karpathy-style wiki: shaped like OKF, not yet conformant."""
    root = tmp_path / "wiki"
    write(root / "SCHEMA.md", "# Wiki Schema\n\n## Domain\nAI research.\n")
    write(root / "index.md", "# Wiki Index\n\n## Entities\n- [[anthropic]]\n")
    write(
        root / "log.md",
        """\
        # Wiki Log

        > Chronological record of all wiki actions. Append-only.

        ## [2026-05-22] ingest | Karpathy LLM Wiki post
        - Created entities/anthropic.md
        - Updated index.md

        ## [2026-05-15] create | Wiki initialized
        - Domain: AI research
        """,
    )
    write(
        root / "entities" / "anthropic.md",
        """\
        ---
        title: Anthropic
        created: 2026-05-22
        updated: 2026-05-22
        type: entity
        tags: [company, lab]
        sources: [raw/articles/karpathy.md]
        confidence: high
        ---

        # Anthropic

        An AI safety lab. See [[transformer-architecture]].
        """,
    )
    write(
        root / "raw" / "articles" / "karpathy.md",
        """\
        ---
        source_url: https://example.test/post
        ingested: 2026-05-22
        sha256: deadbeef
        ---

        Raw article body.
        """,
    )
    return root


# ---------------------------------------------------------------------------
# Authoring standards
# ---------------------------------------------------------------------------


def test_skill_files_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert SCRIPT.is_file()


def test_description_within_limit(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"


def test_skill_documents_the_script(skill_source: str) -> None:
    assert "scripts/okf.py" in skill_source


def test_skill_cites_okf_v02(skill_source: str, okf) -> None:
    """The skill must not advertise a version the tooling does not implement."""
    assert okf.OKF_VERSION == "0.2"
    assert "v0.2" in skill_source
    assert "okf_version: \"0.2\"" in skill_source


# ---------------------------------------------------------------------------
# check: hard failures vs soft issues (OKF §11)
# ---------------------------------------------------------------------------


def test_check_flags_default_wiki_as_nonconformant(wiki: Path) -> None:
    result = run("check", str(wiki), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["conformant"] is False
    blob = "\n".join(report["hard_failures"])
    # SCHEMA.md has no frontmatter; raw/ has frontmatter but no type; the log
    # uses the skill's own legacy heading format. All three are §11 failures.
    assert "SCHEMA.md" in blob
    assert "raw/articles/karpathy.md" in blob
    assert "log.md" in blob


def test_bare_string_sources_are_soft_not_hard(wiki: Path) -> None:
    report = json.loads(run("check", str(wiki), "--json").stdout)
    assert any("resource" in s for s in report["soft_issues"])
    assert not any("resource" in h for h in report["hard_failures"])


def test_unknown_type_values_are_accepted(wiki: Path) -> None:
    """§4.1: consumers MUST tolerate unknown types."""
    write(
        wiki / "concepts" / "widget.md",
        """\
        ---
        type: Wholly Invented Domain Type
        ---

        # Widget
        """,
    )
    report = json.loads(run("check", str(wiki), "--json").stdout)
    assert not any("widget.md" in h for h in report["hard_failures"])


def test_broken_links_are_not_a_failure(wiki: Path) -> None:
    """§6.1: consumers MUST tolerate broken links."""
    write(
        wiki / "concepts" / "dangling.md",
        """\
        ---
        type: Concept
        ---

        # Dangling

        See [nothing](/concepts/does-not-exist.md).
        """,
    )
    report = json.loads(run("check", str(wiki), "--json").stdout)
    assert not any("dangling.md" in issue for issue in report["hard_failures"])


def test_check_never_writes(wiki: Path) -> None:
    before = {p: p.read_bytes() for p in wiki.rglob("*.md")}
    run("check", str(wiki))
    assert {p: p.read_bytes() for p in wiki.rglob("*.md")} == before


# ---------------------------------------------------------------------------
# migrate: karpathy wiki -> OKF v0.2
# ---------------------------------------------------------------------------


def test_migrate_is_dry_run_by_default(wiki: Path) -> None:
    before = {p: p.read_bytes() for p in wiki.rglob("*.md")}
    result = run("migrate", str(wiki))
    assert "Dry run" in result.stdout
    assert {p: p.read_bytes() for p in wiki.rglob("*.md")} == before


def test_migrate_produces_a_conformant_bundle(wiki: Path) -> None:
    assert run("migrate", str(wiki), "--write").returncode == 0
    verify = run("check", str(wiki), "--json")
    assert verify.returncode == 0, verify.stdout
    assert json.loads(verify.stdout)["conformant"] is True


def test_migrate_is_idempotent(wiki: Path) -> None:
    run("migrate", str(wiki), "--write")
    snapshot = {p: p.read_text(encoding="utf-8") for p in wiki.rglob("*.md")}
    second = run("migrate", str(wiki), "--write")
    assert {p: p.read_text(encoding="utf-8") for p in wiki.rglob("*.md")} == snapshot
    assert "nothing to do" in second.stdout


def test_migrate_titlecases_known_types(wiki: Path) -> None:
    run("migrate", str(wiki), "--write")
    page = (wiki / "entities" / "anthropic.md").read_text(encoding="utf-8")
    assert "type: Entity" in page


def test_migrate_preserves_unrelated_frontmatter_and_body(wiki: Path) -> None:
    """Extension keys survive: §4.1 says consumers preserve unknown keys."""
    run("migrate", str(wiki), "--write")
    page = (wiki / "entities" / "anthropic.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(page.split("---")[1])
    assert fm["confidence"] == "high"
    assert fm["tags"] == ["company", "lab"]
    assert str(fm["created"]) == "2026-05-22"
    assert "See [[transformer-architecture]]." in page


def test_migrate_rewrites_sources_with_resource(wiki: Path) -> None:
    run("migrate", str(wiki), "--write")
    fm = yaml.safe_load(
        (wiki / "entities" / "anthropic.md").read_text(encoding="utf-8").split("---")[1]
    )
    assert fm["sources"] == [
        {"id": "karpathy", "resource": "/raw/articles/karpathy.md"}
    ]


def test_migrate_adds_okf_version_only_at_bundle_root(wiki: Path) -> None:
    """§8: index.md carries no frontmatter except root okf_version."""
    write(wiki / "entities" / "index.md", "# Entities\n\n* [Anthropic](anthropic.md)\n")
    run("migrate", str(wiki), "--write")
    root_fm = yaml.safe_load(
        (wiki / "index.md").read_text(encoding="utf-8").split("---")[1]
    )
    assert root_fm == {"okf_version": "0.2"}
    assert not (wiki / "entities" / "index.md").read_text(encoding="utf-8").startswith("---")


def test_migrate_rewrites_log_to_iso_headings(wiki: Path) -> None:
    """§9: date headings MUST be ISO 8601, newest first."""
    run("migrate", str(wiki), "--write")
    log = (wiki / "log.md").read_text(encoding="utf-8")
    assert "## 2026-05-22" in log
    assert "## [2026-05-22]" not in log
    assert log.index("## 2026-05-22") < log.index("## 2026-05-15")
    # the action survives as prose, and its detail lines stay with it
    assert "**Ingest**: Karpathy LLM Wiki post" in log
    assert "- Created entities/anthropic.md" in log


def test_migrate_leaves_wikilinks_alone(wiki: Path) -> None:
    """Obsidian-first vaults keep working: opting in must not break links."""
    run("migrate", str(wiki), "--write")
    page = (wiki / "entities" / "anthropic.md").read_text(encoding="utf-8")
    assert "[[transformer-architecture]]" in page


def test_migrate_skips_archive_and_dotdirs(wiki: Path) -> None:
    stale = write(wiki / "_archive" / "old.md", "no frontmatter here\n")
    write(wiki / ".obsidian" / "note.md", "internal\n")
    run("migrate", str(wiki), "--write")
    assert stale.read_text(encoding="utf-8") == "no frontmatter here\n"
    assert run("check", str(wiki)).returncode == 0


# ---------------------------------------------------------------------------
# upgrade: OKF v0.1 -> v0.2 (spec §13.1 breaking changes)
# ---------------------------------------------------------------------------


@pytest.fixture
def v01_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    write(root / "index.md", '---\nokf_version: "0.1"\n---\n# Index\n')
    write(
        root / "concepts" / "revenue.md",
        """\
        ---
        type: Metric
        title: Revenue
        timestamp: 2026-06-20T22:53:05Z
        ---

        # Definition

        Revenue is gross sales minus returns.

        # Citations

        - [GA4 schema](https://developers.google.com/analytics/bigquery/export-schema)
        - https://internal.example.test/handbook

        # Notes

        Reviewed quarterly.
        """,
    )
    return root


def test_v01_bundle_is_still_conformant(v01_bundle: Path) -> None:
    """A v0.1 bundle must not be reported as broken — only as upgradable."""
    report = json.loads(run("check", str(v01_bundle), "--json").stdout)
    assert report["conformant"] is True
    assert any("timestamp" in s for s in report["soft_issues"])


def test_upgrade_moves_timestamp_to_generated(v01_bundle: Path) -> None:
    run("upgrade", str(v01_bundle), "--write")
    text = (v01_bundle / "concepts" / "revenue.md").read_text(encoding="utf-8")
    # the literal ISO 8601 string is preserved, not a YAML-coerced datetime
    assert "at: 2026-06-20T22:53:05Z" in text
    assert "timestamp:" not in text


def test_upgrade_moves_citations_to_sources(v01_bundle: Path) -> None:
    run("upgrade", str(v01_bundle), "--write")
    text = (v01_bundle / "concepts" / "revenue.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(text.split("---")[1])
    resources = [s["resource"] for s in fm["sources"]]
    assert "https://developers.google.com/analytics/bigquery/export-schema" in resources
    assert "https://internal.example.test/handbook" in resources
    assert fm["sources"][0]["title"] == "GA4 schema"
    assert "# Citations" not in text
    # sibling sections are untouched
    assert "# Notes" in text
    assert "Reviewed quarterly." in text


def test_upgrade_bumps_declared_version(v01_bundle: Path) -> None:
    run("upgrade", str(v01_bundle), "--write")
    fm = yaml.safe_load(
        (v01_bundle / "index.md").read_text(encoding="utf-8").split("---")[1]
    )
    assert fm["okf_version"] == "0.2"


def test_upgrade_is_idempotent(v01_bundle: Path) -> None:
    run("upgrade", str(v01_bundle), "--write")
    snapshot = {p: p.read_text(encoding="utf-8") for p in v01_bundle.rglob("*.md")}
    run("upgrade", str(v01_bundle), "--write")
    assert {p: p.read_text(encoding="utf-8") for p in v01_bundle.rglob("*.md")} == snapshot


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_missing_directory_exits_two(tmp_path: Path) -> None:
    assert run("check", str(tmp_path / "nope")).returncode == 2


def test_log_migration_preserves_already_iso_logs(okf) -> None:
    already = (
        "# Wiki Log\n\n## 2026-05-22\n* **Update**: something\n\n## 2026-05-15\n"
        "* **Initialization**: created\n"
    )
    assert okf.migrate_log(already) == already
