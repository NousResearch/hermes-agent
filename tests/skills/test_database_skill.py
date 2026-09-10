"""Contract tests for the bundled database skill."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "software-development" / "database" / "SKILL.md"
REQUIRED_SECTIONS = [
    "# Database Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter_value(name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", _content(), re.MULTILINE)
    assert match, f"missing {name} frontmatter"
    return match.group(1).strip()


def test_skill_exists_and_has_valid_frontmatter():
    content = _content()
    assert content.startswith("---\n")
    assert "\n---\n\n# Database Skill\n" in content
    assert _frontmatter_value("name") == "database"


def test_description_matches_hardline_limit():
    description = _frontmatter_value("description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert "\n" not in description


def test_platforms_match_posix_shell_examples():
    assert _frontmatter_value("platforms") == "[linux, macos]"
    content = _content()
    for platform in ("Debian/Ubuntu", "macOS"):
        assert platform in content
    assert "winget install" not in content
    assert "gated to Linux and macOS" in content


def test_modern_sections_exist_in_required_order():
    content = _content()
    positions = [content.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_termination_example_requires_one_reviewed_pid():
    content = _content()
    assert "pg_terminate_backend(PID)" not in content
    assert "WHERE pid = 12345" in content
    assert "AND datname = current_database()" in content
    assert "AND pid <> pg_backend_pid()" in content
    assert "reviews and approves one numeric PID" in content
    assert "Do not terminate all matching sessions" in content


def test_skill_forbids_credentials_in_process_arguments():
    content = _content()
    assert "$DATABASE_URL" not in content
    assert "postgresql://" not in content
    assert "postgres://" not in content
    assert "PGPASSWORD=" not in content
    assert re.search(r"postgres(?:ql)?://[^\s]*:[^@\s]+@", content) is None
    assert (
        re.search(
            r"\b(?:psql|pg_dump|pg_restore)\b[^\n`]*\$[A-Z][A-Z0-9_]*URL\b",
            content,
        )
        is None
    )
    assert "password=" not in content.lower()
    assert "Never embed a password in a connection" in content
    assert ".pgpass" in content
    assert "PGSERVICE" in content
    assert 'dbname="service=restore-target"' in content


def test_connection_checks_and_mysql_commands_keep_the_confirmed_target():
    content = _content()
    assert "\\conninfo" in content
    assert "inet_server_addr()" in content
    assert "inet_server_port()" in content
    assert "@@hostname" in content
    assert "@@port" in content
    assert "`mysql -p" not in content
    for line in content.splitlines():
        if line.startswith("mysql ") and "--version" not in line:
            assert '-h "$' in line or line.endswith("\\")
            assert '-u "$' in line or line.endswith("\\")


def test_sqlite_backup_and_restore_formats_match():
    content = _content()
    assert 'sqlite3 app.db ".backup \'backup.db\'"' in content
    assert 'sqlite3 restored.db ".restore \'backup.db\'"' in content
    assert "sqlite3 restored.db < backup.sql" not in content


def test_skill_requires_approval_for_destructive_operations():
    content = _content()
    assert "obtaining approval" in content
    assert "explicitly approved administrative operation" in content
    assert "-pMyPassword" not in content
