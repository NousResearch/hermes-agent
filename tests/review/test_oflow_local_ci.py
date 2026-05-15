from __future__ import annotations

from pathlib import Path

from scripts.review import oflow_local_ci


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_denylist_flags_service_and_remote_mutation_commands(tmp_path: Path) -> None:
    write(
        tmp_path / "scripts" / "unsafe.sh",
        "systemctl restart hermes\n"
        "docker compose up -d\n"
        "ssh prod.example.com\n"
        "sqlite3 app.db 'update orders set status=1'\n",
    )

    violations = oflow_local_ci.scan_file("scripts/unsafe.sh", repo=tmp_path)

    checks = {violation.check for violation in violations}
    assert "systemctl" in checks
    assert "docker compose up/down" in checks
    assert "ssh" in checks
    assert "sqlite mutation" in checks


def test_secret_paths_are_reported_without_reading_contents(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TOKEN=super-secret\n", encoding="utf-8")

    violations = oflow_local_ci.scan_file(".env", repo=tmp_path)

    assert violations == [
        oflow_local_ci.Violation(
            file=".env",
            check="secret path",
            line=None,
            detail="secret-bearing paths are not read by this helper",
        )
    ]


def test_safe_review_helper_passes_denylist(tmp_path: Path) -> None:
    write(
        tmp_path / "scripts" / "review" / "safe.py",
        "from pathlib import Path\n"
        "Path('artifacts/summary.json').write_text('{}')\n",
    )

    assert oflow_local_ci.scan_file("scripts/review/safe.py", repo=tmp_path) == []


def test_runtime_and_database_paths_are_blocked_without_command_content(tmp_path: Path) -> None:
    write(tmp_path / "runtime" / "README.md", "runtime docs\n")
    write(tmp_path / "migrations" / "001.sql", "-- migration docs\n")

    runtime_violations = oflow_local_ci.scan_file("runtime/README.md", repo=tmp_path)
    migration_violations = oflow_local_ci.scan_file("migrations/001.sql", repo=tmp_path)

    assert any(v.check == "runtime path" for v in runtime_violations)
    assert any(v.check == "database mutation path" for v in migration_violations)
