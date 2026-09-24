"""Real PostgreSQL 18 + pgvector Compose-fixture contract."""

import subprocess


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=True, text=True, capture_output=True)


def test_postgresql_state_store_fixture_is_live_with_named_pgdata_and_vector():
    inspect = _docker(
        "inspect", "hermes-agent-postgresql-state-store-dev",
        "--format", "{{range .Mounts}}{{printf \"%s|%s|%s\\n\" .Name .Destination .Type}}{{end}}",
    )
    mounts = inspect.stdout.strip().splitlines()
    assert mounts == ["hermes-agent-postgresql-state-store-pgdata|/var/lib/postgresql|volume"]

    ready = _docker(
        "exec", "hermes-agent-postgresql-state-store-dev", "pg_isready",
        "-U", "hermes_state_store_test", "-d", "hermes_state_store_test",
    )
    assert "accepting connections" in ready.stdout

    extensions = _docker(
        "exec", "hermes-agent-postgresql-state-store-dev", "psql", "-At", "-v", "ON_ERROR_STOP=1",
        "-U", "hermes_state_store_test", "-d", "hermes_state_store_test",
        "-c", "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY extname",
    )
    assert extensions.stdout.strip().splitlines() == ["pg_trgm", "vector"]
