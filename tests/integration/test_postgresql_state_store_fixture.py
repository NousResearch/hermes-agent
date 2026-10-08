"""Real PostgreSQL 18 + pgvector named-volume fixture contract."""

import os
import subprocess

import pytest


def _fixture_target() -> tuple[str, str]:
    container = os.environ.get("HERMES_TEST_PG18_CONTAINER")
    volume = os.environ.get("HERMES_TEST_PG18_VOLUME")
    if (container is None) != (volume is None) or (container is not None and (not container or not volume)):
        raise ValueError("HERMES_TEST_PG18_CONTAINER and HERMES_TEST_PG18_VOLUME must both be set or both absent")
    return (
        container or "hermes-agent-postgresql-state-store-dev",
        volume or "hermes-agent-postgresql-state-store-pgdata",
    )


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=True, text=True, capture_output=True)


def test_postgresql_state_store_fixture_is_live_with_named_pgdata_and_vector():
    container, volume = _fixture_target()
    inspect = _docker(
        "inspect", container,
        "--format", "{{range .Mounts}}{{printf \"%s|%s|%s\\n\" .Name .Destination .Type}}{{end}}",
    )
    mounts = inspect.stdout.strip().splitlines()
    assert mounts == [f"{volume}|/var/lib/postgresql|volume"]

    ready = _docker(
        "exec", container, "pg_isready",
        "-U", "hermes_state_store_test", "-d", "hermes_state_store_test",
    )
    assert "accepting connections" in ready.stdout

    extensions = _docker(
        "exec", container, "psql", "-At", "-v", "ON_ERROR_STOP=1",
        "-U", "hermes_state_store_test", "-d", "hermes_state_store_test",
        "-c", "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY extname",
    )
    assert extensions.stdout.strip().splitlines() == ["pg_trgm", "vector"]


@pytest.mark.parametrize(
    ("container", "volume", "expected"),
    [
        (None, None, ("hermes-agent-postgresql-state-store-dev", "hermes-agent-postgresql-state-store-pgdata")),
        ("pg18-owned", "pg18-owned-data", ("pg18-owned", "pg18-owned-data")),
    ],
)
def test_pg18_target_selection_uses_matching_container_and_volume(monkeypatch, container, volume, expected):
    _set_target_env(monkeypatch, container, volume)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "inspect":
            output = f"{expected[1]}|/var/lib/postgresql|volume\n"
        elif command[3] == "pg_isready":
            output = "accepting connections\n"
        else:
            output = "pg_trgm\nvector\n"
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    test_postgresql_state_store_fixture_is_live_with_named_pgdata_and_vector()
    assert [(call[1], call[2]) for call in calls] == [
        ("inspect", expected[0]), ("exec", expected[0]), ("exec", expected[0]),
    ]


def _set_target_env(monkeypatch, container, volume):
    for name, value in (("HERMES_TEST_PG18_CONTAINER", container), ("HERMES_TEST_PG18_VOLUME", volume)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(("container", "volume"), [("pg18-owned", None), (None, "pg18-owned-data"), ("", "pg18-owned-data"), ("pg18-owned", "")])
def test_pg18_target_selection_rejects_incomplete_pair_before_docker(monkeypatch, container, volume):
    _set_target_env(monkeypatch, container, volume)

    def fail_docker(*_args, **_kwargs):
        pytest.fail("Docker must not be called for an incomplete PG18 target pair")

    monkeypatch.setattr(subprocess, "run", fail_docker)
    with pytest.raises(ValueError, match="must both be set or both absent"):
        test_postgresql_state_store_fixture_is_live_with_named_pgdata_and_vector()


def test_pg18_target_selection_rejects_wrong_named_volume_before_exec(monkeypatch):
    _set_target_env(monkeypatch, "pg18-owned", "pg18-owned-data")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert command[1] == "inspect"
        return subprocess.CompletedProcess(command, 0, stdout="wrong-volume|/var/lib/postgresql|volume\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AssertionError):
        test_postgresql_state_store_fixture_is_live_with_named_pgdata_and_vector()
    assert [(call[1], call[2]) for call in calls] == [("inspect", "pg18-owned")]
