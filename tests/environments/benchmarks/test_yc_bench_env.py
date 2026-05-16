import pathlib
import sqlite3
import sys
from unittest import mock
import pytest

# We cannot globally mock sys.modules['atroposlib'] here because it leaks to other pytest modules
# and breaks tests that do conditional `pytest.skip("atroposlib not installed")`.

try:
    import atroposlib
except ImportError:

    class MockClass:
        pass

    class MockAtroposlib:
        class type_definitions:
            Item = MockClass

        class envs:
            class base:
                EvalHandlingEnum = MockClass
                BaseEnv = MockClass
                BaseEnvConfig = MockClass
                ScoredDataGroup = MockClass
                ScoredDataItem = MockClass

            class server_handling:
                class server_manager:
                    APIServerConfig = MockClass
                    ServerBaseline = MockClass
                    ServerManager = MockClass

                class openai_server:
                    OpenAIServerConfig = MockClass

    # Temporarily patch sys.modules so the import of yc_bench_env succeeds
    # Then we restore the original state so we don't break other tests.
    original_modules = dict(sys.modules)

    sys.modules["atroposlib"] = MockAtroposlib
    sys.modules["atroposlib.envs"] = MockAtroposlib.envs
    sys.modules["atroposlib.envs.base"] = MockAtroposlib.envs.base
    sys.modules["atroposlib.envs.server_handling"] = MockAtroposlib.envs.server_handling
    sys.modules["atroposlib.envs.server_handling.server_manager"] = (
        MockAtroposlib.envs.server_handling.server_manager
    )
    sys.modules["atroposlib.envs.server_handling.openai_server"] = (
        MockAtroposlib.envs.server_handling.openai_server
    )
    sys.modules["atroposlib.type_definitions"] = MockAtroposlib.type_definitions

    from environments.benchmarks.yc_bench.yc_bench_env import _read_final_score

    sys.modules.clear()
    sys.modules.update(original_modules)
else:
    from environments.benchmarks.yc_bench.yc_bench_env import _read_final_score


def test_missing_db_file() -> None:
    """Test behavior when the database file does not exist."""
    with mock.patch("os.path.exists", return_value=False):
        result = _read_final_score("/fake/path/db.sqlite")

    assert isinstance(result, dict)
    assert type(result["final_funds_cents"]) is int
    assert type(result["survived"]) is bool
    assert type(result["terminal_reason"]) is str

    assert result["final_funds_cents"] == 0
    assert result["survived"] is False
    assert result["terminal_reason"] == "db_missing"


def test_db_read_error(tmp_path: pathlib.Path) -> None:
    """Test behavior when the database encounters an operational error."""
    db_file = tmp_path / "test.sqlite"
    db_file.write_text("fake db content")  # Ensure file exists to pass os.path.exists

    with mock.patch(
        "sqlite3.connect", side_effect=sqlite3.DatabaseError("Corrupted file")
    ):
        result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 0
    assert result["survived"] is False
    assert result["terminal_reason"] == "db_error: Corrupted file"


def create_mock_db(
    db_path: pathlib.Path | str,
    companies_rows: list[int] | None,
    sim_events_rows: list[tuple[str, float]] | None = None,
) -> None:
    """Helper to create a temporary DB with given rows."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE companies (funds_cents INTEGER)")
    if companies_rows is not None:
        for funds in companies_rows:
            cur.execute("INSERT INTO companies (funds_cents) VALUES (?)", (funds,))

    if sim_events_rows is not None:
        cur.execute("CREATE TABLE sim_events (event_type TEXT, scheduled_at REAL)")
        for event, sched in sim_events_rows:
            cur.execute(
                "INSERT INTO sim_events (event_type, scheduled_at) VALUES (?, ?)",
                (event, sched),
            )
    conn.commit()
    conn.close()


def test_happy_path_survived(tmp_path: pathlib.Path) -> None:
    """Test standard survival scenario."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(
        db_file, companies_rows=[1500000], sim_events_rows=[("horizon_end", 100.0)]
    )
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 1500000
    assert result["survived"] is True
    assert result["terminal_reason"] == "horizon_end"


def test_happy_path_bankrupt(tmp_path: pathlib.Path) -> None:
    """Test standard bankruptcy scenario."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(
        db_file, companies_rows=[-50000], sim_events_rows=[("bankruptcy", 100.0)]
    )
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == -50000
    assert result["survived"] is False
    assert result["terminal_reason"] == "bankruptcy"


def test_boundary_funds_zero(tmp_path: pathlib.Path) -> None:
    """Test boundary condition where funds_cents is exactly 0."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(
        db_file, companies_rows=[0], sim_events_rows=[("horizon_end", 100.0)]
    )
    result = _read_final_score(str(db_file))

    # 0 funds without bankruptcy means survival based on >= 0 logic
    assert result["final_funds_cents"] == 0
    assert result["survived"] is True
    assert result["terminal_reason"] == "horizon_end"


def test_empty_companies_table(tmp_path: pathlib.Path) -> None:
    """Test when the companies table exists but is empty."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(db_file, companies_rows=[], sim_events_rows=[("horizon_end", 100.0)])
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 0
    assert result["survived"] is True
    assert result["terminal_reason"] == "horizon_end"


def test_empty_sim_events_table(tmp_path: pathlib.Path) -> None:
    """Test when the sim_events table exists but is empty."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(db_file, companies_rows=[100], sim_events_rows=[])
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 100
    assert result["survived"] is True
    assert result["terminal_reason"] == "unknown"


def test_missing_sim_events_table(tmp_path: pathlib.Path) -> None:
    """Test when the sim_events table is entirely missing."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(db_file, companies_rows=[100], sim_events_rows=None)
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 100
    assert result["survived"] is True
    assert result["terminal_reason"] == "unknown"


def test_casing_and_whitespace_robustness(tmp_path: pathlib.Path) -> None:
    """Test terminal_reason robust parsing for casing and whitespace."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(
        db_file, companies_rows=[100], sim_events_rows=[("  BankRuptcy  ", 100.0)]
    )
    result = _read_final_score(str(db_file))

    assert result["final_funds_cents"] == 100
    assert result["survived"] is False
    assert result["terminal_reason"] == "bankruptcy"


def test_return_shape(tmp_path: pathlib.Path) -> None:
    """Test that the return dictionary has exactly the right shape and types."""
    db_file = tmp_path / "test.sqlite"
    create_mock_db(
        db_file, companies_rows=[100], sim_events_rows=[("horizon_end", 100.0)]
    )
    result = _read_final_score(str(db_file))

    # Assert exact keys
    assert set(result.keys()) == {"final_funds_cents", "survived", "terminal_reason"}

    # Assert exact types
    assert type(result["final_funds_cents"]) is int
    assert type(result["survived"]) is bool
    assert type(result["terminal_reason"]) is str
