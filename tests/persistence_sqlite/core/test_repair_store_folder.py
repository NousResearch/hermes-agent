"""State.db repair loads the CJK tokenizer from the folder of the file it repairs, as the store does: a store's cjk
index is probed and rebuilt the same way in the ambient home and in a profile folder outside it."""

import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from persistence.sqlite import SQLiteBackend
from persistence.sqlite._state import repair
from tests.persistence_sqlite.conftest import profile_home

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "native" / "fts5_cjk" / "fts5_cjk.c"
VENDOR = REPO / "native" / "fts5_cjk" / "vendor"
_CJK_HITS = "SELECT COUNT(*) FROM messages_fts_cjk WHERE messages_fts_cjk MATCH '일본'"


@pytest.fixture(scope="session")
def cjk_so(tmp_path_factory):
    if shutil.which("gcc") is None or not SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    out = tmp_path_factory.mktemp("fts5cjk") / "libfts5_cjk.so"
    try:
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", f"-I{VENDOR}", str(SRC), "-o", str(out)],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"tokenizer build failed: {e.stderr[:200]}")
    probe = sqlite3.connect(":memory:")
    try:
        probe.enable_load_extension(True)
        probe.load_extension(str(out))
    except Exception as e:
        pytest.skip(f"extension loading unavailable: {e}")
    finally:
        probe.close()
    return out


def _cjk_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(str(db_path.parent / "lib" / "libfts5_cjk.so"))
    conn.enable_load_extension(False)
    return conn


@pytest.fixture(params=["ambient home", "profile folder"])
def cjk_store(request, cjk_so, scope_a, monkeypatch):
    """A state.db holding one indexed CJK message, the tokenizer library in the store's folder only."""
    monkeypatch.delenv("HERMES_FTS5_CJK_SO", raising=False)
    folder = get_hermes_home() if request.param == "ambient home" else profile_home(scope_a.name)
    db_path = folder / "state.db"
    (folder / "lib").mkdir(parents=True)
    shutil.copy(cjk_so, folder / "lib" / "libfts5_cjk.so")
    backend = SQLiteBackend({scope_a.hermes_home_key: db_path})
    try:
        backend.open(scope_a).ready()
    finally:
        backend.close()
    with closing(_cjk_conn(db_path)) as conn:
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 0)")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', '일본 MCP 후보', 0)")
        conn.commit()
        assert conn.execute(_CJK_HITS).fetchone()[0] == 1
    return db_path


def test_a_healthy_store_probes_clean(cjk_store):
    assert repair._db_opens_cleanly(cjk_store) is None


def test_the_probe_reports_a_damaged_cjk_index(cjk_store):
    with closing(sqlite3.connect(cjk_store)) as conn:
        conn.execute("UPDATE messages_fts_cjk_data SET block = x'00' WHERE id = 10")
        conn.commit()
    assert repair._db_opens_cleanly(cjk_store) is not None


def test_the_fts_rebuild_strategy_rebuilds_the_cjk_index(cjk_store):
    with closing(_cjk_conn(cjk_store)) as conn:
        conn.execute("INSERT INTO messages_fts_cjk(messages_fts_cjk) VALUES('delete-all')")
        conn.commit()
        assert conn.execute(_CJK_HITS).fetchone()[0] == 0
    report = repair._run_repair_strategies(cjk_store, {"repaired": False, "strategy": None, "error": None})
    assert (report["repaired"], report["strategy"]) == (True, "rebuild_fts")
    with closing(_cjk_conn(cjk_store)) as conn:
        assert conn.execute(_CJK_HITS).fetchone()[0] == 1
