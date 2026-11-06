"""Generated titles remain bounded and preserve lineage numbering."""
import pytest
from hermes_state import SessionDB

@pytest.mark.parametrize("base", ["X"*98, "Y"*100, "Z"*101, "雪"*100, "a_%"])
def test_generated_title_bounds_and_number_rollover(tmp_path, base):
    with SessionDB(tmp_path / "state.db") as db:
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('parent','test',1,?)", (base,))
        db._conn.commit()
        generated = []
        for n in range(2, 12):
            title = db.get_next_title_in_lineage(base)
            assert len(title) <= db.MAX_TITLE_LENGTH
            assert title.endswith(f" #{n}"), (base, title, n)
            assert title not in generated
            db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES (?,?,1,?)", (str(n), "test", title))
            db._conn.commit()
            generated.append(title)
        assert db.get_next_title_in_lineage(base + " #9").endswith(" #12")

def test_generated_ordinary_empty_and_numbered_titles(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        assert db.get_next_title_in_lineage("new") == "new"
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('p','test',1,'name')")
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('q','test',1,'name #9')")
        db._conn.commit()
        assert db.get_next_title_in_lineage("name #9") == "name #10"


def test_numbered_one_still_advances_when_base_is_absent(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('p','test',1,'name #1')")
        db._conn.commit()
        assert db.get_next_title_in_lineage("name") == "name #2"


@pytest.mark.parametrize("existing,expected", [
    ("foo #2", "Foo #3"),
    ("Foo #2 #9", "Foo #10"),
])
def test_existing_lineage_prefix_numbering_is_preserved(tmp_path, existing, expected):
    with SessionDB(tmp_path / "state.db") as db:
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('p','test',1,'Foo')")
        db._conn.execute("INSERT INTO sessions(id,source,started_at,title) VALUES ('q','test',1,?)", (existing,))
        db._conn.commit()
        assert db.get_next_title_in_lineage("Foo") == expected
