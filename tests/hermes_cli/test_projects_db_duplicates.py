"""One folder, one project: concurrent creation cannot duplicate a primary path, and rows that
already share one resolve to the same project everywhere (lookup, listing order)."""

from __future__ import annotations

import threading

from hermes_cli import projects_db as pdb


def test_concurrent_creates_of_one_folder_keep_a_single_project(tmp_path, monkeypatch):
    path = tmp_path / "db"
    pdb.connect(db_path=path).close()
    # Hold both creators between their duplicate check and their insert. A check outside the
    # write transaction lets both pass; inside it, the second creator waits for the first to
    # commit, the barrier times out, and the second then sees the row.
    barrier = threading.Barrier(2, timeout=2)
    check = pdb.find_by_primary_path

    def racing_check(*args, **kwargs):
        found = check(*args, **kwargs)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return found

    monkeypatch.setattr(pdb, "find_by_primary_path", racing_check)
    outcomes: list[str] = []

    def create():
        conn = pdb.connect(db_path=path)
        try:
            outcomes.append(pdb.create_project(conn, name="repo", folders=["/www/repo"]))
        except ValueError:
            outcomes.append("refused")
        finally:
            conn.close()

    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    conn = pdb.connect(db_path=path)
    try:
        assert len(pdb.list_projects(conn)) == 1
    finally:
        conn.close()
    assert sorted(outcome == "refused" for outcome in outcomes) == [False, True]


def test_projects_sharing_a_folder_resolve_to_the_same_one_everywhere(tmp_path):
    conn = pdb.connect(db_path=tmp_path / "db")
    try:
        # Same second, as a lost race leaves them; insertion order disagrees with id order.
        for pid, slug in (("p_bbbb", "repo-2"), ("p_aaaa", "repo-3")):
            conn.execute("INSERT INTO projects (id, slug, name, primary_path, created_at) VALUES (?, ?, 'repo', '/www/repo', 7)",
                         (pid, slug))
            conn.execute("INSERT INTO project_folders (project_id, path, is_primary, added_at) VALUES (?, '/www/repo', 1, 7)",
                         (pid,))
        conn.commit()

        listed = [p.id for p in pdb.list_projects(conn)]
        assert listed == ["p_bbbb", "p_aaaa"]
        for found in (pdb.find_by_primary_path(conn, "/www/repo"), pdb.project_for_path(conn, "/www/repo"),
                      pdb.project_for_path(conn, "/www/repo/src")):
            assert found is not None and found.id == "p_bbbb"
    finally:
        conn.close()
