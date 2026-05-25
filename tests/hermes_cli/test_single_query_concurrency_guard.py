import cli


def test_single_query_guard_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    monkeypatch.setattr(cli, "_single_query_guard_config", lambda: (0, 0.0))

    with cli._single_query_concurrency_guard("hello") as slot:
        assert slot is None


def test_single_query_guard_rejects_live_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    monkeypatch.setattr(cli, "_single_query_guard_config", lambda: (1, 0.0))
    slot_dir = tmp_path / "runtime" / "single-query-slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    (slot_dir / "slot-0.lock").write_text('{"pid": %d}' % cli.os.getpid(), encoding="utf-8")

    try:
        with cli._single_query_concurrency_guard("hello"):
            raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "single-query concurrency limit reached (1)" in str(exc)


def test_single_query_guard_reclaims_stale_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    monkeypatch.setattr(cli, "_single_query_guard_config", lambda: (1, 0.0))
    monkeypatch.setattr(cli, "_pid_is_running", lambda pid: False)
    slot_dir = tmp_path / "runtime" / "single-query-slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    stale = slot_dir / "slot-0.lock"
    stale.write_text('{"pid": 999999}', encoding="utf-8")

    with cli._single_query_concurrency_guard("hello") as slot:
        assert slot == stale
        assert stale.exists()
        payload = stale.read_text(encoding="utf-8")
        assert str(cli.os.getpid()) in payload

    assert not stale.exists()