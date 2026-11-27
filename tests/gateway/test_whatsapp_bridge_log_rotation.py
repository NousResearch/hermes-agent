"""bridge.log is rotated at bridge spawn so it cannot grow without bound (#149860).

The bridge holds its stdout/stderr handle for its whole life, so the adapter rotates the file
when it opens a fresh handle, keeping one ``.1`` generation.
"""


def test_oversized_bridge_log_rotates_to_one_generation(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp import adapter as wa

    monkeypatch.setattr(wa, "_BRIDGE_LOG_ROTATE_BYTES", 16)
    logs = tmp_path / "whatsapp"
    logs.mkdir()
    log = logs / "bridge.log"
    log.write_text("first generation, over the cap\n", encoding="utf-8")
    (logs / "bridge.log.1").write_text("oldest\n", encoding="utf-8")

    with wa._open_bridge_log(log, private=False) as fh:
        fh.write("fresh\n")

    assert log.read_text(encoding="utf-8") == "fresh\n"
    assert (logs / "bridge.log.1").read_text(encoding="utf-8") == "first generation, over the cap\n"
    assert sorted(p.name for p in logs.iterdir()) == ["bridge.log", "bridge.log.1"]


def test_small_bridge_log_keeps_appending(tmp_path):
    from plugins.platforms.whatsapp import adapter as wa

    log = tmp_path / "bridge.log"
    log.write_text("earlier run\n", encoding="utf-8")

    with wa._open_bridge_log(log, private=False) as fh:
        fh.write("this run\n")

    assert log.read_text(encoding="utf-8") == "earlier run\nthis run\n"
    assert not (tmp_path / "bridge.log.1").exists()
