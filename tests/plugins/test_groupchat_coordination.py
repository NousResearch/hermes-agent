"""Work arbitration distinguishes explicit hand-off from ordinary completion."""
from plugins.groupchat.coordination import WorkRound, WorkStore


def signal(actor, state="processing", seq=0, stopped=False, started=100):
    return dict(message_id="$request", run=actor, seq=seq, started=started,
                state=state, stopped=stopped, note="")


def test_explicit_work_and_fallback_are_one_shot_and_request_scoped(tmp_path):
    a = WorkRound("$request", "a")
    a.observe("a", signal("a"), 100)
    assert not a.observe("b", dict(signal("b"), message_id="$different"), 100)
    a.observe("b", signal("b", started=102), 102)
    assert a.next_action(111, 10) is None
    assert a.next_action(112, 10)[0] == "notice"
    assert a.next_action(112, 20) is None
    a.observe("a", signal("a", "working", 1), 103)
    a.observe("b", signal("b", "contributing", 1, started=102), 103)
    assert a.next_action(103, 20)[0] == "notice"
    a.acknowledge("notice")
    store = WorkStore(tmp_path / "groupchat" / "coordination.sqlite3")
    store.save("room/request", a)
    restored = store.load("room/request")
    assert restored.next_action(104, 10) is None
    assert not restored.observe("b", signal("b"), 104)


def test_only_all_explicit_stopped_yields_resume_exactly_first(tmp_path):
    a, b = WorkRound("$request", "a"), WorkRound("$request", "b")
    for round in (a, b):
        round.observe("a", signal("a", "yielded", stopped=True), 100)
        round.observe("b", signal("b", "completed", stopped=True, started=101), 101)
        assert round.next_action(102, 10) is None
        round.observe("b", signal("b", "yielded", seq=1, started=101), 102)
        assert round.next_action(102, 10) is None  # still finishing its tool/turn
        round.observe("b", signal("b", "yielded", seq=2, stopped=True, started=101), 103)
    assert a.next_action(103, 10)[0] == "resume"
    assert b.next_action(103, 10) is None
    assert a.next_action(300, 10)[0] == "resume"  # an explicit stopped yield is durable
    a.participants["b"]["state"] = "processing"
    assert a.next_action(300, 10) is None  # expired processing is not a hand-off
    a.participants["b"]["state"] = "yielded"
    a.acknowledge("resume")
    store = WorkStore(tmp_path / "coordination.sqlite3")
    store.save("room/request", a)
    assert store.load("room/request").next_action(104, 10) is None
