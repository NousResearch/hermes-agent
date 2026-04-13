import json
from pathlib import Path

from agent.dreaming import DreamingEngine
from tools.memory_tool import MemoryStore


class StubSessionDB:
    def list_sessions_rich(self, limit=5):
        return [
            {"id": "s1", "source": "cli"},
            {"id": "s2", "source": "telegram"},
            {"id": "cron_1", "source": "cron"},
        ]

    def get_messages(self, session_id):
        mapping = {
            "s1": [
                {"role": "user", "content": "Fix login auth"},
                {"role": "assistant", "content": "I will inspect the auth flow."},
            ],
            "s2": [
                {"role": "user", "content": "Build the dreaming cron runner"},
                {"role": "assistant", "content": "I will wire the nightly dream review."},
            ],
            "cron_1": [
                {"role": "user", "content": "ignore me"},
            ],
        }
        return mapping.get(session_id, [])


def make_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    store = MemoryStore(memory_char_limit=2000, user_char_limit=2000)
    store.load_from_disk()
    store.add("user", "User prefers concise updates")
    store.add("memory", "Build command is npm run build")
    return store


def test_dreaming_engine_writes_local_state_and_journal(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(hermes_home=tmp_path / "hermes-home", memory_store=store)

    artifact = engine.run(
        [
            {"role": "user", "content": "Please build the dreaming MVP and keep updates concise."},
            {"role": "assistant", "content": "I will implement the dreaming MVP with a bounded review note."},
        ],
        session_id="s1",
        platform="telegram",
        workspace="/tmp/project",
    )

    assert artifact["session_id"] == "s1"
    assert (tmp_path / "hermes-home" / "dreams" / "state.json").exists()
    assert (tmp_path / "hermes-home" / "dreams" / "dreams.jsonl").exists()

    state = json.loads((tmp_path / "hermes-home" / "dreams" / "state.json").read_text(encoding="utf-8"))
    assert state["last_session_id"] == "s1"
    assert state["run_count"] >= 1

    journal_lines = (tmp_path / "hermes-home" / "dreams" / "dreams.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(journal_lines) == 1
    recorded = json.loads(journal_lines[0])
    assert recorded["session_id"] == "s1"


def test_dreaming_engine_renders_bounded_dream_review(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(hermes_home=tmp_path / "hermes-home", memory_store=store)
    artifact = engine.run(
        [
            {"role": "user", "content": "Fix login auth and build the dreaming MVP."},
            {"role": "assistant", "content": "I will keep the review note bounded and local-only."},
        ],
        session_id="s2",
        platform="cli",
        workspace="/repo",
    )

    text = engine.render_dream_review(artifact)
    assert "# Dream Review" in text
    assert "## Stable candidates" in text
    assert "## Open loops" in text
    assert "## Tomorrow cue" in text
    assert "raw transcript" not in text.lower()


def test_dreaming_engine_collects_recent_messages_from_session_db(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(
        hermes_home=tmp_path / "hermes-home",
        memory_store=store,
        session_db=StubSessionDB(),
    )

    messages = engine.collect_recent_messages(session_limit=5, per_session_limit=4)
    assert any(m["content"] == "Fix login auth" for m in messages)
    assert any(m["content"] == "Build the dreaming cron runner" for m in messages)
    assert all("ignore me" not in m["content"] for m in messages)


def test_dreaming_engine_run_nightly_uses_recent_sessions(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(
        hermes_home=tmp_path / "hermes-home",
        memory_store=store,
        session_db=StubSessionDB(),
    )

    artifact = engine.run_nightly(session_limit=5, per_session_limit=4)
    assert artifact["platform"] == "cron"
    assert artifact["session_id"] == "nightly-dream"
    assert any("Build the dreaming cron runner" in item for item in artifact["open_loops"] + artifact["do_not_promote_yet"] + artifact["tomorrow_cue"])


def test_dreaming_filters_low_signal_support_chatter(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(hermes_home=tmp_path / "hermes-home", memory_store=store)

    artifact = engine.run(
        [
            {"role": "user", "content": "i dont know which folder to load in xcode to open the app"},
            {"role": "user", "content": "how do i do step 1"},
            {"role": "user", "content": "Fix login auth for the certain email and tune dream quality."},
            {"role": "assistant", "content": "I will add a bounded dream review note."},
            {"role": "assistant", "content": "Open Terminal, then paste this exact command."},
        ],
        session_id="s3",
        platform="telegram",
        workspace="/tmp/project",
    )

    flattened = "\n".join(artifact["open_loops"] + artifact["tomorrow_cue"] + artifact["do_not_promote_yet"])
    assert "Fix login auth" in flattened
    assert "how do i do step 1" not in flattened
    assert "i dont know which folder" not in flattened
    assert "Open Terminal" not in flattened


class NoisySessionDB:
    def list_sessions_rich(self, limit=5):
        return [
            {"id": "good", "source": "telegram"},
            {"id": "noisy", "source": "cli"},
        ]

    def get_messages(self, session_id):
        mapping = {
            "good": [
                {"role": "user", "content": "Fix login auth for the certain email"},
                {"role": "assistant", "content": "Investigate the auth failure and patch the login flow."},
            ],
            "noisy": [
                {"role": "user", "content": "how do i do step 1"},
                {"role": "assistant", "content": "Open Terminal, then paste this exact command."},
            ],
        }
        return mapping[session_id]


def test_dreaming_nightly_prefers_real_work_over_noisy_help_text(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(
        hermes_home=tmp_path / "hermes-home",
        memory_store=store,
        session_db=NoisySessionDB(),
    )

    artifact = engine.run_nightly(session_limit=5, per_session_limit=4)
    flattened = "\n".join(artifact["open_loops"] + artifact["tomorrow_cue"] + artifact["do_not_promote_yet"])
    assert "Fix login auth for the certain email" in flattened
    assert "how do i do step 1" not in flattened
    assert "Open Terminal" not in flattened


def test_dreaming_prefers_active_work_over_static_preferences(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    engine = DreamingEngine(hermes_home=tmp_path / "hermes-home", memory_store=store)

    artifact = engine.run(
        [
            {"role": "user", "content": "Fix login auth for the certain email and improve dream ranking."},
            {"role": "assistant", "content": "Investigate auth failure and tighten nightly dream selection."},
        ],
        session_id="s4",
        platform="cli",
        workspace="/repo",
    )

    assert artifact["open_loops"]
    assert "Fix login auth" in artifact["tomorrow_cue"][0]
    assert not artifact["tomorrow_cue"][0].casefold().startswith("user prefers")
