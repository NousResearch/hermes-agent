from agent.beta.chief_profile import ChiefFact, ChiefProfileStore


def test_profile_persists_durable_fact(tmp_path):
    store = ChiefProfileStore(tmp_path)
    updated = store.add_fact("user-1", ChiefFact(type="preference", value="Prefers concise answers"))
    loaded = store.load("user-1")
    assert loaded.revision == updated.revision
    assert loaded.facts[0].value == "Prefers concise answers"
    assert "do not grant technical permissions" in loaded.prompt_block()


def test_profile_rejects_secret_like_content(tmp_path):
    store = ChiefProfileStore(tmp_path)
    try:
        store.add_fact("user-1", ChiefFact(type="preference", value="api_key is abc"))
    except ValueError as exc:
        assert "credentials" in str(exc)
    else:
        raise AssertionError("secret-like profile fact should be rejected")


def test_profile_rejects_temporary_fact_type(tmp_path):
    store = ChiefProfileStore(tmp_path)
    try:
        store.add_fact("user-1", ChiefFact(type="task_progress", value="half complete"))
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("temporary task progress should be rejected")
