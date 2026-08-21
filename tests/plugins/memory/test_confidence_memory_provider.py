from datetime import timedelta
import stat

from plugins.memory import load_memory_provider
from plugins.memory.confidence import ConfidenceMemoryProvider
from plugins.memory.confidence.schemas import Confidence, Layer, MemorySource, SourceKind, Status, now_utc
from plugins.memory.confidence.store import ConfidenceMemoryStore


def source(kind=SourceKind.USER_STATED, excerpt="user said it"):
    return MemorySource(kind=kind, observed_at=now_utc(), excerpt=excerpt)


def test_provider_is_discoverable():
    provider = load_memory_provider("confidence")
    assert isinstance(provider, ConfidenceMemoryProvider)
    assert provider.is_available()


def test_provider_default_db_uses_injected_profile_home(tmp_path):
    profile_home = tmp_path / "profiles" / "work"
    provider = ConfidenceMemoryProvider(config={})

    provider.initialize("session-1", hermes_home=str(profile_home))

    assert provider._store is not None
    assert provider._store.db_path == profile_home / "confidence_memory.db"
    assert provider._store.db_path.exists()
    provider.shutdown()


def test_store_database_is_owner_only(tmp_path):
    db_path = tmp_path / "confidence.db"

    store = ConfidenceMemoryStore(db_path)

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    store.close()


def test_store_confirmed_profile_injection_and_tentative_exclusion(tmp_path):
    store = ConfidenceMemoryStore(tmp_path / "confidence.db")
    confirmed = store.add(
        statement="User prefers concise executive summaries.",
        layer=Layer.PROFILE,
        confidence=Confidence.CONFIRMED,
        sources=[source()],
    )
    tentative = store.add(
        statement="User may prepare weekly materials on Monday mornings.",
        layer=Layer.PROFILE,
        confidence=Confidence.TENTATIVE,
        sources=[source(SourceKind.ACTIVITY_PATTERN, "single observation")],
        scope="injection",
    )

    assert store.get(tentative).scope == "retrieval_only"
    selected = store.select_for_injection(query="")
    assert [item.id for item in selected] == [confirmed]


def test_store_ttl_stale_expired_and_user_statement_supersedes(tmp_path):
    store = ConfidenceMemoryStore(tmp_path / "confidence.db")
    created = now_utc() - timedelta(days=8)
    old = store.add(
        statement="User prefers long-form reports.",
        layer=Layer.PROFILE,
        confidence=Confidence.INFERRED,
        sources=[source(SourceKind.ACTIVITY_PATTERN, "three prior approvals")],
        created_at=created,
        ttl="14d",
    )

    store.refresh_statuses(as_of=created + timedelta(days=8))
    assert store.get(old).status == Status.STALE.value

    new = store.resolve_user_stated_conflict(
        old,
        "User prefers short executive summaries.",
        source(SourceKind.USER_STATED, "短くして"),
    )

    assert store.get(old).status == Status.SUPERSEDED.value
    assert store.get(old).superseded_by == new
    assert store.get(new).confidence == Confidence.CONFIRMED.value


def test_provider_tool_add_list_confirm_delete(tmp_path):
    provider = ConfidenceMemoryProvider(config={"db_path": str(tmp_path / "confidence.db")})
    provider.initialize("session-1")

    add_result = provider.handle_tool_call("confidence_memory", {
        "action": "add",
        "statement": "User likes verified citations.",
        "layer": "profile",
        "confidence": "tentative",
        "source_excerpt": "single observation",
        "source_kind": "activity_pattern",
    })
    assert '"success": true' in add_result
    item_id = provider._store.list_items(include_inactive=True)[0].id
    assert provider._store.get(item_id).scope == "retrieval_only"

    provider.handle_tool_call("confidence_memory", {
        "action": "confirm",
        "id": item_id,
        "source_excerpt": "user confirmed",
    })
    assert provider._store.get(item_id).confidence == "confirmed"
    assert provider._store.get(item_id).scope == "injection"

    listed = provider.handle_tool_call("confidence_memory", {"action": "list"})
    assert "User likes verified citations" in listed

    provider.handle_tool_call("confidence_memory", {"action": "delete", "id": item_id})
    assert provider._store.list_items(include_inactive=True) == []
    provider.shutdown()


def test_provider_prefetch_formats_confirmed_and_inferred_but_not_tentative(tmp_path):
    provider = ConfidenceMemoryProvider(config={"db_path": str(tmp_path / "confidence.db")})
    provider.initialize("session-1")
    provider._store.add(
        statement="Confirmed profile fact.",
        layer=Layer.PROFILE,
        confidence=Confidence.CONFIRMED,
        sources=[source()],
    )
    provider._store.add(
        statement="Tentative profile hint.",
        layer=Layer.PROFILE,
        confidence=Confidence.TENTATIVE,
        sources=[source(SourceKind.ACTIVITY_PATTERN)],
    )

    context = provider.prefetch("anything")
    assert "[confirmed]" in context
    assert "Confirmed profile fact" in context
    assert "Tentative profile hint" not in context
    provider.shutdown()
