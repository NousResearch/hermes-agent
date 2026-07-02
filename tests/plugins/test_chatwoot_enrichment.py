"""Unit tests for CRWD → Chatwoot/Honcho contact enrichment.

Pure mapping + matcher + idempotency logic, plus the orchestrator wired to a
fake adapter and monkeypatched Mongo/Honcho readers. No live Mongo/Chatwoot/
Honcho is touched.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from plugins.platforms.chatwoot import enrichment as en


# --- sample data (mirrors real crwd_staging docs) ---------------------------

SAMPLE_USER = {
    "_id": "69e273fb1d163ce2fd86754c",
    "first_name": "Stephanie",
    "last_name": "Reyes",
    "full_name": "Stephanie Reyes",
    "email": "lizardreyes13@gmail.com",
    "phone": "+13018358984",
    "city": "Frederick",
    "state": "Maryland",
    "country": "US",
    "bio": "",
    "gender": "female",
    "dob": "1998-12-20",
    "status": "Active",
    "profile_pic": "profile_pic_1776448509925_ABC.jpeg",
    "avatar": "",
    "insta_url": "1.estefania",
    "twitter_url": "",
    "tiktok_url": "@creolepride",
}


# --- social / avatar / interest helpers -------------------------------------

def test_social_url_expands_handles():
    assert en._social_url("1.estefania", "instagram") == "https://instagram.com/1.estefania"
    assert en._social_url("@creolepride", "tiktok") == "https://tiktok.com/@creolepride"
    assert en._social_url("someone", "twitter") == "https://x.com/someone"
    assert en._social_url("", "instagram") is None


def test_social_url_from_full_url_keeps_last_segment():
    assert en._social_url("https://instagram.com/foo", "instagram") == "https://instagram.com/foo"


def test_avatar_url_builds_from_base():
    assert (
        en._avatar_url("pic.jpeg", "https://cdn.crwd.app/uploads")
        == "https://cdn.crwd.app/uploads/pic.jpeg"
    )
    # No base → cannot build → None.
    assert en._avatar_url("pic.jpeg", "") is None
    # Already absolute → passthrough regardless of base.
    assert en._avatar_url("https://x/y.jpg", "") == "https://x/y.jpg"
    assert en._avatar_url("", "https://cdn") is None


def test_clean_interest_title_strips_emoji():
    assert en._clean_interest_title("🎵 Music") == "Music"
    assert en._clean_interest_title("🎬 Movies") == "Movies"
    assert en._clean_interest_title("Sports") == "Sports"


def test_build_interest_labels_slugifies_and_dedupes():
    labels = en.build_interest_labels(["🎵 Music", "🎬 Movies", "Music", "Live Music"])
    assert labels == ["music", "movies", "live-music"]


# --- contact field mapping --------------------------------------------------

def test_build_contact_fields_maps_expected():
    fields = en.build_contact_fields(
        SAMPLE_USER, "https://cdn.crwd.app/uploads", synced_at="2026-07-02T00:00:00+00:00"
    )
    assert fields["name"] == "Stephanie Reyes"
    assert fields["email"] == "lizardreyes13@gmail.com"
    assert fields["phone_number"] == "+13018358984"
    assert fields["avatar_url"] == "https://cdn.crwd.app/uploads/profile_pic_1776448509925_ABC.jpeg"
    assert fields["additional_attributes"] == {
        "city": "Frederick", "country": "US", "state": "Maryland",
    }
    custom = fields["custom_attributes"]
    assert custom["joincrwd_user_id"] == "69e273fb1d163ce2fd86754c"
    assert custom["crwd_synced_at"] == "2026-07-02T00:00:00+00:00"
    assert custom["crwd_instagram"] == "https://instagram.com/1.estefania"
    assert custom["crwd_tiktok"] == "https://tiktok.com/@creolepride"
    # Empty bio / twitter are skipped.
    assert "crwd_bio" not in custom
    assert "crwd_twitter" not in custom


def test_build_contact_fields_falls_back_to_first_last_name():
    user = dict(SAMPLE_USER, full_name="")
    fields = en.build_contact_fields(user, "")
    assert fields["name"] == "Stephanie Reyes"
    # No base URL → no avatar.
    assert "avatar_url" not in fields


# --- honcho card / conclusions ----------------------------------------------

def test_build_peer_card_and_conclusions():
    interests = ["Music", "Movies"]
    answers = [
        ("What is your approximate annual household income?", "$50K–$75K"),
        ("Highest education", "Associate degree"),
    ]
    card = en.build_peer_card(SAMPLE_USER, interests, answers)
    assert "Name: Stephanie Reyes" in card
    assert "Location: Frederick, Maryland, US" in card
    assert "Interests: Music, Movies" in card
    assert "What is your approximate annual household income?: $50K–$75K" in card

    conclusions = en.build_conclusions(answers)
    assert conclusions == [
        "What is your approximate annual household income? — $50K–$75K",
        "Highest education — Associate degree",
    ]


# --- matcher query ordering -------------------------------------------------

def test_match_queries_email_first_then_phone():
    q = en._match_queries("A@B.com", "+1555")
    assert len(q) == 2
    # Email is a case-insensitive regex; phone is exact.
    assert q[0]["email"].pattern == "^A@B\\.com$"
    assert q[1] == {"phone": "+1555"}
    # Only phone given → single query.
    assert en._match_queries("", "+1555") == [{"phone": "+1555"}]
    assert en._match_queries("", "") == []


# --- idempotency ------------------------------------------------------------

def test_synced_at_is_fresh(monkeypatch):
    monkeypatch.setenv("CRWD_ENRICH_TTL_DAYS", "30")
    recent = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    assert en._synced_at_is_fresh({"custom_attributes": {"crwd_synced_at": recent}})
    assert not en._synced_at_is_fresh({"custom_attributes": {"crwd_synced_at": old}})
    assert not en._synced_at_is_fresh({"custom_attributes": {}})
    assert not en._synced_at_is_fresh({})


def test_recent_syncs_cache(monkeypatch):
    monkeypatch.setenv("CRWD_ENRICH_TTL_DAYS", "30")
    en._recent_syncs.clear()
    assert not en._recently_synced("c1")
    en._mark_synced("c1")
    assert en._recently_synced("c1")


# --- orchestrator -----------------------------------------------------------

class FakeAdapter:
    def __init__(self, contact=None, avatar_ok=True):
        self._contact = contact
        self._avatar_ok = avatar_ok
        self.updated = None
        self.labels = None

    async def get_contact(self, account_id, contact_id):
        return self._contact

    async def update_contact(self, account_id, contact_id, fields):
        self.updated = fields
        return True

    async def add_contact_labels(self, account_id, contact_id, labels):
        self.labels = labels
        return True

    async def url_is_image(self, url):
        return self._avatar_ok


def _event(email="lizardreyes13@gmail.com", phone=None):
    payload = {
        "sender": {"id": "77", "email": email, "phone_number": phone},
        "account": {"id": "1"},
        "conversation": {"id": "9"},
    }
    return SimpleNamespace(raw_message=payload, source=SimpleNamespace(chat_id="1:9"))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x")  # gate on
    monkeypatch.setenv("CRWD_ENRICH_ENABLED", "true")
    monkeypatch.setenv("CRWD_ENRICH_TTL_DAYS", "30")
    monkeypatch.setattr(en, "_write_honcho", lambda *a, **k: None)
    en._recent_syncs.clear()
    yield


def test_enrich_happy_path(monkeypatch):
    monkeypatch.setenv("CRWD_ASSET_BASE_URL", "https://cdn.crwd.app/uploads")
    monkeypatch.setattr(en, "fetch_user", lambda email, phone: dict(SAMPLE_USER))
    monkeypatch.setattr(en, "fetch_interests", lambda uid: ["Music", "Movies"])
    monkeypatch.setattr(en, "fetch_answers", lambda uid, cap=25: [("demographics", "$50K–$75K")])
    adapter = FakeAdapter(contact=None)

    asyncio.run(en.enrich(adapter, _event()))

    assert adapter.updated["email"] == "lizardreyes13@gmail.com"
    assert adapter.updated["avatar_url"].endswith("profile_pic_1776448509925_ABC.jpeg")
    assert adapter.labels == ["music", "movies"]
    assert en._recently_synced("77")  # marked synced


def test_enrich_drops_avatar_when_not_an_image(monkeypatch):
    monkeypatch.setattr(en, "fetch_user", lambda email, phone: dict(SAMPLE_USER))
    monkeypatch.setattr(en, "fetch_interests", lambda uid: [])
    monkeypatch.setattr(en, "fetch_answers", lambda uid, cap=25: [])
    monkeypatch.setenv("CRWD_ASSET_BASE_URL", "https://cdn.crwd.app/uploads")
    adapter = FakeAdapter(contact=None, avatar_ok=False)  # URL serves HTML, not image

    asyncio.run(en.enrich(adapter, _event()))

    assert "avatar_url" not in adapter.updated  # broken avatar stripped
    assert adapter.updated["email"] == "lizardreyes13@gmail.com"  # rest still synced


def test_enrich_no_match_negative_caches(monkeypatch):
    calls = {"n": 0}

    def _fetch(email, phone):
        calls["n"] += 1
        return None

    monkeypatch.setattr(en, "fetch_user", _fetch)
    adapter = FakeAdapter(contact=None)

    asyncio.run(en.enrich(adapter, _event()))
    asyncio.run(en.enrich(adapter, _event()))  # second message

    assert adapter.updated is None
    assert calls["n"] == 1  # negative-cached after first miss


def test_enrich_skips_when_recently_synced(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(en, "fetch_user", lambda e, p: called.__setitem__("n", called["n"] + 1) or dict(SAMPLE_USER))
    monkeypatch.setattr(en, "fetch_interests", lambda uid: [])
    monkeypatch.setattr(en, "fetch_answers", lambda uid, cap=25: [])
    adapter = FakeAdapter(contact=None)

    asyncio.run(en.enrich(adapter, _event()))
    asyncio.run(en.enrich(adapter, _event()))
    assert called["n"] == 1  # second call short-circuited by in-proc cache


def test_enrich_skips_when_contact_marked_fresh(monkeypatch):
    fresh = {"custom_attributes": {"crwd_synced_at": datetime.now(timezone.utc).isoformat()}}
    monkeypatch.setattr(en, "fetch_user", lambda e, p: pytest.fail("should not query Mongo"))
    adapter = FakeAdapter(contact=fresh)
    asyncio.run(en.enrich(adapter, _event()))
    assert adapter.updated is None


def test_enrich_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("CRWD_MONGO_URI", raising=False)
    adapter = FakeAdapter(contact=None)
    asyncio.run(en.enrich(adapter, _event()))
    assert adapter.updated is None
