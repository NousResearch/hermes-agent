"""Property: Telegram photo bursts and albums never move an event out of the topic it was routed to.

Invariant under test: every event in one batch has the same resolved session key. Seeded sequences
of questions, photo bursts and albums (one album may span two questions, as when a question is typed
mid-album), with topic bindings changing between them, run through the real ``_handle_text_message``
and ``_handle_media_message`` handlers, real topic recovery over a real SessionDB, and the production
0.8 s burst / 1.5 s album windows. For every sequence:

- each question and each image lands in the topic recovery gives it on arrival (what main gives it
  when nothing rebinds before dispatch); one sent before the user had any topic binding lands in the
  first topic bound after it, together with the rest of its burst or album;
- no image is dropped or duplicated, no question is delivered twice;
- a question and the photos sent right after it are one turn when no binding changes between them.
"""

import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_telegram_documents import _make_file_obj, _make_message, _make_photo
from tests.plugins.platforms.telegram.test_text_then_photo_one_turn import _text, _topic_mode_rig, _update
from tests.waiting import async_wait_until

TOPICS = ("10", "20", "30")


def _step(bind=None, question=True, bind_before_photo=None, thread=None, photos=2, album=None, bind_mid=None) -> dict:
    return {"bind": bind, "question": question, "bind_before_photo": bind_before_photo,
            "thread": thread, "photos": photos, "album": album, "bind_mid": bind_mid}


def _grid() -> list[list[dict]]:
    """Every two-step shape: a question with one photo or album item, then a second question with two,
    as a new burst, a new album or the same album, with or without topic 20 binding in between."""
    return [[_step(photos=1, album=first), _step(bind=bind, album=second)]
            for first in (None, "A") for second in (None, "B", "A") for bind in (None, "20")
            if not (first is None and second == "A")]


def _random(seed: int) -> list[dict]:
    """Longer sequences: question-less photos, explicit topics, a rebind between question and photo."""
    rng = random.Random(seed)
    steps, album = [], None
    for index in range(3):
        kind = rng.choice(("burst", "album", "continue"))
        album = None if kind == "burst" else (album if kind == "continue" and album else f"alb-{index}")
        steps.append(_step(bind=rng.choice(TOPICS) if index and rng.random() < 0.6 else None,
                           question=rng.random() < 0.85,
                           bind_before_photo=rng.choice(TOPICS) if rng.random() < 0.2 else None,
                           thread=rng.choice(TOPICS) if rng.random() < 0.15 else None,
                           photos=rng.choice((1, 2, 2)), album=album))
    return steps


def _first_binding() -> list[list[dict]]:
    """The user's first topic binding lands mid-burst or mid-album (after its first photo), with or
    without a question before it, optionally followed by a pair in a newer topic (same lane)."""
    return [[{**_step(question=question, album=album, bind_mid="10"), "unbound_start": True}, *tail]
            for question in (False, True) for album in (None, "A")
            for tail in ([], [_step(bind="20", album=album)])]


CASES = _grid() + [_random(seed) for seed in range(8)] + _first_binding()


def _photo(name: str, release: asyncio.Event, thread, album):
    file_obj = _make_file_obj(name.encode())

    async def download():
        await release.wait()
        return bytearray(name.encode())
    file_obj.download_as_bytearray = AsyncMock(side_effect=download)
    msg = _make_message(photo=[_make_photo(file_obj)], media_group_id=album)
    msg.message_thread_id = int(thread) if thread else None
    return msg


async def _run(tmp_path, steps):
    adapter, db, bind, routed = _topic_mode_rig(tmp_path)
    adapter._media_batch_delay_seconds = 0.8  # production burst window
    newest, unbound = None, []
    expect, pairs = {}, []

    def rebind(topic):
        nonlocal newest
        if topic:
            bind(topic)
            newest = topic
            for key in unbound:  # sent before any binding: joins the first topic bound after it
                expect[key] = topic
            unbound.clear()

    def expect_at(key, thread):
        expect[key] = thread or newest
        if expect[key] is None:
            unbound.append(key)

    if not steps[0].get("unbound_start"):
        rebind("10")

    try:
        with patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async",
                   new=AsyncMock(side_effect=lambda data, **_kw: f"/cache/{data.decode()}.jpg")):
            for index, step in enumerate(steps):
                rebind(step["bind"])
                question = f"q{index}" if step["question"] else None
                if question:
                    msg = _text(question)
                    msg.message_thread_id = int(step["thread"]) if step["thread"] else None
                    await adapter._handle_text_message(_update(msg), MagicMock())
                    expect_at(question, step["thread"])
                rebind(step["bind_before_photo"])
                names = [f"p{index}-{n}" for n in range(step["photos"])]
                for n, name in enumerate(names):
                    release = asyncio.Event()
                    task = asyncio.create_task(adapter._handle_media_message(
                        _update(_photo(name, release, step["thread"], step["album"])), MagicMock()))
                    if n == 0:  # the first download outlasts the 0.18 s text delay: the claim path
                        await asyncio.sleep(0.3)  # health: allow HX013 -- a download outlasting the text delay IS the race
                    release.set()
                    await task
                    expect_at(f"/cache/{name}.jpg", step["thread"])
                    if n == 0:
                        rebind(step["bind_mid"])
                if question and not step["bind_before_photo"]:
                    pairs.append((question, [f"/cache/{name}.jpg" for name in names]))
            images = [key for key in expect if key.startswith("/cache/")]
            await async_wait_until(lambda: sum(len(media) for *_, media in routed) >= len(images), timeout=15)
            await async_wait_until(lambda: not (adapter._pending_photo_batches or adapter._media_group_events
                                                or adapter._pending_text_batches), timeout=15)
    finally:
        db.close()
    return routed, expect, pairs


@pytest.mark.asyncio
@pytest.mark.parametrize("steps", CASES, ids=[f"case{n}" for n in range(len(CASES))])
async def test_every_event_lands_in_its_own_topic_and_question_photos_stay_one_turn(tmp_path, steps):
    routed, expect, pairs = await _run(tmp_path, steps)
    landed = [(image, thread) for _text_, thread, media in routed for image in media]
    landed += [(q, thread) for text, thread, _m in routed for q in expect if not q.startswith("/") and q in (text or "").split()]
    assert sorted(landed) == sorted(expect.items()), (steps, routed)  # right topic, none dropped or duplicated
    for question, images in pairs:
        turn = next(media for text, _t, media in routed if question in (text or "").split())
        assert set(images) <= set(turn), (question, steps, routed)
    if steps[0].get("unbound_start"):  # the burst or album straddling the first binding is one turn
        first = {f"/cache/p0-{n}.jpg" for n in range(steps[0]["photos"])}
        assert any(first <= set(media) for *_, media in routed), (steps, routed)
