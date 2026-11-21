"""Contract suite for the admission media verbs: held paths, transcript mentions, settled media and the durable retirement candidates."""

from __future__ import annotations

import hashlib

import pytest

from persistence import PermanentStoreError
from persistence.authority.runtime import Delete
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from session_rules.media import HOSTED_DOCUMENT_LINE

SESSION = "sess-media"
ROOT = "/contract-media/native-inputs"
DIGEST = "a" * 64


def _ref(name: str, digest: str = DIGEST) -> dict[str, str]:
    return {"path": f"{ROOT}/{digest}/{name}", "sha256": digest, "size": 1}


def _seed(store) -> int:
    store.sessions.create(NewSession(session_id=SESSION, source="contract"))
    return store.runtime.begin_epoch(instance_id="instance-media")


def _admit(store, epoch, request_id, payload):
    return store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id=request_id, payload=payload)[0]


def _settle_next(store, epoch):
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.admissions.settle(epoch=epoch, admission_id=claimed.admission_id, generation=claimed.generation, outcome="completed")


def _unheld(store, references, **kwargs):
    kwargs.setdefault("after_id", 0)
    kwargs.setdefault("retain_history", False)
    return store.admissions.unheld_media(references, media_root=ROOT, **kwargs)


def test_held_paths_follow_status(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    attachment, native, api = _ref("a.png"), _ref("n.png"), _ref("api.png")
    _admit(store, epoch, "r1", {"attachments_v1": {"media": [attachment]}, "api_turn_v1": {"media": [api]}})
    _admit(store, epoch, "r2", {"native_text_v1": {"media": [native]}})
    # CONTRACT: live rows hold their attachment and native media; API media is held in every status.
    unheld = _unheld(store, [attachment, native, api, _ref("free.png")])
    assert unheld.held == frozenset({attachment["path"], native["path"], api["path"]})
    assert [c["path"] for c in unheld.candidates] == [_ref("free.png")["path"]]
    _settle_next(store, epoch)
    unheld = _unheld(store, [attachment, native, api])
    assert unheld.held == frozenset({native["path"], api["path"]})
    assert unheld.candidates == (attachment,)


def test_live_hosted_prompts_hold_their_documents_under_the_media_root(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    document = _ref("report.pdf", "b" * 64)
    outside = {"path": f"/elsewhere/{'c' * 64}/x.pdf", "sha256": "c" * 64}
    text = f"see\n[Shared attachment] file: {document['path']}\n\n[Shared attachment] file: {outside['path']}\n"
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="hosted:room:1", payload={"text": text})
    assert _unheld(store, [document, outside]).held == frozenset({document["path"]})
    # CONTRACT: the same text on a non-hosted request id holds nothing.
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="plain", payload={"text": text})
    _settle_next(store, epoch)
    assert _unheld(store, [document]).held == frozenset()


def test_history_mentions_and_the_watermark(backend, scope_a):
    store = backend.open(scope_a)
    _seed(store)
    image = _ref("pic.png")
    assert store.admissions.history_mentions([image]) == (0, frozenset())
    first = store.messages.append(SESSION, NewMessage(role="user", content=f"look at {DIGEST}/pic.png"))
    second = store.messages.append(SESSION, NewMessage(role="assistant", content=[{"type": "text", "text": "x"}]))
    watermark, mentioned = store.admissions.history_mentions([image, _ref("other.png")])
    assert watermark == int(second) and mentioned == frozenset({image["path"]})
    # CONTRACT: with retain_history, only rows above after_id are re-checked inside the write.
    assert _unheld(store, [image], retain_history=True, after_id=int(first)).held == frozenset()
    assert _unheld(store, [image], retain_history=True, after_id=0).held == frozenset({image["path"]})


def test_settled_media_lists_terminal_rows_with_media(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    attachment, native = _ref("a.png"), _ref("n.png")
    _admit(store, epoch, "r1", {"attachments_v1": {"media": [attachment]}, "native_text_v1": {"media": [native]}})
    _admit(store, epoch, "r2", {"text": "no media"})
    assert store.admissions.settled_media() == []
    _settle_next(store, epoch)
    _settle_next(store, epoch)
    assert store.admissions.settled_media() == [[attachment, native]]


def test_retired_candidates_record_first_write_and_drop(backend, scope_a):
    store = backend.open(scope_a)
    _seed(store)
    image = _ref("pic.png")
    store.admissions.record_retired_media([image, dict(image, size=2)])
    key = "gateway.media.retired.v1." + hashlib.sha256(image["path"].encode()).hexdigest()
    assert store.admissions.retired_candidates("media") == [(key, image)]
    assert store.admissions.retired_candidates("spillover") == []
    store.admissions.drop_retired_candidates([key, "absent"])
    assert store.admissions.retired_candidates("media") == []


def test_corrupt_media_reference_is_a_permanent_store_error(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    _admit(store, epoch, "r1", {"attachments_v1": {"media": [{"sha256": DIGEST}]}})
    with pytest.raises(PermanentStoreError):
        _unheld(store, [_ref("x.png")])


def test_a_delete_retires_the_hosted_documents_under_the_given_media_root(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    other_root, other_digest = "/other-media/native-inputs", "b" * 64
    inside, outside = f"{ROOT}/{DIGEST}/doc.pdf", f"{other_root}/{other_digest}/doc.pdf"
    text = "q" + HOSTED_DOCUMENT_LINE + inside + "\n" + HOSTED_DOCUMENT_LINE + outside + "\n"
    store.sessions.create(NewSession(session_id="sess-media-2", source="contract"))
    for session_id in (SESSION, "sess-media-2"):
        record, _ = store.admissions.admit(epoch=epoch, principal_id="p", session_id=session_id, request_id="hosted:room:1", payload={"text": text})
        store.admissions.cancel(epoch=epoch, admission_id=record.admission_id)
    # CONTRACT: a retired admission's hosted documents become candidates only under the media root the caller passes.
    store.sessions.delete(SESSION, media_root=ROOT)
    assert [value for _key, value in store.admissions.retired_candidates("media")] == [{"path": inside, "sha256": DIGEST}]
    store.sessions.delete("sess-media-2", media_root=other_root)
    assert sorted(value["path"] for _key, value in store.admissions.retired_candidates("media")) == sorted([inside, outside])


def test_a_runtime_delete_retires_the_hosted_documents_under_its_media_root(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    other_root, other_digest = "/other-media/native-inputs", "b" * 64
    inside, outside = f"{ROOT}/{DIGEST}/doc.pdf", f"{other_root}/{other_digest}/doc.pdf"
    text = "q" + HOSTED_DOCUMENT_LINE + inside + "\n" + HOSTED_DOCUMENT_LINE + outside + "\n"
    store.sessions.create(NewSession(session_id="sess-media-2", source="contract"))
    for session_id in (SESSION, "sess-media-2"):
        record, _ = store.admissions.admit(epoch=epoch, principal_id="p", session_id=session_id, request_id="hosted:room:1", payload={"text": text})
        store.admissions.cancel(epoch=epoch, admission_id=record.admission_id)

    def delete(session_id, media_root, revision=None, generation=None):
        row = store.sessions.get(session_id)
        return store.runtime.mutate(epoch, principal_id="p", session_id=session_id, request_id="req-delete",
                                    expected_revision=row.runtime_revision if row else revision,
                                    expected_generation=row.runtime_generation if row else generation, command=Delete(), media_root=media_root, owner=None)

    # CONTRACT: the runtime delete retires a retired admission's hosted documents only under the media root it is given.
    row = store.sessions.get(SESSION)
    first = delete(SESSION, ROOT)
    assert [value for _key, value in store.admissions.retired_candidates("media")] == [{"path": inside, "sha256": DIGEST}]
    # CONTRACT: the media root is in no payload, so the exact retry with another root replays the same receipt and retires nothing more.
    replay = delete(SESSION, other_root, row.runtime_revision, row.runtime_generation)
    assert replay.replayed and replay.result == first.result
    assert [value for _key, value in store.admissions.retired_candidates("media")] == [{"path": inside, "sha256": DIGEST}]
    delete("sess-media-2", other_root)
    assert sorted(value["path"] for _key, value in store.admissions.retired_candidates("media")) == sorted([inside, outside])
