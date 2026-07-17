"""Tests for the pr_review_buttons plugin.

The plugin stages verbatim PR reviews, posts them to Slack as Block Kit
messages with Approve / Request-changes / Comment buttons, and posts the
*exact* staged review to GitHub when the operator clicks a button.

These tests cover the pure, deterministic pieces (store, blocks, github
wrapper decisions, action decision logic). The async Slack wiring is thin and
exercised through injected fakes so no live Slack/GitHub calls are made.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from plugins.pr_review_buttons import actions, blocks, github, slackio, store


# ---------------------------------------------------------------------------
# store.py — pending-review persistence
# ---------------------------------------------------------------------------


def _entry(**over):
    base = {
        "repo": "Strike48/matrix",
        "number": 42,
        "head_sha": "abc123",
        "title": "Add foo",
        "url": "https://github.com/Strike48/matrix/pull/42",
        "verdict": "Needs work",
        "body": "## Hermes Review\n\nUse parameterized queries.",
    }
    base.update(over)
    return base


def test_stage_and_get_roundtrip(tmp_path: Path):
    p = tmp_path / "pending.json"
    e = _entry()
    store.stage(e, path=p)

    got = store.get("Strike48/matrix#42", path=p)
    assert got is not None
    assert got["body"] == e["body"]
    assert got["head_sha"] == "abc123"


def test_stage_overwrites_same_key(tmp_path: Path):
    p = tmp_path / "pending.json"
    store.stage(_entry(body="v1", head_sha="old"), path=p)
    store.stage(_entry(body="v2", head_sha="new"), path=p)

    got = store.get("Strike48/matrix#42", path=p)
    assert got["body"] == "v2"
    assert got["head_sha"] == "new"
    assert len(store.load(p)) == 1  # not duplicated


def test_pop_is_atomic_and_removes(tmp_path: Path):
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)

    first = store.pop("Strike48/matrix#42", path=p)
    second = store.pop("Strike48/matrix#42", path=p)
    assert first is not None
    assert second is None  # second caller (double-click) gets nothing
    assert store.get("Strike48/matrix#42", path=p) is None


def test_key_for_builds_owner_repo_number():
    assert store.key_for("Strike48/matrix", 42) == "Strike48/matrix#42"


def test_unpublished_returns_only_entries_without_message_ts(tmp_path: Path):
    p = tmp_path / "pending.json"
    store.stage(_entry(number=1, url="u1"), path=p)
    store.stage(_entry(number=2, url="u2"), path=p)
    store.mark_published("Strike48/matrix#1", message_ts="1700.1", path=p)

    pending = store.unpublished(path=p)
    keys = {e["_key"] for e in pending}
    assert keys == {"Strike48/matrix#2"}


def test_load_missing_file_returns_empty(tmp_path: Path):
    assert store.load(tmp_path / "nope.json") == {}


def test_stage_strips_unknown_fields(tmp_path: Path):
    p = tmp_path / "pending.json"
    store.stage(dict(_entry(), evil_extra="drop-me", another="x"), path=p)
    got = store.get("Strike48/matrix#42", path=p)
    assert "evil_extra" not in got
    assert "another" not in got
    assert got["body"]  # real fields survive


def test_stage_write_is_atomic_no_partial(tmp_path: Path):
    # A concurrent reader must never see a truncated file: the store writes to a
    # temp file and renames. After stage, the file must be valid JSON.
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)
    parsed = json.loads(p.read_text())
    assert "Strike48/matrix#42" in parsed


# ---------------------------------------------------------------------------
# blocks.py — Block Kit rendering
# ---------------------------------------------------------------------------


def test_digest_blocks_emit_three_buttons_per_pr():
    e = dict(_entry(), _key="Strike48/matrix#42")
    bk = blocks.build_digest_blocks([e])

    action_blocks = [b for b in bk if b.get("type") == "actions"]
    assert len(action_blocks) == 1
    ids = {el["action_id"] for el in action_blocks[0]["elements"]}
    assert ids == {
        "prreview_approve",
        "prreview_request_changes",
        "prreview_comment",
    }


def test_digest_buttons_carry_key_as_value():
    e = dict(_entry(), _key="Strike48/matrix#42")
    bk = blocks.build_digest_blocks([e])
    action_block = next(b for b in bk if b.get("type") == "actions")
    for el in action_block["elements"]:
        assert el["value"] == "Strike48/matrix#42"


def test_digest_button_value_within_slack_limit():
    # Slack caps a button value at 2000 chars. The key must always fit even for
    # long repo names — this is the whole reason the body is stored, not encoded.
    e = dict(_entry(repo="Some-Really-Long-Org-Name/an-extremely-long-repository-name"),
             _key="Some-Really-Long-Org-Name/an-extremely-long-repository-name#999999")
    bk = blocks.build_digest_blocks([e])
    action_block = next(b for b in bk if b.get("type") == "actions")
    for el in action_block["elements"]:
        assert len(el["value"]) <= 2000


def test_digest_shows_verbatim_body_so_wysiwyg_holds():
    # The operator must see exactly what will be posted. The review body appears
    # verbatim in the message (across section blocks).
    body = "## Hermes Review\n\nUnique-Marker-XYZ parameterize this query."
    e = dict(_entry(body=body), _key="Strike48/matrix#42")
    bk = blocks.build_digest_blocks([e])
    text = json.dumps(bk)
    assert "Unique-Marker-XYZ" in text


def test_digest_splits_body_over_section_limit():
    big = "x" * 7000  # > 3000-char section cap; must split, never drop
    e = dict(_entry(body=big), _key="Strike48/matrix#42")
    bk = blocks.build_digest_blocks([e])
    for b in bk:
        if b.get("type") == "section":
            assert len(b["text"]["text"]) <= 3000
    # No content lost: concatenated section text still contains the body length.
    total = sum(
        len(b["text"]["text"]) for b in bk if b.get("type") == "section"
    )
    assert total >= 7000


def test_posted_blocks_have_no_buttons_and_show_decision():
    bk = blocks.build_posted_blocks(
        _entry(), event="request-changes", user="jtomek", detail="posted"
    )
    assert all(b.get("type") != "actions" for b in bk)  # buttons removed
    assert "jtomek" in json.dumps(bk)
    assert "request" in json.dumps(bk).lower()


def test_digest_blocks_empty_for_no_entries():
    assert blocks.build_digest_blocks([]) == []


def test_digest_within_50_block_limit_for_full_batch():
    entries = [
        dict(_entry(number=n, url=f"u{n}"), _key=f"Strike48/matrix#{n}")
        for n in range(6)  # per-run cap
    ]
    bk = blocks.build_digest_blocks(entries)
    assert len(bk) <= 50


# ---------------------------------------------------------------------------
# github.py — gh wrapper decisions (subprocess mocked)
# ---------------------------------------------------------------------------


class _FakeRun:
    """Records calls and returns queued CompletedProcess results."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, argv, *a, **k):
        self.calls.append(argv)
        rc, out, err = self._results.pop(0)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)


def test_current_head_sha_parses_json():
    run = _FakeRun([(0, '{"headRefOid": "deadbeef"}', "")])
    sha = github.current_head_sha("Strike48/matrix", 42, run=run)
    assert sha == "deadbeef"


def test_current_head_sha_none_on_failure():
    run = _FakeRun([(1, "", "not found")])
    assert github.current_head_sha("Strike48/matrix", 42, run=run) is None


def test_submit_review_maps_event_to_flag():
    run = _FakeRun([(0, "", "")])
    res = github.submit_review("Strike48/matrix", 42, "approve", "LGTM", run=run)
    assert res.ok
    argv = run.calls[0]
    assert "pr" in argv and "review" in argv
    assert "--approve" in argv
    assert "-R" in argv and "Strike48/matrix" in argv


def test_submit_review_request_changes_flag():
    run = _FakeRun([(0, "", "")])
    github.submit_review("Strike48/matrix", 42, "request-changes", "fix", run=run)
    assert "--request-changes" in run.calls[0]


def test_submit_review_own_pr_is_refused_not_posted():
    # gh prints the own-PR error to stderr while EXITING 0. We must detect that
    # (not trust rc) and REFUSE — no comment fallback, nothing posted, own_pr set.
    run = _FakeRun([
        (0, "", "failed to create review: GraphQL: Can not approve your own pull request (addPullRequestReview)"),
    ])
    res = github.submit_review("Strike48/matrix", 42, "approve", "LGTM", run=run)
    assert not res.ok
    assert res.own_pr
    assert len(run.calls) == 1  # no second call — did NOT fall back to a comment


def test_submit_review_rc0_with_stderr_is_not_reported_posted():
    # Regression: gh pr review exiting 0 while stderr shows a creation failure
    # (that is NOT the own-PR case) must be reported as a failure, not "posted".
    run = _FakeRun([
        (0, "", "failed to create review: GraphQL: Something else went wrong"),
    ])
    res = github.submit_review("Strike48/matrix", 42, "approve", "LGTM", run=run)
    assert not res.ok
    assert not res.own_pr
    assert "went wrong" in res.detail


def test_submit_review_hard_failure_surfaces_error():
    run = _FakeRun([(1, "", "network is down")])
    res = github.submit_review("Strike48/matrix", 42, "comment", "hi", run=run)
    assert not res.ok
    assert "network is down" in res.detail


def test_submit_review_body_passed_through():
    run = _FakeRun([(0, "", "")])
    github.submit_review("Strike48/matrix", 42, "comment", "VERBATIM-BODY", run=run)
    assert "VERBATIM-BODY" in run.calls[0]


# ---------------------------------------------------------------------------
# actions.py — click decision logic
# ---------------------------------------------------------------------------


def test_action_id_maps_to_event():
    assert actions.event_for("prreview_approve") == "approve"
    assert actions.event_for("prreview_request_changes") == "request-changes"
    assert actions.event_for("prreview_comment") == "comment"
    assert actions.event_for("garbage") is None


def test_is_authorized_requires_allowlist_membership(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    assert actions.is_authorized("U0B4FTHRY05")
    assert not actions.is_authorized("U_STRANGER")


def test_is_authorized_wildcard(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "*")
    assert actions.is_authorized("anyone")


def test_is_authorized_denies_when_unset(monkeypatch):
    # Posting is a write action and this handler is the only gate — fail closed.
    monkeypatch.delenv("SLACK_ALLOWED_USERS", raising=False)
    assert not actions.is_authorized("anyone")


def test_is_authorized_denies_when_blank(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "   ")
    assert not actions.is_authorized("anyone")


def test_stale_when_head_moved():
    assert actions.is_stale(stored_sha="abc", current_sha="def")
    assert not actions.is_stale(stored_sha="abc", current_sha="abc")
    # Unknown current sha (fetch failed): fail CLOSED — can't prove the head is
    # unchanged, so treat as stale. The entry is kept, so the operator retries.
    assert actions.is_stale(stored_sha="abc", current_sha=None)


@pytest.mark.asyncio
async def test_handle_action_posts_verbatim_body(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    p = tmp_path / "pending.json"
    store.stage(_entry(body="EXACT-REVIEW-TEXT"), path=p)

    posted = {}

    def fake_submit(repo, number, event, body, **k):
        posted.update(repo=repo, number=number, event=event, body=body)
        return github.ReviewResult(ok=True, detail="ok")

    monkeypatch.setattr(github, "submit_review", fake_submit)
    monkeypatch.setattr(github, "current_head_sha", lambda *a, **k: "abc123")

    updates = []
    fake_client = _FakeSlack(updates)
    ack = _AsyncFlag()

    await actions.handle_action(
        ack,
        _body("prreview_approve", "Strike48/matrix#42", user_id="U0B4FTHRY05"),
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
        client=fake_client,
        store_path=p,
    )

    assert ack.called
    assert posted["event"] == "approve"
    assert posted["body"] == "EXACT-REVIEW-TEXT"
    assert store.get("Strike48/matrix#42", path=p) is None  # popped after post
    assert updates  # slack message was updated to show the decision


@pytest.mark.asyncio
async def test_handle_action_rejects_unauthorized(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)

    monkeypatch.setattr(github, "submit_review",
                        lambda *a, **k: pytest.fail("must not post when unauthorized"))

    ack = _AsyncFlag()
    await actions.handle_action(
        ack,
        _body("prreview_approve", "Strike48/matrix#42", user_id="U_STRANGER"),
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
        client=_FakeSlack([]),
        store_path=p,
    )
    assert ack.called  # we still ack so Slack doesn't retry
    assert store.get("Strike48/matrix#42", path=p) is not None  # untouched


@pytest.mark.asyncio
async def test_handle_action_rejects_missing_user_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "*")  # even wide-open must not post anon
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)
    monkeypatch.setattr(github, "submit_review",
                        lambda *a, **k: pytest.fail("must not post for anonymous click"))

    body = _body("prreview_approve", "Strike48/matrix#42", user_id="")
    ack = _AsyncFlag()
    await actions.handle_action(
        ack, body,
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
        client=_FakeSlack([]), store_path=p,
    )
    assert ack.called
    assert store.get("Strike48/matrix#42", path=p) is not None  # untouched


@pytest.mark.asyncio
async def test_handle_action_blocks_stale_post(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    p = tmp_path / "pending.json"
    store.stage(_entry(head_sha="OLD"), path=p)

    monkeypatch.setattr(github, "current_head_sha", lambda *a, **k: "NEW")
    monkeypatch.setattr(github, "submit_review",
                        lambda *a, **k: pytest.fail("must not post a stale review"))

    updates = []
    await actions.handle_action(
        _AsyncFlag(),
        _body("prreview_approve", "Strike48/matrix#42", user_id="U0B4FTHRY05"),
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
        client=_FakeSlack(updates),
        store_path=p,
    )
    # Stale: not posted, entry kept so the operator can re-review, message warns.
    assert store.get("Strike48/matrix#42", path=p) is not None
    assert "moved" in json.dumps(updates).lower() or "stale" in json.dumps(updates).lower()


@pytest.mark.asyncio
async def test_handle_action_own_pr_skips_without_posting(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)
    monkeypatch.setattr(github, "current_head_sha", lambda *a, **k: "abc123")
    monkeypatch.setattr(github, "submit_review",
                        lambda *a, **k: github.ReviewResult(ok=False, own_pr=True,
                                                            detail="skipped: cannot review your own PR"))

    updates = []
    await actions.handle_action(
        _AsyncFlag(),
        _body("prreview_approve", "Strike48/matrix#42", user_id="U0B4FTHRY05"),
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
        client=_FakeSlack(updates), store_path=p,
    )
    # Refused: dropped (not re-staged), message says own-PR skip, no buttons.
    assert store.get("Strike48/matrix#42", path=p) is None
    blob = json.dumps(updates).lower()
    assert "own pr" in blob
    assert "actions" not in blob  # no button row on the rewritten message


@pytest.mark.asyncio
async def test_handle_action_double_click_posts_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U0B4FTHRY05")
    p = tmp_path / "pending.json"
    store.stage(_entry(), path=p)
    monkeypatch.setattr(github, "current_head_sha", lambda *a, **k: "abc123")

    calls = []
    monkeypatch.setattr(github, "submit_review",
                        lambda *a, **k: (calls.append(1), github.ReviewResult(ok=True, detail="ok"))[1])

    args = (
        _body("prreview_approve", "Strike48/matrix#42", user_id="U0B4FTHRY05"),
        {"action_id": "prreview_approve", "value": "Strike48/matrix#42"},
    )
    await actions.handle_action(_AsyncFlag(), *args, client=_FakeSlack([]), store_path=p)
    await actions.handle_action(_AsyncFlag(), *args, client=_FakeSlack([]), store_path=p)
    assert len(calls) == 1  # second click found nothing to post


# ---------------------------------------------------------------------------
# slackio.py — stdlib Slack client (no slack_sdk dependency)
# ---------------------------------------------------------------------------


def test_make_client_needs_no_slack_sdk(monkeypatch):
    # The whole point of the rewrite: publish must work without slack_sdk. Hide
    # it and confirm the client still builds.
    import builtins

    real_import = builtins.__import__

    def _no_slack_sdk(name, *a, **k):
        if name.startswith("slack_sdk"):
            raise ImportError("slack_sdk is not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_slack_sdk)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    client = slackio.make_client()
    assert client is not None


def test_make_client_raises_without_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        slackio.make_client()


@pytest.mark.asyncio
async def test_stdlib_client_parses_ts_and_raises_on_error(monkeypatch):
    client = slackio.make_client(token="xoxb-test")
    responses = {"chat.postMessage": {"ok": True, "ts": "1700.5"}}

    def fake_sync(method, payload):
        r = responses.get(method)
        if r is None or not r.get("ok"):
            raise slackio.SlackError(f"{method}: boom")
        return r

    monkeypatch.setattr(client, "_call_sync", fake_sync)
    ts = await slackio.post_message(client, "C1", "hi", [{"type": "divider"}])
    assert ts == "1700.5"

    with pytest.raises(slackio.SlackError):
        await slackio.update_message(client, "C1", "1700.5", "x", [])


def test_bot_token_takes_first_of_csv(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-a, xoxb-b")
    assert slackio.bot_token() == "xoxb-a"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _AsyncFlag:
    def __init__(self):
        self.called = False

    async def __call__(self, *a, **k):
        self.called = True


class _FakeSlack:
    """Minimal async Slack client capturing chat_update / chat_postMessage."""

    def __init__(self, sink):
        self.sink = sink

    async def chat_update(self, **kwargs):
        self.sink.append(("update", kwargs))
        return {"ok": True, "ts": kwargs.get("ts")}

    async def chat_postMessage(self, **kwargs):
        self.sink.append(("post", kwargs))
        return {"ok": True, "ts": "1700.9"}


def _body(action_id, value, *, user_id):
    return {
        "user": {"id": user_id, "name": "jtomek"},
        "channel": {"id": "C0BHM1F7W10"},
        "message": {"ts": "1700.1", "blocks": []},
    }
