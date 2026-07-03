"""Tests for /v1/runs `attachments` handling (KarinAI managed runs).

The backend materializes a conversation's uploaded files into the workspace
before dispatch and sends a manifest in body["attachments"]. The handler must
surface them to the agent: a context note per file (platform-document style),
inlined content for small text files, dedup across runs of the same session
(the backend resends the full manifest every run), and strict manifest
validation. Covers the pure helpers directly plus the HTTP path end-to-end.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server as api_server_mod
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _prepare_run_attachment_blocks,
    _read_inline_attachment_text,
    _validate_run_attachments,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _manifest_entry(path, name=None, mime="text/csv", file_id="file_1"):
    return {
        "file_id": file_id,
        "safe_name": name or getattr(path, "name", str(path).rsplit("/", 1)[-1]),
        "mime": mime,
        "size": 64,
        "local_path": str(path),
        "purpose": "attachment",
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A tmp 'workspace' accepted as the inline root."""
    monkeypatch.setattr(api_server_mod, "ATTACHMENT_INLINE_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestValidateRunAttachments:
    def test_accepts_well_formed_manifest(self, tmp_path):
        assert _validate_run_attachments([_manifest_entry(tmp_path / "a.csv")]) is None

    def test_rejects_non_list(self):
        assert "must be an array" in _validate_run_attachments({"safe_name": "x"})

    def test_rejects_entry_missing_required_fields(self):
        err = _validate_run_attachments([{"safe_name": "a.csv"}])
        assert err == "attachments[0] must have non-empty 'local_path'"
        err = _validate_run_attachments([{"local_path": "/workspace/x"}])
        assert err == "attachments[0] must have non-empty 'safe_name'"
        assert "must be an object" in _validate_run_attachments(["nope"])


class TestInlineRead:
    def test_reads_utf8_text_under_root(self, workspace):
        f = workspace / "inputs" / "c1" / "f1" / "a.csv"
        f.parent.mkdir(parents=True)
        f.write_text("region,code\nEast,CHARLIE-8\n", encoding="utf-8")
        assert _read_inline_attachment_text(str(f)) == "region,code\nEast,CHARLIE-8\n"

    def test_refuses_paths_outside_the_inline_root(self, workspace, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "secrets.txt"
        outside.write_text("nope", encoding="utf-8")
        assert _read_inline_attachment_text(str(outside)) is None

    def test_over_cap_is_all_or_nothing(self, workspace, monkeypatch):
        monkeypatch.setattr(api_server_mod, "MAX_ATTACHMENT_INLINE_BYTES", 10)
        f = workspace / "big.csv"
        f.write_text("x" * 11, encoding="utf-8")
        assert _read_inline_attachment_text(str(f)) is None

    def test_non_utf8_degrades_to_none(self, workspace):
        f = workspace / "weird.csv"
        f.write_bytes(b"\xff\xfe\x00bad")
        assert _read_inline_attachment_text(str(f)) is None

    def test_missing_file_degrades_to_none(self, workspace):
        assert _read_inline_attachment_text(str(workspace / "gone.csv")) is None


class TestPrepareBlocks:
    def test_text_file_gets_note_plus_inlined_content(self, workspace):
        f = workspace / "a.csv"
        f.write_text("k,v\n1,2\n", encoding="utf-8")
        blocks, noted = _prepare_run_attachment_blocks([_manifest_entry(f)], set())

        assert len(blocks) == 2
        assert "text document: 'a.csv'" in blocks[0]
        assert "content has been included below" in blocks[0]
        # The note carries the workspace-RELATIVE path (agent cwd = workspace),
        # so absolute sandbox paths never flow into user-visible transcripts.
        assert "saved at: a.csv" in blocks[0]
        assert str(f) not in blocks[0]
        assert blocks[1] == "[Content of a.csv]:\nk,v\n1,2\n"
        assert noted == {"file_1"}

    def test_binary_file_gets_self_extraction_note_only(self, workspace):
        f = workspace / "report.pdf"
        f.write_bytes(b"%PDF-1.4 ...")
        blocks, _ = _prepare_run_attachment_blocks(
            [_manifest_entry(f, name="report.pdf", mime="application/pdf")], set()
        )

        assert len(blocks) == 1
        assert "extract the document's text yourself" in blocks[0]
        assert "report.pdf" in blocks[0]
        assert str(f) not in blocks[0]  # relative, not absolute

    def test_oversized_text_degrades_to_self_extraction_note(self, workspace, monkeypatch):
        monkeypatch.setattr(api_server_mod, "MAX_ATTACHMENT_INLINE_BYTES", 4)
        f = workspace / "big.csv"
        f.write_text("too big to inline", encoding="utf-8")
        blocks, _ = _prepare_run_attachment_blocks([_manifest_entry(f)], set())

        # Never claim content was included when it wasn't.
        assert len(blocks) == 1
        assert "content has been included below" not in blocks[0]
        assert "extract the document's text yourself" in blocks[0]

    def test_already_inlined_file_still_gets_a_path_note(self, workspace):
        # Notes re-inject on EVERY run — /v1/runs context is per-request, so a
        # note injected only once would vanish from turn 2 onward. Only the
        # content inlining is deduped.
        f = workspace / "a.csv"
        f.write_text("k,v\n", encoding="utf-8")
        blocks, inlined = _prepare_run_attachment_blocks([_manifest_entry(f)], {"file_1"})

        assert len(blocks) == 1
        assert "a.csv" in blocks[0]
        assert "extract the document's text yourself" in blocks[0]
        assert "content has been included below" not in blocks[0]
        assert inlined == set()  # nothing NEWLY inlined

    def test_duplicate_entries_in_one_manifest_surfaced_once(self, workspace):
        f = workspace / "a.csv"
        f.write_text("k,v\n", encoding="utf-8")
        entry = _manifest_entry(f)
        blocks, inlined = _prepare_run_attachment_blocks([entry, dict(entry)], set())
        assert len(blocks) == 2  # one note + one content block
        assert inlined == {"file_1"}

    def test_aggregate_inline_budget_bounds_many_file_manifests(self, workspace, monkeypatch):
        monkeypatch.setattr(api_server_mod, "MAX_ATTACHMENT_INLINE_TOTAL_BYTES", 10)
        f1 = workspace / "a.csv"
        f1.write_text("12345678", encoding="utf-8")  # 8 bytes: fits
        f2 = workspace / "b.csv"
        f2.write_text("123456", encoding="utf-8")  # 6 bytes: would exceed 10 total

        blocks, inlined = _prepare_run_attachment_blocks(
            [_manifest_entry(f1), _manifest_entry(f2, file_id="file_2")], set()
        )

        assert "[Content of a.csv]:\n12345678" in blocks
        assert not any("[Content of b.csv]" in b for b in blocks)
        assert any("b.csv" in b and "extract the document's text yourself" in b for b in blocks)
        # b.csv is NOT marked inlined — a later run (fresh budget) may inline it.
        assert inlined == {"file_1"}

    def test_display_name_is_sanitized(self, workspace):
        f = workspace / "a.csv"
        f.write_text("k,v\n", encoding="utf-8")
        blocks, _ = _prepare_run_attachment_blocks(
            [_manifest_entry(f, name="we`ird$na;me.csv")], set()
        )
        assert "we_ird_na_me.csv" in blocks[0]
        assert "`" not in blocks[0].split("saved at")[0]

    def test_note_path_is_workspace_relative(self, workspace):
        # The materialized layout: <root>/inputs/<conversation>/<file_id>/<name>.
        # The note must carry the RELATIVE path (agent cwd = workspace): the note
        # text flows into user-visible transcripts and public run events, where
        # an absolute sandbox path is internal detail (leak-scrubber territory).
        f = workspace / "inputs" / "cnv_1" / "file_1" / "sales.csv"
        f.parent.mkdir(parents=True)
        f.write_text("k,v\n", encoding="utf-8")
        blocks, _ = _prepare_run_attachment_blocks([_manifest_entry(f, name="sales.csv")], set())

        assert "saved at: inputs/cnv_1/file_1/sales.csv" in blocks[0]
        assert str(workspace) not in blocks[0]  # absolute sandbox path never appears

    def test_note_path_outside_root_stays_as_sent(self, workspace, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "doc.pdf"
        outside.write_bytes(b"%PDF")
        blocks, _ = _prepare_run_attachment_blocks(
            [_manifest_entry(outside, name="doc.pdf", mime="application/pdf")], set()
        )
        assert str(outside) in blocks[0]  # non-managed caller: path passed through


# ---------------------------------------------------------------------------
# HTTP path: POST /v1/runs with attachments
# ---------------------------------------------------------------------------


def _make_capturing_agent():
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


async def _wait_for_run_completion(cli, run_id, timeout=5.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        resp = await cli.get(f"/v1/runs/{run_id}")
        data = await resp.json()
        if data.get("status") in {"completed", "failed"}:
            return data
        await asyncio.sleep(0.05)
    raise AssertionError("run did not complete in time")


class TestRunsWithAttachments:
    @pytest.mark.asyncio
    async def test_attachment_note_and_content_reach_the_agent(self, workspace):
        f = workspace / "inputs" / "cnv_1" / "file_1" / "sales.csv"
        f.parent.mkdir(parents=True)
        f.write_text("code,revenue\nCHARLIE-8,90731\n", encoding="utf-8")

        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "What is the revenue for CHARLIE-8?",
                        "session_id": "cnv_1",
                        "attachments": [_manifest_entry(f, name="sales.csv")],
                    },
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_for_run_completion(cli, run_id)

        sent = agent.run_conversation.call_args.kwargs["user_message"]
        assert "text document: 'sales.csv'" in sent
        assert "[Content of sales.csv]:\ncode,revenue\nCHARLIE-8,90731" in sent
        # The user's own message stays last, unmodified.
        assert sent.endswith("What is the revenue for CHARLIE-8?")

    @pytest.mark.asyncio
    async def test_second_run_keeps_the_note_but_not_the_content(self, workspace):
        # Regression (review finding): /v1/runs context is per-request and the
        # backend resends CLEAN history — a note injected only once would make
        # the agent forget the file exists from turn 2 onward. Notes re-inject
        # every run; only the content inlining is once-per-session.
        f = workspace / "a.csv"
        f.write_text("k,v\n", encoding="utf-8")
        manifest = [_manifest_entry(f)]

        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                for message in ("first turn", "second turn"):
                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": message, "session_id": "cnv_dedup", "attachments": manifest},
                    )
                    assert resp.status == 202
                    await _wait_for_run_completion(cli, (await resp.json())["run_id"])

        first_sent = agent.run_conversation.call_args_list[0].kwargs["user_message"]
        second_sent = agent.run_conversation.call_args_list[1].kwargs["user_message"]
        assert "[Content of a.csv]" in first_sent
        # Second run: the note (with the relative path) is still there...
        assert "a.csv" in second_sent
        # ...as a workspace-relative path (never the absolute sandbox path)...
        assert str(f) not in second_sent
        # ...but the content is not re-inlined, and the user text stays last.
        assert "[Content of a.csv]" not in second_sent
        assert second_sent.endswith("second turn")

    @pytest.mark.asyncio
    async def test_new_file_in_later_run_inlines_only_the_new_content(self, workspace):
        f1 = workspace / "a.csv"
        f1.write_text("k,v\n", encoding="utf-8")
        f2 = workspace / "b.csv"
        f2.write_text("x,y\n", encoding="utf-8")

        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "one", "session_id": "s1", "attachments": [_manifest_entry(f1)]},
                )
                await _wait_for_run_completion(cli, (await resp.json())["run_id"])
                # Backend resends the full manifest, now with a second file.
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "two",
                        "session_id": "s1",
                        "attachments": [
                            _manifest_entry(f1),
                            _manifest_entry(f2, name="b.csv", file_id="file_2"),
                        ],
                    },
                )
                await _wait_for_run_completion(cli, (await resp.json())["run_id"])

        second_sent = agent.run_conversation.call_args_list[1].kwargs["user_message"]
        assert "[Content of b.csv]" in second_sent  # new file: inlined
        assert "[Content of a.csv]" not in second_sent  # old file: content once only
        assert "a.csv" in second_sent  # old file: note still present every run

    @pytest.mark.asyncio
    async def test_preflight_failure_does_not_swallow_the_inline(self, workspace):
        # Regression (review finding): committing the inline dedup at request
        # time meant a run that failed BEFORE reaching the agent permanently
        # swallowed the content for the session. The commit now happens after
        # run_conversation executes, so the backend's retry re-inlines.
        f = workspace / "a.csv"
        f.write_text("k,v\n1,2\n", encoding="utf-8")
        manifest = [_manifest_entry(f)]

        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(
            adapter, "_create_agent", side_effect=[RuntimeError("transient boot failure"), agent]
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "try one", "session_id": "s_retry", "attachments": manifest},
                )
                assert resp.status == 202
                first = await _wait_for_run_completion(cli, (await resp.json())["run_id"])
                assert first["status"] == "failed"

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "retry", "session_id": "s_retry", "attachments": manifest},
                )
                await _wait_for_run_completion(cli, (await resp.json())["run_id"])

        sent = agent.run_conversation.call_args.kwargs["user_message"]
        assert "[Content of a.csv]" in sent  # retry still gets the content

    @pytest.mark.asyncio
    async def test_malformed_manifest_is_rejected_400(self, workspace):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hi", "attachments": [{"safe_name": "a.csv"}]},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "attachments[0]" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_no_attachments_leaves_message_untouched(self, workspace):
        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "plain message"})
                assert resp.status == 202
                await _wait_for_run_completion(cli, (await resp.json())["run_id"])

        assert agent.run_conversation.call_args.kwargs["user_message"] == "plain message"

    @pytest.mark.asyncio
    async def test_content_parts_input_gets_context_as_text_part(self, workspace):
        f = workspace / "a.csv"
        f.write_text("k,v\n", encoding="utf-8")

        adapter = _make_adapter()
        agent = _make_capturing_agent()
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "read the file"}],
                            }
                        ],
                        "session_id": "s_parts",
                        "attachments": [_manifest_entry(f)],
                    },
                )
                assert resp.status == 202
                await _wait_for_run_completion(cli, (await resp.json())["run_id"])

        sent = agent.run_conversation.call_args.kwargs["user_message"]
        assert isinstance(sent, list)
        assert sent[0]["type"] == "text" and "[Content of a.csv]" in sent[0]["text"]
        assert sent[-1] == {"type": "text", "text": "read the file"}


class TestDedupSessionCap:
    def test_oldest_sessions_pruned_at_cap(self, monkeypatch):
        monkeypatch.setattr(api_server_mod, "MAX_ATTACHMENT_DEDUP_SESSIONS", 3)
        adapter = _make_adapter()
        for i in range(3):
            adapter._inlined_attachments_for_session(f"s{i}").add("f")
        assert list(adapter._inlined_run_attachments) == ["s0", "s1", "s2"]

        adapter._inlined_attachments_for_session("s3")
        assert "s0" not in adapter._inlined_run_attachments
        assert "s3" in adapter._inlined_run_attachments
