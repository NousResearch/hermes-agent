"""Catalog / Hybrid / Standard compaction modes on ContextCompressor."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from agent.catalog_residual import (
    CATALOG_BUDGET_CHARS,
    CATALOG_HEADING,
    HYBRID_INDEX_HEADING,
    LEAN_ANCHOR_HEADING,
    normalize_compression_mode,
)
from tools.session_search_tool import (
    SESSION_SEARCH_DISCOVERY_CALL,
    SESSION_SEARCH_DISCOVERY_HINT,
)
from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _NO_USER_TASK_SENTINEL,
)
from hermes_state import SessionDB
from tools.session_search_tool import session_search


SECRET = "sk-proj-" + ("a" * 40)
_SESSION_SEARCH_CALL_RE = re.compile(r"session_search\(([^)]*)\)")


def _session_search_call_from_footer(text: str) -> dict[str, str]:
    """Parse the taught session_search(...) invocation from a residual."""
    match = _SESSION_SEARCH_CALL_RE.search(text)
    assert match, text
    taught: dict[str, str] = {}
    for key, raw in re.findall(
        r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^\s,]+)", match.group(1)
    ):
        taught[key] = raw.strip("'\"")
    return taught


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _long_transcript(*, unique_middle: str = "spectral phoenix bait"):
    """Enough turns to create a compressible middle under protect 2/2.

    The first exchange is generic so extractable files/tools land in the
    compacted middle, not the protected head.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, start when ready."},
        {"role": "assistant", "content": "Ready."},
        {"role": "user", "content": "Start the project in /tmp/project/app.py"},
        {
            "role": "assistant",
            "content": "Opening the file.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/project/app.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": f"{unique_middle} lives only in this compacted middle.\n"
            + ("x" * 4200),
        },
        {
            "role": "user",
            "content": "Also review https://example.com/spec and ticket HERMES-99",
        },
        {"role": "assistant", "content": "Reviewed the spec."},
        {"role": "user", "content": "Keep going on /tmp/project/app.py"},
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": "current live request should stay in tail"},
        {"role": "assistant", "content": "Acknowledged live tail."},
    ]


def _compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=2,
        protect_last_n=2,
    )
    defaults.update(kwargs)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(**defaults)


class TestModeNormalization:
    def test_default_is_standard(self):
        c = _compressor()
        assert c.mode == "standard"

    def test_invalid_falls_back_to_standard(self):
        c = _compressor(mode="banana")
        assert c.mode == "standard"
        assert normalize_compression_mode("nope") == "standard"

    def test_config_default_is_standard(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["compression"]["mode"] == "standard"


class TestStandardUnchanged:
    def test_standard_calls_llm_and_keeps_head_tail(self):
        c = _compressor(mode="standard")
        msgs = _long_transcript()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue the file work."),
        ) as llm:
            out = c.compress(msgs)
        assert llm.called
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert SUMMARY_PREFIX in text
        assert "You are a helpful assistant." in text
        assert "current live request should stay in tail" in text
        assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in out)
        assert CATALOG_HEADING not in text
        assert HYBRID_INDEX_HEADING not in text


class TestCatalogMode:
    def test_catalog_skips_llm_preserves_head_tail_and_metadata(self):
        c = _compressor(mode="catalog")
        msgs = _long_transcript()
        with patch("agent.context_compressor.call_llm") as llm:
            out = c.compress(msgs)
        llm.assert_not_called()
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert SUMMARY_PREFIX in text
        assert CATALOG_HEADING in text
        assert "/tmp/project/app.py" in text
        assert "read_file" in text
        assert "You are a helpful assistant." in text
        assert "current live request should stay in tail" in text
        summary = next(m for m in out if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
        assert summary[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is True
        assert c._last_summary_fallback_used is False

    def test_compression_path_has_no_memory_store_dependency(self):
        import ast
        from pathlib import Path

        import agent.catalog_residual as catalog_mod
        import agent.context_compressor as compressor_mod
        from tools.memory_tool import MemoryStore

        forbidden = {"MemoryStore", "memory_tool"}
        for module in (catalog_mod, compressor_mod):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert "memory_tool" not in (node.module or "")
                    for alias in node.names:
                        assert alias.name not in forbidden
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "memory_tool" not in alias.name
                elif isinstance(node, ast.Name):
                    assert node.id not in forbidden
                elif isinstance(node, ast.Attribute):
                    assert node.attr not in forbidden

        c = _compressor(mode="catalog")
        with (
            patch("agent.context_compressor.call_llm") as llm,
            patch.object(MemoryStore, "add") as add,
            patch.object(MemoryStore, "replace") as replace,
            patch.object(MemoryStore, "remove") as remove,
        ):
            out = c.compress(_long_transcript())
        llm.assert_not_called()
        add.assert_not_called()
        replace.assert_not_called()
        remove.assert_not_called()
        assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in out)

    def test_catalog_redacts_secrets(self):
        c = _compressor(mode="catalog")
        msgs = _long_transcript()
        msgs[1] = {
            "role": "user",
            "content": f"Put {SECRET} into /tmp/project/app.py",
        }
        with patch("agent.redact._REDACT_ENABLED", False):
            out = c.compress(msgs)
        text = "\n".join(
            str(m.get("content") or "")
            for m in out
            if m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        )
        assert SECRET not in text
        assert "sk-proj-" not in text

    @pytest.mark.parametrize("abort_flag", [False, True])
    def test_catalog_construction_failure_always_aborts(self, abort_flag):
        c = _compressor(mode="catalog", abort_on_summary_failure=abort_flag)
        original = _long_transcript()
        with patch(
            "agent.context_compressor.build_catalog_residual",
            side_effect=RuntimeError("catalog boom"),
        ):
            out = c.compress(original)
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert out == original
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert "Summary generation was unavailable" not in text
        assert "deterministic fallback" not in text

    def test_catalog_lean_adds_recovery_without_summarizer_or_lean_aids(self):
        c = _compressor(mode="catalog", tail_mode="lean")
        c._session_id = "s_catalog_lean"
        with (
            patch("agent.context_compressor.call_llm") as llm,
            patch.object(ContextCompressor, "_build_chunk_digests") as digests,
            patch.object(ContextCompressor, "_augment_summary_lean") as lean_aids,
        ):
            out = c.compress(_long_transcript())
        llm.assert_not_called()
        digests.assert_not_called()
        lean_aids.assert_not_called()
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert "## Context Recovery" in text
        assert "session_search" in text
        assert "s_catalog_lean" in text
        taught = _session_search_call_from_footer(text)
        assert SESSION_SEARCH_DISCOVERY_CALL in text
        assert SESSION_SEARCH_DISCOVERY_HINT in text
        assert "session_id" not in taught
        assert taught["role_filter"] == "user,assistant,tool"
        assert "## Detailed Session Log" not in text
        assert "## Anchor Index (mechanically extracted, exact)" not in text
        assert "## User Messages (verbatim" not in text
        assert CATALOG_HEADING in text

    def test_catalog_preserves_full_historical_task_snapshot(self):
        long_ask = (
            "Please implement the full migration of the auth module across "
            "every service boundary and keep the rollback plan ready. "
            + ("auth module " * 12)
        )
        assert len(long_ask) > 200
        msgs = _long_transcript()
        msgs[8] = {"role": "user", "content": long_ask}
        c = _compressor(mode="catalog")
        out = c.compress(msgs)
        summary = next(m for m in out if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
        text = str(summary.get("content") or "")
        assert "User asked (deterministic" in text
        assert "Historical only" in text
        assert "auth module" in text
        assert long_ask[:80] in text
        assert summary[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is True

    def test_archived_middle_remains_session_searchable(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        db.create_session("s_catalog", source="cli")
        unique = "spectral phoenix only spawns during full moons"
        db.append_message("s_catalog", role="user", content="Start the project")
        db.append_message("s_catalog", role="assistant", content="Working")
        db.append_message("s_catalog", role="user", content=unique)
        db.append_message("s_catalog", role="assistant", content="Noted the phoenix")
        c = _compressor(mode="catalog")
        live = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Start the project"},
            {"role": "assistant", "content": "Working"},
            {"role": "user", "content": unique},
            {"role": "assistant", "content": "Noted the phoenix"},
            {"role": "user", "content": "tail ask"},
            {"role": "assistant", "content": "tail reply"},
            {"role": "user", "content": "live"},
            {"role": "assistant", "content": "live ack"},
        ]
        compressed = c.compress(live)
        db.archive_and_compact("s_catalog", compressed)
        result = json.loads(
            session_search(
                query="spectral phoenix",
                db=db,
                current_session_id="s_catalog",
            )
        )
        assert result["success"] is True
        assert result["count"] >= 1
        assert result["results"][0]["session_id"] == "s_catalog"

    def test_catalog_lean_footer_recovers_tool_only_needle(self, tmp_path):
        """Footer-taught discovery must find a tool-role compacted needle."""
        needle = "umbravore lanternwort blooms only after frost"
        db = SessionDB(tmp_path / "state.db")
        sid = "s_catalog_lean_tool"
        db.create_session(sid, source="cli")
        live = _long_transcript(unique_middle=needle)
        for msg in live:
            db.append_message(
                sid,
                role=msg["role"],
                content=msg.get("content") or "",
                tool_name=msg.get("name") or msg.get("tool_name"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
        c = _compressor(mode="catalog", tail_mode="lean")
        c._session_id = sid
        with patch("agent.context_compressor.call_llm") as llm:
            compressed = c.compress(live)
        llm.assert_not_called()
        summary = next(m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
        footer_text = str(summary.get("content") or "")
        assert needle not in footer_text
        assert CATALOG_HEADING in footer_text
        taught = _session_search_call_from_footer(footer_text)
        assert SESSION_SEARCH_DISCOVERY_CALL in footer_text
        assert SESSION_SEARCH_DISCOVERY_HINT in footer_text
        assert "session_id" not in taught
        assert taught["role_filter"] == "user,assistant,tool"
        assert taught["query"] == "<keywords>"
        db.archive_and_compact(sid, compressed)
        result = json.loads(
            session_search(
                query=needle,
                role_filter=taught["role_filter"],
                db=db,
                current_session_id=sid,
            )
        )
        assert result["success"] is True
        assert result["count"] >= 1
        blob = json.dumps(result)
        assert needle in blob

    def test_catalog_skips_when_window_smaller_than_budget(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Edit /tmp/tiny.py"},
            {"role": "assistant", "content": "Editing."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "live tail"},
            {"role": "assistant", "content": "ack"},
        ]
        catalog = _compressor(mode="catalog")
        skip_path = _compressor(mode="standard")
        skip_path._ineffective_compression_count = 1
        with patch("agent.context_compressor.call_llm") as llm:
            catalog_out = catalog.compress(list(msgs))
            skip_out = skip_path.compress(list(msgs), force=False)
        llm.assert_not_called()
        catalog_text = "\n".join(str(m.get("content") or "") for m in catalog_out)
        skip_text = "\n".join(str(m.get("content") or "") for m in skip_out)
        assert CATALOG_HEADING not in catalog_text
        assert catalog._last_feasibility_skip is True
        assert (
            catalog._last_compression_telemetry.get("catalog_skip_reason")
            == "window_smaller_than_budget"
        )
        assert len(catalog_text) <= len(skip_text)

    def test_catalog_force_builds_residual_for_small_window(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Edit /tmp/tiny.py"},
            {"role": "assistant", "content": "Editing."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "live tail"},
            {"role": "assistant", "content": "ack"},
        ]
        c = _compressor(mode="catalog")
        with patch("agent.context_compressor.call_llm") as llm:
            out = c.compress(msgs, force=True)
        llm.assert_not_called()
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert CATALOG_HEADING in text
        assert "/tmp/tiny.py" in text
        assert c._last_feasibility_skip is False
        assert "catalog_skip_reason" not in (c._last_compression_telemetry or {})

    def test_catalog_min_reclaim_skip_after_ineffective_strike(self):
        c = _compressor(mode="catalog")
        c._ineffective_compression_count = 1
        msgs = _long_transcript()
        middle_chars = sum(
            len(str(m.get("content") or "")) for m in msgs[2:-2]
        )
        assert middle_chars >= CATALOG_BUDGET_CHARS
        with patch("agent.context_compressor.call_llm") as llm:
            out = c.compress(msgs, force=False)
        llm.assert_not_called()
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert CATALOG_HEADING not in text
        assert c._last_feasibility_skip is True
        assert (
            c._last_compression_telemetry.get("catalog_skip_reason")
            == "feasibility_min_reclaim"
        )

    def test_catalog_lean_two_pass_keeps_first_window_files(self):
        first = _long_transcript()
        for msg in first:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
            for tc in msg.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments", "")
                tc["function"]["arguments"] = args.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
        c = _compressor(mode="catalog", tail_mode="lean")
        with patch("agent.context_compressor.call_llm") as llm:
            out1 = c.compress(first)
        llm.assert_not_called()
        text1 = "\n".join(str(m.get("content") or "") for m in out1)
        assert CATALOG_HEADING in text1
        assert "/tmp/first-only.py" in text1
        assert text1.count("/tmp/first-only.py") >= 1

        second = list(out1)
        second.extend([
            {"role": "user", "content": "Now work on /tmp/second-pass.py"},
            {
                "role": "assistant",
                "content": "Working the second pass.\n" + ("x" * 4200),
            },
            {"role": "user", "content": "Continue second pass on /tmp/second-pass.py"},
            {
                "role": "assistant",
                "content": "Still working second.\n" + ("y" * 4200),
            },
            {"role": "user", "content": "live second tail"},
            {"role": "assistant", "content": "live second ack"},
        ])
        with patch("agent.context_compressor.call_llm") as llm:
            out2 = c.compress(second)
        llm.assert_not_called()
        text2 = "\n".join(str(m.get("content") or "") for m in out2)
        assert CATALOG_HEADING in text2
        assert "/tmp/first-only.py" in text2
        assert "/tmp/second-pass.py" in text2
        files_section = text2.split("## Files", 1)[-1].split("## ", 1)[0]
        assert files_section.count("/tmp/first-only.py") == 1


class TestHybridMode:
    def test_hybrid_is_summary_plus_one_index(self):
        c = _compressor(mode="hybrid")
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue the file work."),
        ) as llm:
            out = c.compress(_long_transcript())
        assert llm.called
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert SUMMARY_PREFIX in text
        assert "Continue the file work." in text
        assert HYBRID_INDEX_HEADING in text
        assert text.count(HYBRID_INDEX_HEADING) == 1
        assert CATALOG_HEADING not in text
        assert "/tmp/project/app.py" in text

    def test_hybrid_does_not_double_append_lean_sections(self):
        c = _compressor(mode="hybrid", tail_mode="lean")
        lean_summary = (
            "## Goal\nContinue.\n\n"
            "## Anchor Index (mechanically extracted, exact)\n"
            "files: /tmp/project/app.py\n\n"
            "## Unique handles\n"
            "files: /tmp/project/app.py"
        )
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(lean_summary),
        ):
            out = c.compress(_long_transcript())
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert text.count("## Anchor Index (mechanically extracted, exact)") == 1
        assert HYBRID_INDEX_HEADING not in text
        assert text.count("files: /tmp/project/app.py") == 1

    def test_hybrid_replaces_llm_echoed_unique_handles(self):
        c = _compressor(mode="hybrid")
        echoed = (
            "## Goal\nContinue the file work.\n\n"
            "## Unique handles\nfiles: /tmp/stale-echo.py"
        )
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(echoed),
        ):
            out = c.compress(_long_transcript())
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert text.count(HYBRID_INDEX_HEADING) == 1
        assert "/tmp/project/app.py" in text
        assert "/tmp/stale-echo.py" not in text

    def test_hybrid_reingests_prior_residual_across_compactions(self):
        first = _long_transcript()
        for msg in first:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
            for tc in msg.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments", "")
                tc["function"]["arguments"] = args.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
        c = _compressor(mode="hybrid")
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue first window."),
        ):
            out1 = c.compress(first)
        text1 = "\n".join(str(m.get("content") or "") for m in out1)
        assert "/tmp/first-only.py" in text1
        assert text1.count(HYBRID_INDEX_HEADING) == 1

        second = list(out1)
        second.extend([
            {"role": "user", "content": "Now work on /tmp/second-pass.py"},
            {"role": "assistant", "content": "Working the second pass. " + ("x" * 80)},
            {"role": "user", "content": "Continue second pass on /tmp/second-pass.py"},
            {"role": "assistant", "content": "Still working second. " + ("y" * 80)},
            {"role": "user", "content": "live second tail"},
            {"role": "assistant", "content": "live second ack"},
        ])
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue second window."),
        ):
            out2 = c.compress(second)
        text2 = "\n".join(str(m.get("content") or "") for m in out2)
        assert text2.count(HYBRID_INDEX_HEADING) == 1
        assert "/tmp/first-only.py" in text2
        assert "/tmp/second-pass.py" in text2

    def test_hybrid_lean_preserves_first_window_handles_without_duplicate_index(self):
        first = _long_transcript()
        for msg in first:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
            for tc in msg.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments", "")
                tc["function"]["arguments"] = args.replace(
                    "/tmp/project/app.py", "/tmp/first-only.py"
                )
        c = _compressor(mode="hybrid", tail_mode="lean")
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue first window."),
        ):
            out1 = c.compress(first)
        text1 = "\n".join(str(m.get("content") or "") for m in out1)
        assert "/tmp/first-only.py" in text1
        assert text1.count(LEAN_ANCHOR_HEADING) == 1
        assert HYBRID_INDEX_HEADING not in text1

        second = list(out1)
        second.extend([
            {"role": "user", "content": "Now work on /tmp/second-pass.py"},
            {"role": "assistant", "content": "Working the second pass. " + ("x" * 80)},
            {"role": "user", "content": "Continue second pass on /tmp/second-pass.py"},
            {"role": "assistant", "content": "Still working second. " + ("y" * 80)},
            {"role": "user", "content": "live second tail"},
            {"role": "assistant", "content": "live second ack"},
        ])
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("## Goal\nContinue second window."),
        ):
            out2 = c.compress(second)
        text2 = "\n".join(str(m.get("content") or "") for m in out2)
        assert "/tmp/first-only.py" in text2
        assert "/tmp/second-pass.py" in text2
        assert text2.count(LEAN_ANCHOR_HEADING) == 1
        assert HYBRID_INDEX_HEADING not in text2
        assert text2.count("files:") == 1

    def test_hybrid_summary_failure_keeps_standard_fallback(self):
        c = _compressor(mode="hybrid")
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=Exception("404 model not found"),
        ):
            out = c.compress(_long_transcript())
        assert c._last_summary_fallback_used is True
        text = "\n".join(str(m.get("content") or "") for m in out)
        assert "Summary generation was unavailable" in text
        # Fallback is still the Standard residual; Hybrid then adds one index.
        assert text.count(HYBRID_INDEX_HEADING) <= 1


class TestConfigAndSchema:
    def test_web_schema_select_override(self):
        from hermes_cli.web_server import CONFIG_SCHEMA

        entry = CONFIG_SCHEMA["compression.mode"]
        assert entry["type"] == "select"
        assert entry["options"] == ["standard", "catalog", "hybrid"]
        assert entry["category"] == "compression"
        assert "context.engine=compressor" in entry["description"]
        assert "Codex" in entry["description"]

    def test_cli_show_config_displays_mode(self, tmp_path, capsys, monkeypatch):
        import os

        from hermes_cli.config import show_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        show_config()
        out = capsys.readouterr().out
        assert "Mode:" in out
        assert "standard" in out.lower()

    def test_agent_init_passes_mode(self):
        import inspect

        import agent.agent_init as agent_init

        source = inspect.getsource(agent_init)
        assert "mode=compression_mode" in source
        assert "normalize_compression_mode" in source
        # Plugin engines stay on their own contract — do not inject mode.
        assert source.count("mode=compression_mode") == 1


@pytest.mark.parametrize("mode", ["standard", "catalog", "hybrid"])
def test_zero_user_preserves_sentinel_and_metadata(mode):
    c = _compressor(mode=mode)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "cron start"},
        {"role": "assistant", "content": "cron mid 1 " + ("x" * 40)},
        {"role": "assistant", "content": "cron mid 2 " + ("x" * 40)},
        {"role": "assistant", "content": "cron mid 3 " + ("x" * 40)},
        {"role": "assistant", "content": "cron mid 4 " + ("x" * 40)},
        {"role": "assistant", "content": "cron mid 5 " + ("x" * 40)},
        {"role": "assistant", "content": "cron tail"},
    ]
    summary_text = (
        f"## Historical Task Snapshot\n{_NO_USER_TASK_SENTINEL}\n\n"
        "## Goal\nScheduled work only."
    )
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(summary_text),
    ) as llm:
        out = c.compress(msgs)
    if mode == "catalog":
        llm.assert_not_called()
    marked = [m for m in out if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert marked
    text = str(marked[0].get("content") or "")
    assert _NO_USER_TASK_SENTINEL in text
    assert marked[0][COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False
    if mode == "catalog":
        assert "User asked:" not in text
