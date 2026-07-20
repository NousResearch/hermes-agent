"""Model Desk deep tests — spend ledger, parity probe, spec decode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.model_desk.parity_matrix import parity_matrix
from agent.model_desk.spec_decode import speculative_decode_advice
from agent.model_desk.spend_ledger import spend_ledger_snapshot


class TestSpendLedger:
    def test_snapshot_never_raises(self):
        result = spend_ledger_snapshot()
        assert result["ok"] is True
        assert "layers" in result
        assert isinstance(result["layers"], list)

    def test_session_db_layer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()

        fake_db = MagicMock()
        fake_db.list_sessions_rich.return_value = [
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_tokens": 40,
                "cache_write_tokens": 10,
                "reasoning_tokens": 5,
            }
        ]
        with patch("hermes_state.SessionDB", return_value=fake_db):
            with patch(
                "agent.model_desk.spend_ledger._credits_layer",
                return_value={"ok": False, "source": "credits_view", "error": "skip"},
            ):
                with patch(
                    "agent.model_desk.spend_ledger._account_usage_layer",
                    return_value={"ok": False, "source": "account_usage", "error": "skip"},
                ):
                    result = spend_ledger_snapshot()
        assert result["ok"] is True
        assert result.get("skipped") is False
        assert result["totals"]["input_tokens"] == 100
        assert result["totals"]["cache_read_tokens"] == 40

    def test_skipped_when_empty(self):
        with patch(
            "agent.model_desk.spend_ledger._session_db_spend", return_value=None
        ):
            with patch(
                "agent.model_desk.spend_ledger._credits_layer",
                return_value={"ok": True, "source": "credits_view", "empty": True},
            ):
                with patch(
                    "agent.model_desk.spend_ledger._account_usage_layer",
                    return_value={"ok": True, "source": "account_usage", "empty": True},
                ):
                    result = spend_ledger_snapshot()
        assert result["skipped"] is True


class TestParityMatrix:
    def test_full_matrix(self):
        result = parity_matrix()
        assert result["ok"] is True
        assert "anthropic" in result["matrix"]
        assert result["matrix"]["anthropic"]["tools"] is True

    def test_provider_row(self):
        result = parity_matrix("ollama")
        assert result["provider"] == "ollama"
        assert "stream" in result["parity"]

    def test_probe_unreachable(self):
        result = parity_matrix(
            "ollama", probe=True, base_url="http://127.0.0.1:1"
        )
        assert result["ok"] is True
        assert "probe" in result
        assert result["probe"]["reachable"] is False


class TestSpecDecode:
    def test_advice_structure(self):
        result = speculative_decode_advice("qwen2.5-32b")
        assert result["ok"] is True
        assert result["main_model"] == "qwen2.5-32b"
        assert "suggestion" in result
        assert "backends_detected" in result

    def test_detects_when_binary_present(self, monkeypatch):
        monkeypatch.setattr(
            "agent.model_desk.spec_decode.shutil.which",
            lambda name: "/usr/bin/llama-server" if name == "llama-server" else None,
        )
        result = speculative_decode_advice("main")
        assert result["live_detection"] is True
        assert any(b["backend"] == "llama.cpp" for b in result["backends_detected"])
        assert "draft" in result["suggestion"]
