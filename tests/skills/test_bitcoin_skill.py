"""Tests for optional-skills/blockchain/bitcoin/scripts/bitcoin_client.py."""

import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "bitcoin"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import bitcoin_client as btc


class TestFmtHelpers:
    def test_fmt_btc(self):
        assert btc.fmt_btc(100_000_000) == "1.00000000 BTC"
        assert btc.fmt_btc(0) == "0.00000000 BTC"

    def test_fmt_fiat_usd(self):
        assert btc.fmt_fiat(100_000_000, 50000, "usd") == "$50,000.00"

    def test_fmt_fiat_eur(self):
        assert btc.fmt_fiat(100_000_000, 45000, "eur") == "€45,000.00"

    def test_fmt_fiat_none(self):
        assert btc.fmt_fiat(100_000_000, None, "usd") is None

    def test_format_hashrate(self):
        assert btc.format_hashrate(1e18) == "1.000 EH/s"
        assert btc.format_hashrate(1e15) == "1.000 PH/s"
        assert btc.format_hashrate(0) is None
        assert btc.format_hashrate(None) is None


class TestBlockSubsidy:
    def test_genesis_subsidy(self):
        assert btc.block_subsidy(0) == 50 * 100_000_000

    def test_first_halving(self):
        assert btc.block_subsidy(210_000) == 25 * 100_000_000

    def test_second_halving(self):
        assert btc.block_subsidy(420_000) == 12.5 * 100_000_000

    def test_post_64_halvings(self):
        assert btc.block_subsidy(64 * 210_000) == 0


class TestFeeRateForTx:
    def test_fee_rate_from_vsize(self):
        tx = {"fee": 1000, "vsize": 250}
        assert btc.fee_rate_for_tx(tx) == 4.0

    def test_fee_rate_from_weight(self):
        tx = {"fee": 1000, "weight": 1000}
        assert btc.fee_rate_for_tx(tx) == 4.0

    def test_fee_rate_missing_size(self):
        tx = {"fee": 1000}
        assert btc.fee_rate_for_tx(tx) is None


class TestCmdStatsDifficultyAdjustmentOrdering:
    """mempool.space returns difficulty-adjustment rows newest-first, but the
    implementation must select the row with the maximum timestamp regardless of
    list ordering.
    """

    @mock.patch("bitcoin_client.fetch_text")
    @mock.patch("bitcoin_client.fetch_json")
    @mock.patch("bitcoin_client.get_btc_price")
    def test_selects_max_timestamp_not_last_element(
        self, mock_price, mock_fetch_json, mock_fetch_text, capsys
    ):
        mock_fetch_text.side_effect = ["850000", "0000000000000000000000000000000000000000000000000000000000000001"]
        mock_price.return_value = {"usd": 50000.0}

        # Rows are [timestamp, height, difficulty, estimated_change_percent]
        # Newest-first ordering: the first row has the largest timestamp.
        diff_adj = [
            [1752009600, 850000, 85000000000000.0, 2.5],
            [1751923200, 848832, 83000000000000.0, -1.2],
            [1751836800, 847664, 82000000000000.0, 0.5],
        ]

        mock_fetch_json.side_effect = [
            {"timestamp": 1752009600, "difficulty": 85000000000000.0},
            {"currentHashrate": 1e18, "currentDifficulty": 85000000000000.0, "hashrates": []},
            diff_adj,
        ]

        class Args:
            currency = "usd"
            no_fiat = False

        btc.cmd_stats(Args())
        captured = capsys.readouterr()
        result = json.loads(captured.out)

        # If the code used diff_adj[-1], it would pick -1.2% instead of 2.5%.
        assert result["next_retarget"]["estimated_change_percent"] == 2.5
        assert result["next_retarget"]["last_adjustment_height"] == 850000

    @mock.patch("bitcoin_client.fetch_text")
    @mock.patch("bitcoin_client.fetch_json")
    @mock.patch("bitcoin_client.get_btc_price")
    def test_selects_max_timestamp_when_oldest_first(
        self, mock_price, mock_fetch_json, mock_fetch_text, capsys
    ):
        mock_fetch_text.side_effect = ["850000", "0000000000000000000000000000000000000000000000000000000000000001"]
        mock_price.return_value = {"usd": 50000.0}

        # Oldest-first ordering: the last row has the largest timestamp.
        diff_adj = [
            [1751836800, 847664, 82000000000000.0, 0.5],
            [1751923200, 848832, 83000000000000.0, -1.2],
            [1752009600, 850000, 85000000000000.0, 2.5],
        ]

        mock_fetch_json.side_effect = [
            {"timestamp": 1752009600, "difficulty": 85000000000000.0},
            {"currentHashrate": 1e18, "currentDifficulty": 85000000000000.0, "hashrates": []},
            diff_adj,
        ]

        class Args:
            currency = "usd"
            no_fiat = False

        btc.cmd_stats(Args())
        captured = capsys.readouterr()
        result = json.loads(captured.out)

        assert result["next_retarget"]["estimated_change_percent"] == 2.5
        assert result["next_retarget"]["last_adjustment_height"] == 850000


class TestCmdStatsOutput:
    @mock.patch("bitcoin_client.fetch_text")
    @mock.patch("bitcoin_client.fetch_json")
    @mock.patch("bitcoin_client.get_btc_price")
    def test_next_retarget_fields(self, mock_price, mock_fetch_json, mock_fetch_text, capsys):
        mock_fetch_text.side_effect = ["850000", "0000000000000000000000000000000000000000000000000000000000000001"]
        mock_price.return_value = {"usd": 50000.0}

        diff_adj = [
            [1752009600, 850000, 85000000000000.0, 2.5],
            [1751923200, 848832, 83000000000000.0, -1.2],
        ]

        mock_fetch_json.side_effect = [
            {"timestamp": 1752009600, "difficulty": 85000000000000.0},
            {"currentHashrate": 1e18, "currentDifficulty": 85000000000000.0, "hashrates": []},
            diff_adj,
        ]

        class Args:
            currency = "usd"
            no_fiat = False

        btc.cmd_stats(Args())
        captured = capsys.readouterr()
        result = json.loads(captured.out)

        assert result["tip_height"] == 850000
        assert result["next_retarget"]["estimated_change_percent"] == 2.5
        assert result["next_retarget"]["last_adjustment_height"] == 850000
        assert result["next_retarget"]["current_difficulty"] == 85000000000000.0


def _cm(body_bytes: bytes):
    """Return a context-manager-compatible mock response."""
    resp = mock.MagicMock()
    resp.read.return_value = body_bytes
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestFetchJsonWithFallback:
    """Tests for automatic fallback from mempool.space to blockstream.info."""

    @mock.patch("bitcoin_client.urllib.request.urlopen")
    def test_primary_success_no_fallback(self, mock_urlopen):
        mock_urlopen.return_value = _cm(json.dumps({"ok": True}).encode())

        result = btc.fetch_json_with_fallback("/test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 1
        first_url = mock_urlopen.call_args_list[0][0][0].full_url
        assert first_url.startswith("https://mempool.space/api/test")

    @mock.patch("bitcoin_client.urllib.request.urlopen")
    def test_fallback_on_4xx_breaks_to_next_candidate(self, mock_urlopen):
        primary_error = mock.Mock()
        primary_error.code = 404
        primary_error.read.return_value = b"not found"

        side_effects = [
            urllib.error.HTTPError("url", 404, "not found", {}, primary_error),
            _cm(json.dumps({"fallback": True}).encode()),
        ]

        def urlopen_side_effect(req, **kwargs):
            item = side_effects.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        mock_urlopen.side_effect = urlopen_side_effect

        result = btc.fetch_json_with_fallback("/test")
        assert result == {"fallback": True}
        urls = [c[0][0].full_url for c in mock_urlopen.call_args_list]
        assert any(u.startswith("https://blockstream.info") for u in urls)

    @mock.patch("bitcoin_client.urllib.request.urlopen")
    def test_429_retries_on_primary_then_fallback(self, mock_urlopen):
        primary_error = mock.Mock()
        primary_error.code = 429
        primary_error.read.return_value = b"rate limited"

        # retries=2 -> 3 attempts on the same URL, then fallback candidate.
        side_effects = [
            urllib.error.HTTPError("url", 429, "rate limited", {}, primary_error),
            urllib.error.HTTPError("url", 429, "rate limited", {}, primary_error),
            urllib.error.HTTPError("url", 429, "rate limited", {}, primary_error),
            _cm(json.dumps({"fallback_after_429": True}).encode()),
        ]

        def urlopen_side_effect(req, **kwargs):
            item = side_effects.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        mock_urlopen.side_effect = urlopen_side_effect

        result = btc.fetch_json_with_fallback("/test")
        assert result == {"fallback_after_429": True}
        urls = [c[0][0].full_url for c in mock_urlopen.call_args_list]
        mempool_calls = [u for u in urls if u.startswith("https://mempool.space")]
        fallback_calls = [u for u in urls if u.startswith("https://blockstream.info")]
        assert len(mempool_calls) == 3
        assert len(fallback_calls) == 1

    @mock.patch("bitcoin_client.urllib.request.urlopen")
    def test_all_candidates_fail_raises(self, mock_urlopen):
        err = mock.Mock()
        err.code = 500
        err.read.return_value = b"error"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "error", {}, err
        )

        with pytest.raises(RuntimeError) as exc_info:
            btc.fetch_json_with_fallback("/test")
        assert "HTTP 500" in str(exc_info.value)


    def test_skill_description_length(self):
        skill_md = SCRIPTS_DIR.parent / "SKILL.md"
        text = skill_md.read_text()
        import re
        m = re.search(r'^description:\s*"([^"]+)"', text, re.MULTILINE)
        assert m, "description frontmatter not found"
        desc = m.group(1)
        assert len(desc) <= 60, f"description too long ({len(desc)} chars): {desc}"
        assert desc.endswith("."), f"description must end with a period: {desc}"
