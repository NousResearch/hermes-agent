"""Tests for the morning_brief_discord delivery script.

Each test is anchored to a specific AC from issue #7 (the in-gateway cron
job registration — old AC1/AC2/AC7 — was dropped: the launchd morning
chain in deploy/ is the real trigger path; per-job cron timezones remain
covered under AC8):

  AC3 — Discord channel ID from config.discord.morning_brief_channel_id (not hardcoded)
  AC4 — contract file paths from config.discord.morning_brief_contracts
  AC5 — no-agent mode: no interactive steps, no LLM required
  AC6 — Discord delivery failure → error logged, process exits non-zero
  AC8 — existing jobs are unaffected (regression gate), incl. per-job tz support
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with scripts/ and cron/ directories."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    for mod_name in ("hermes_constants", "cron.jobs", "cron.scheduler"):
        try:
            importlib.reload(importlib.import_module(mod_name))
        except Exception:
            pass

    return home


@pytest.fixture
def discord_config(hermes_env):
    """Write a config.yaml with morning brief discord settings."""
    config_path = hermes_env / "config.yaml"
    config_path.write_text(
        "discord:\n"
        "  morning_brief_channel_id: '1234567890'\n"
        "  morning_brief_contracts: []\n",
        encoding="utf-8",
    )
    return hermes_env


# ---------------------------------------------------------------------------
# AC3 — channel ID from config, not hardcoded
# ---------------------------------------------------------------------------

class TestChannelIdFromConfig:
    """AC3: Discord channel ID must come from config, never hardcoded."""

    def test_script_reads_channel_id_from_config(self, tmp_path, monkeypatch):
        """Script exits non-zero when channel ID is missing from config."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "scripts").mkdir()
        (home / "cron").mkdir()
        # Config WITHOUT channel ID
        (home / "config.yaml").write_text("discord: {}\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            env={**os.environ, "HERMES_HOME": str(home)},
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "script must exit non-zero when channel ID is missing"
        assert "morning_brief_channel_id" in result.stderr or "channel" in result.stderr.lower()

    def test_script_fails_fast_without_token(self, tmp_path, monkeypatch):
        """Script exits non-zero when DISCORD_BOT_TOKEN is unset."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '111'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items() if "TOKEN" not in k and "SECRET" not in k}
        env["HERMES_HOME"] = str(home)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "script must exit non-zero when token is missing"
        assert "DISCORD_BOT_TOKEN" in result.stderr or "token" in result.stderr.lower()


# ---------------------------------------------------------------------------
# AC4 — contract file paths from config
#
# The delivery script now reuses plugins.life_ops.scripts.morning_brief_composer, which
# degrades gracefully on missing/stale contracts instead of failing. Config
# override only takes effect when discord.morning_brief_contracts lists
# EXACTLY three paths that map unambiguously to journal/perfcoach/commander
# by filename (see _resolve_contract_paths in morning_brief_discord.py).
# Anything else (0, 1, 2, or unmappable paths) falls back to the composer's
# own defaults/env vars.
# ---------------------------------------------------------------------------

def _today_iso() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Bangkok")).date().isoformat()


class TestContractFilePaths:
    """AC4: contract file paths are configurable via config.discord.morning_brief_contracts."""

    def test_three_paths_mapped_by_filename_are_used(self, tmp_path):
        """Exactly three configured paths, matched by filename substring
        ("journal"/"perfcoach"/"commander"), override the defaults."""
        home = tmp_path / ".hermes"
        home.mkdir()

        journal = tmp_path / "journal_brief.latest.json"
        journal.write_text(
            json.dumps({
                "for_date": _today_iso(),
                "reflection": {"markdown": "Daily goals: exercise, read, code."},
                "todos": [],
            }),
            encoding="utf-8",
        )
        perfcoach = tmp_path / "perfcoach_brief.latest.json"
        perfcoach.write_text(
            json.dumps({"for_date": _today_iso(), "advisories": ["Stay hydrated."]}),
            encoding="utf-8",
        )
        commander = tmp_path / "commander_report.latest.json"
        commander.write_text(
            json.dumps({
                "for_date": _today_iso(),
                "completed": ["shipped the config mapping"],
                "needs_review": [], "dead_letter": [], "cost": "$0.05",
            }),
            encoding="utf-8",
        )

        (home / "config.yaml").write_text(
            "discord:\n"
            "  morning_brief_channel_id: '999'\n"
            "  morning_brief_contracts:\n"
            f"    - {journal}\n"
            f"    - {perfcoach}\n"
            f"    - {commander}\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        # These env vars would otherwise take precedence in _resolve_contract_paths
        # over the (correctly-mapped) config-supplied paths' fallback defaults.
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"dry-run failed: {result.stderr}"
        assert "Daily goals" in result.stdout, "journal contract content must appear in brief output"
        assert "Stay hydrated." in result.stdout, "perfcoach contract content must appear in brief output"
        assert "shipped the config mapping" in result.stdout, "commander contract content must appear in brief output"

    def test_fewer_than_three_paths_falls_back_to_defaults(self, tmp_path):
        """A single configured path (not exactly three) is ignored; the
        composer's defaults are used instead, degrading gracefully."""
        home = tmp_path / ".hermes"
        home.mkdir()
        contract = tmp_path / "my_contract.txt"
        contract.write_text("Daily goals: exercise, read, code.", encoding="utf-8")

        (home / "config.yaml").write_text(
            f"discord:\n"
            f"  morning_brief_channel_id: '999'\n"
            f"  morning_brief_contracts:\n"
            f"    - {contract}\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"dry-run failed: {result.stderr}"
        assert "Daily goals" not in result.stdout, "single unmapped path must not override defaults"
        assert "⚠️ unavailable" in result.stdout, "defaults are missing, so sections should degrade"

    def test_three_paths_that_do_not_map_by_filename_falls_back_to_defaults(self, tmp_path):
        """Exactly three paths are configured, but none of their filenames
        identify journal/perfcoach/commander — falls back to defaults."""
        home = tmp_path / ".hermes"
        home.mkdir()
        a = tmp_path / "a.json"
        a.write_text(json.dumps({"for_date": _today_iso()}), encoding="utf-8")
        b = tmp_path / "b.json"
        b.write_text(json.dumps({"for_date": _today_iso()}), encoding="utf-8")
        c = tmp_path / "c.json"
        c.write_text(json.dumps({"for_date": _today_iso()}), encoding="utf-8")

        (home / "config.yaml").write_text(
            "discord:\n"
            "  morning_brief_channel_id: '999'\n"
            "  morning_brief_contracts:\n"
            f"    - {a}\n"
            f"    - {b}\n"
            f"    - {c}\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"dry-run failed: {result.stderr}"
        assert "falling back to defaults" in result.stderr.lower() or "unavailable" in result.stdout.lower()

    def test_missing_configured_contract_file_degrades_gracefully_exit_zero(self, tmp_path):
        """A configured (but nonexistent) contract file no longer causes a
        non-zero exit — the composer degrades that section instead."""
        home = tmp_path / ".hermes"
        home.mkdir()
        journal = tmp_path / "journal_brief.latest.json"  # never written
        perfcoach = tmp_path / "perfcoach_brief.latest.json"
        perfcoach.write_text(
            json.dumps({"for_date": _today_iso(), "advisories": ["Rest well."]}),
            encoding="utf-8",
        )
        commander = tmp_path / "commander_report.latest.json"  # never written

        (home / "config.yaml").write_text(
            "discord:\n"
            "  morning_brief_channel_id: '999'\n"
            "  morning_brief_contracts:\n"
            f"    - {journal}\n"
            f"    - {perfcoach}\n"
            f"    - {commander}\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"script must exit 0 even with missing contract files: {result.stderr}"
        assert "Rest well." in result.stdout, "the one present contract still renders"
        assert "⚠️ unavailable" in result.stdout, "the two missing contracts degrade instead of erroring"


class TestMissingContractsExitZero:
    """No discord.morning_brief_contracts configured at all, and none of the
    default contract paths exist: the script still exits 0 and prints a
    brief with every section degraded (former exit code 2 is retired)."""

    def test_no_contracts_key_at_all_exits_zero_degraded(self, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '999'\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"missing contracts must not cause a non-zero exit: {result.stderr}"
        # Only 3 of the 4 sections degrade to "⚠️ unavailable" — the todo
        # section (Section 2) now renders from plugins.life_ops.todo_store
        # regardless of journal-contract availability, so an empty store
        # renders "(no open todos)" rather than an unavailable marker.
        assert result.stdout.count("⚠️ unavailable") == 3, "the 3 non-todo sections should degrade"
        assert "(no open todos)" in result.stdout

    def test_empty_contracts_list_exits_zero_degraded(self, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n"
            "  morning_brief_channel_id: '999'\n"
            "  morning_brief_contracts: []\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        for var in ("JOURNAL_BRIEF_PATH", "PERFCOACH_BRIEF_PATH", "COMMANDER_REPORT_PATH"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # See test_no_contracts_key_at_all_exits_zero_degraded above — the
        # todo section no longer contributes an "⚠️ unavailable" marker.
        assert result.stdout.count("⚠️ unavailable") == 3
        assert "(no open todos)" in result.stdout


# ---------------------------------------------------------------------------
# Message chunking — _split_into_chunks splits on newline boundaries to stay
# under Discord's 2000-char hard cap (this module uses a 1900-char budget).
# ---------------------------------------------------------------------------

class TestChunkSplitting:
    """Unit tests for _split_into_chunks, imported directly (no subprocess)."""

    @staticmethod
    def _import_discord_module():
        import importlib
        return importlib.import_module("plugins.life_ops.scripts.morning_brief_discord")

    def test_short_text_is_a_single_chunk(self):
        mod = self._import_discord_module()
        text = "line one\nline two\nline three"
        chunks = mod._split_into_chunks(text, max_chars=1900)
        assert chunks == [text]

    def test_empty_text_produces_no_chunks(self):
        mod = self._import_discord_module()
        assert mod._split_into_chunks("", max_chars=1900) == []

    def test_all_chunks_respect_max_chars(self):
        mod = self._import_discord_module()
        text = "\n".join(f"line {i} " + ("x" * 50) for i in range(200))
        chunks = mod._split_into_chunks(text, max_chars=100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_splits_only_at_newline_boundaries_when_lines_fit(self):
        """No chunk boundary should fall mid-line when every line individually
        fits under max_chars — each chunk is a clean set of whole lines."""
        mod = self._import_discord_module()
        lines = [f"item-{i}" for i in range(50)]
        text = "\n".join(lines)
        chunks = mod._split_into_chunks(text, max_chars=30)
        # Reassembling the chunks (joined by \n, since a chunk boundary
        # replaces exactly one \n) must reproduce the original text exactly.
        assert "\n".join(chunks) == text
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_multi_line_text_exactly_at_boundary(self):
        """A line that lands exactly at max_chars should be included whole
        in the current chunk, not pushed to a new one prematurely."""
        mod = self._import_discord_module()
        max_chars = 20
        line_a = "a" * max_chars  # exactly fills a chunk on its own
        line_b = "b" * 5
        text = f"{line_a}\n{line_b}"
        chunks = mod._split_into_chunks(text, max_chars=max_chars)
        assert chunks[0] == line_a
        assert chunks[1] == line_b
        assert len(chunks) == 2

    def test_single_oversized_line_is_hard_split(self):
        """A single line longer than max_chars is hard-split at max_chars
        boundaries even though that means breaking mid-line."""
        mod = self._import_discord_module()
        max_chars = 10
        line = "x" * 25  # 25 > 10, needs 3 pieces: 10 + 10 + 5
        chunks = mod._split_into_chunks(line, max_chars=max_chars)
        assert chunks == ["x" * 10, "x" * 10, "x" * 5]
        assert "".join(chunks) == line

    def test_oversized_line_amid_normal_lines(self):
        """The oversized middle line is hard-split; its final (short)
        remainder piece may then be merged with the following short line if
        it still fits within max_chars — no content is lost either way."""
        mod = self._import_discord_module()
        max_chars = 10
        text = "short\n" + ("y" * 22) + "\nshort2"
        chunks = mod._split_into_chunks(text, max_chars=max_chars)
        assert all(len(c) <= max_chars for c in chunks)
        assert chunks[0] == "short"
        # Reconstruct: joining the hard-split middle pieces recovers the
        # original 22-char oversized line, and "short2" survives at the end.
        rejoined = "".join(chunks[1:]).replace("\n", "")
        assert rejoined == ("y" * 22) + "short2"
        assert chunks[-1].endswith("short2")

    def test_realistic_brief_stays_under_discord_hard_cap(self):
        """A long multi-section brief (bigger than DISCORD_MAX_CHARS) is
        split into pieces that each stay under Discord's real 2000-char
        hard cap, and no content is lost or duplicated."""
        mod = self._import_discord_module()
        section = "\n".join(f"- todo item number {i} with some detail text" for i in range(200))
        text = f"# Morning Brief\n\n{section}"
        assert len(text) > mod.DISCORD_MAX_CHARS
        chunks = mod._split_into_chunks(text, max_chars=mod.DISCORD_MAX_CHARS)
        assert len(chunks) > 1
        assert all(len(c) <= 2000 for c in chunks)
        assert "\n".join(chunks) == text


# ---------------------------------------------------------------------------
# AC5 — no-agent mode
# ---------------------------------------------------------------------------

class TestNoAgentMode:
    """AC5: job runs without an LLM agent — no interactive prompts."""

    def test_dry_run_completes_without_discord_api(self, tmp_path):
        """Dry-run env var makes the script print instead of posting."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '42'\n",
            encoding="utf-8",
        )

        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items()}
        env["HERMES_HOME"] = str(home)
        env["MORNING_BRIEF_DRY_RUN"] = "1"
        env.pop("DISCORD_BOT_TOKEN", None)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"dry-run failed: {result.stderr}"
        assert "DRY RUN" in result.stdout or "morning brief" in result.stdout.lower()


# ---------------------------------------------------------------------------
# AC6 — Discord delivery failure → error logged, exit non-zero
# ---------------------------------------------------------------------------

class TestDiscordDeliveryFailure:
    """AC6: Discord API failures are logged and cause non-zero exit."""

    def _run_script(self, home: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        script = Path(__file__).parent.parent.parent / "plugins" / "life_ops" / "scripts" / "morning_brief_discord.py"
        env = {k: v for k, v in os.environ.items() if "TOKEN" not in k and "SECRET" not in k}
        env["HERMES_HOME"] = str(home)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True, text=True,
        )

    def test_bad_token_exits_nonzero(self, tmp_path, monkeypatch):
        """Invalid bot token (401) → logged error, non-zero exit, no hang."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '111'\n",
            encoding="utf-8",
        )

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.text = "401: Unauthorized"

        with patch("httpx.post", return_value=mock_response):
            result = self._run_script(home, {"DISCORD_BOT_TOKEN": "bad-token"})

        assert result.returncode != 0
        assert "401" in result.stderr or "auth" in result.stderr.lower()

    def test_channel_not_found_exits_nonzero(self, tmp_path):
        """404 channel not found → logged error, non-zero exit."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '000000'\n",
            encoding="utf-8",
        )

        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.is_success = False
        mock_response.text = "404: Unknown Channel"

        with patch("httpx.post", return_value=mock_response):
            result = self._run_script(home, {"DISCORD_BOT_TOKEN": "valid-token"})

        assert result.returncode != 0
        assert "404" in result.stderr or "channel" in result.stderr.lower()

    def test_network_error_exits_nonzero(self, tmp_path):
        """Network error → logged error, non-zero exit, no hang."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "discord:\n  morning_brief_channel_id: '111'\n",
            encoding="utf-8",
        )

        import httpx

        with patch("httpx.post", side_effect=httpx.NetworkError("connection refused")):
            result = self._run_script(home, {"DISCORD_BOT_TOKEN": "valid-token"})

        assert result.returncode != 0


# ---------------------------------------------------------------------------
# AC8 — existing cron jobs unaffected
# ---------------------------------------------------------------------------

class TestExistingJobsUnaffected:
    """AC8: existing jobs still load and schedule correctly after this feature."""

    def test_existing_jobs_survive_import(self, discord_config, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(discord_config))
        for mod in ("hermes_constants", "cron.jobs"):
            importlib.reload(importlib.import_module(mod))

        from cron.jobs import create_job, load_jobs

        pre = create_job(prompt="check logs", schedule="every 1h", name="log_check")
        pre_id = pre["id"]

        # Simulate startup: reimport the module (as if scheduler just loaded)
        importlib.reload(importlib.import_module("cron.jobs"))

        jobs = load_jobs()
        assert any(j["id"] == pre_id for j in jobs), "pre-existing job must survive module reload"

    def test_compute_next_run_without_tz_unchanged(self):
        """compute_next_run for existing jobs (no tz field) must behave as before."""
        from cron.jobs import compute_next_run

        schedule = {"kind": "cron", "expr": "0 9 * * *"}
        result = compute_next_run(schedule)
        assert result is not None, "existing cron jobs must still compute next_run_at"

    def test_compute_next_run_with_bangkok_tz(self):
        """compute_next_run correctly uses Asia/Bangkok timezone when tz is set."""
        from zoneinfo import ZoneInfo
        from datetime import datetime

        from cron.jobs import compute_next_run

        schedule = {"kind": "cron", "expr": "0 6 * * *", "tz": "Asia/Bangkok"}
        result = compute_next_run(schedule)
        assert result is not None

        next_dt = datetime.fromisoformat(result)
        # Convert to Bangkok time and verify it's 06:00
        bkk = ZoneInfo("Asia/Bangkok")
        next_bkk = next_dt.astimezone(bkk)
        assert next_bkk.hour == 6, f"next run should be at 06:xx Bangkok, got {next_bkk}"
        assert next_bkk.minute == 0


# ---------------------------------------------------------------------------
# Fence-aware chunking — the morning brief's todo section is a fenced
# ```...``` code block; a naive per-line/per-char splitter (the old
# behavior) can land a chunk boundary inside the fence, leaving one Discord
# message with an unclosed ``` and the next with a stray closing ```. The
# fence-aware splitter (_extract_fenced_segments / _split_fenced_block) is
# supposed to prevent that: every produced chunk must have a balanced
# (even) count of ``` markers.
# ---------------------------------------------------------------------------

class TestFencedChunkSplitting:
    @staticmethod
    def _import_discord_module():
        import importlib
        return importlib.import_module("plugins.life_ops.scripts.morning_brief_discord")

    @staticmethod
    def _assert_all_chunks_fence_balanced(chunks):
        for i, chunk in enumerate(chunks):
            count = chunk.count("```")
            assert count % 2 == 0, (
                f"chunk {i} has an odd number of \\`\\`\\` markers ({count}): {chunk!r}"
            )

    def test_short_brief_with_small_fenced_block_is_one_chunk(self):
        """Regression: a normal-sized brief containing a small fenced todo
        block still produces exactly one chunk, unchanged from before the
        fence-aware splitter was added."""
        mod = self._import_discord_module()
        text = (
            "# Morning Brief\n\n"
            "## Section 2 — Todo List\n\n"
            "```\n"
            "To-do · 2 open · /done <key>\n\n"
            "  ! key-one   Fix the thing              07-14\n"
            "  · key-two   Review the other thing      07-13\n"
            "```\n"
        )
        chunks = mod._split_into_chunks(text, max_chars=1900)
        assert len(chunks) == 1
        assert chunks[0] == text
        self._assert_all_chunks_fence_balanced(chunks)

    def test_fence_never_broken_when_it_straddles_the_naive_split_point(self):
        """A fenced block that fits whole under max_chars, but whose
        start-to-end span crosses the max_chars boundary once preceding
        text is included, must stay intact as a single atomic segment — the
        whole fence moves to the next chunk rather than being torn (the bug
        a naive per-line splitter would have hit)."""
        mod = self._import_discord_module()
        max_chars = 500
        preamble = "\n".join(f"padding line {i}" for i in range(6))
        fence_lines = "\n".join(
            f"  · key-{i:03d}  todo body text here   07-{(i % 28) + 1:02d}" for i in range(10)
        )
        fenced_block = f"```\n{fence_lines}\n```"
        text = f"{preamble}\n{fenced_block}\nafter-fence-marker"

        assert len(fenced_block) < max_chars, "block itself must fit whole in one chunk"
        assert len(preamble) + 1 + len(fenced_block) > max_chars, "must straddle the naive boundary"

        chunks = mod._split_into_chunks(text, max_chars=max_chars)

        self._assert_all_chunks_fence_balanced(chunks)
        fence_chunk = next(c for c in chunks if "```" in c)
        assert fence_chunk.count("```") == 2
        for line in fence_lines.splitlines():
            assert line in fence_chunk
        # No oversized-block hard-splitting was needed here, so rejoining
        # every chunk on "\n" must reproduce the original text exactly.
        assert "\n".join(chunks) == text

    def test_oversized_fenced_block_is_hard_split_with_rewrapped_fences(self):
        """A fenced block bigger than max_chars on its own can't stay
        atomic — it's torn into pieces, but every piece is re-wrapped in its
        own ``` pair (documented behavior of _split_fenced_block) so it
        still renders monospace, and every resulting chunk stays
        fence-balanced."""
        mod = self._import_discord_module()
        max_chars = 100
        fence_lines = "\n".join(f"line-{i:03d} " + ("x" * 20) for i in range(30))
        fenced_block = f"```\n{fence_lines}\n```"
        assert len(fenced_block) > max_chars

        chunks = mod._split_into_chunks(fenced_block, max_chars=max_chars)

        assert len(chunks) > 1
        self._assert_all_chunks_fence_balanced(chunks)
        for chunk in chunks:
            assert len(chunk) <= max_chars

        # Stripping each chunk's re-wrapped outer fence and rejoining
        # reproduces every original inner line in order — no content lost
        # or duplicated despite the hard split.
        reassembled_lines = []
        for chunk in chunks:
            inner = chunk.split("\n")[1:-1]
            reassembled_lines.extend(inner)
        assert "\n".join(reassembled_lines) == fence_lines

    def test_full_synthetic_brief_with_oversized_todo_block_all_chunks_balanced(self):
        """End-to-end: a realistic brief shape (prose sections + a large
        fenced todo block, the exact shape render_todo_section() produces)
        chunked at the real DISCORD_MAX_CHARS budget — every produced chunk
        must be fence-balanced and every todo key must survive somewhere in
        the reassembled output."""
        mod = self._import_discord_module()
        todo_rows = "\n".join(
            f"  {'!' if i % 5 == 0 else '·'} key-{i:04d}          "
            f"a reasonably long todo description number {i}           07-{(i % 28) + 1:02d}"
            for i in range(120)
        )
        text = (
            "# Morning Brief\n\n"
            "## Section 1 — Journal Reflection\n\n"
            + "\n".join(f"reflection line {i}" for i in range(40))
            + "\n\n## Section 2 — Todo List\n\n"
            + f"```\nTo-do · 120 open · /done <key>\n\n{todo_rows}\n```\n\n"
            "## Section 3 — Training\n\n- Easy 30-min run.\n\n"
            "## Section 4 — Overnight Dev Report\n\n**Completed:** (none)\n"
        )
        assert len(text) > mod.DISCORD_MAX_CHARS

        chunks = mod._split_into_chunks(text, max_chars=mod.DISCORD_MAX_CHARS)

        assert len(chunks) > 1
        self._assert_all_chunks_fence_balanced(chunks)
        for chunk in chunks:
            assert len(chunk) <= mod.DISCORD_MAX_CHARS
        joined = "\n".join(chunks)
        for i in range(120):
            assert f"key-{i:04d}" in joined

    def test_unterminated_fence_falls_back_to_per_line_segments(self):
        """A fence with no closing ``` must not swallow the rest of the text
        as one giant atomic block — it degrades to per-line splitting like
        the surrounding text (the documented fallback in
        _extract_fenced_segments)."""
        mod = self._import_discord_module()
        text = "before\n```\nunterminated line one\nunterminated line two\nafter-should-still-split"
        segments = mod._extract_fenced_segments(text)
        assert segments == [
            "before",
            "```",
            "unterminated line one",
            "unterminated line two",
            "after-should-still-split",
        ]

    def test_two_small_fences_each_stay_atomic_and_balanced(self):
        """Two separate small fenced blocks, neither individually oversized,
        both end up fence-balanced across however many chunks they land in."""
        mod = self._import_discord_module()
        max_chars = 60
        block_a = "```\nrow-a1\nrow-a2\n```"
        block_b = "```\nrow-b1\nrow-b2\n```"
        text = f"intro\n{block_a}\nmiddle\n{block_b}\noutro"

        chunks = mod._split_into_chunks(text, max_chars=max_chars)

        self._assert_all_chunks_fence_balanced(chunks)
        joined = "\n".join(chunks)
        for marker in ("row-a1", "row-a2", "row-b1", "row-b2"):
            assert marker in joined
