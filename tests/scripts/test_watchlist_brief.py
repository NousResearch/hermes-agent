import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "watchlist_brief.py"

spec = importlib.util.spec_from_file_location("watchlist_brief", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
watchlist_brief = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = watchlist_brief
spec.loader.exec_module(watchlist_brief)


def test_parse_watchlist_extracts_names_platforms_and_urls(tmp_path):
    watchlist = tmp_path / "watchlist.md"
    watchlist.write_text(
        """
# Investor Watchlist

## Crypto funds
- [high] Alice Example — X: https://x.com/alice; Substack: https://alice.substack.com
- Bob Builder: https://twitter.com/bob and https://github.com/bob

## Fintech operators
- Carol Chen — no public URL yet
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = watchlist_brief.parse_watchlist(watchlist)

    assert [entry.name for entry in entries] == ["Alice Example", "Bob Builder", "Carol Chen"]
    assert entries[0].section == "Crypto funds"
    assert entries[0].priority == "high"
    assert entries[0].platforms == ["substack", "x"]
    assert entries[1].platforms == ["github", "x"]
    assert entries[2].urls == []


def test_build_brief_groups_review_queue_and_missing_urls(tmp_path):
    watchlist = tmp_path / "watchlist.md"
    watchlist.write_text(
        """
# Watchlist
- [high] Alice Example — https://x.com/alice
- Bob Builder — no URL
""".strip()
        + "\n",
        encoding="utf-8",
    )

    brief = watchlist_brief.build_markdown_brief(watchlist_brief.parse_watchlist(watchlist))

    assert "# Watchlist Brief" in brief
    assert "## Review queue" in brief
    assert "Alice Example" in brief
    assert "## Missing URLs" in brief
    assert "Bob Builder" in brief


def test_parse_watchlist_skips_guidance_sections_and_nested_bullets(tmp_path):
    watchlist = tmp_path / "watchlist.md"
    watchlist.write_text(
        """
# Investor Watchlist

## Threads accounts
- james93.lin — https://www.threads.com/@james93.lin

## Intended output format for future monitoring
- New notable posts since last check
- Why it matters:
  - personal investment thesis

## Safety / permission notes
- Read-only monitoring only
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = watchlist_brief.parse_watchlist(watchlist)

    assert [entry.name for entry in entries] == ["james93.lin"]
    assert entries[0].platforms == ["threads"]


def test_cli_outputs_json(tmp_path):
    watchlist = tmp_path / "watchlist.md"
    watchlist.write_text("- Alice Example — https://x.com/alice\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(watchlist), "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["entries"][0]["name"] == "Alice Example"
    assert payload["entries"][0]["platforms"] == ["x"]
