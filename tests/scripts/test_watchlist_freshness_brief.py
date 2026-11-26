import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "watchlist_freshness_brief.py"
    spec = importlib.util.spec_from_file_location("watchlist_freshness_brief", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_watchlist_entries_ignores_instructional_sections(tmp_path):
    mod = load_module()
    watchlist = tmp_path / "watchlist.md"
    watchlist.write_text(
        """# Investor Social Watchlist

## Threads accounts
- james93.lin — https://www.threads.com/@james93.lin
- market.sherlock - https://www.threads.com/@market.sherlock

## Intended output format for future monitoring
- New notable posts since last check
- Confidence / evidence label:
  - Fact: directly observed from post

## Safety / permission notes
- Do not like, reply, repost, follow, DM, or otherwise interact with real people.
""",
        encoding="utf-8",
    )

    entries = mod.parse_watchlist(watchlist)

    assert [(entry.label, entry.url, entry.section) for entry in entries] == [
        ("james93.lin", "https://www.threads.com/@james93.lin", "Threads accounts"),
        ("market.sherlock", "https://www.threads.com/@market.sherlock", "Threads accounts"),
    ]


def test_classify_entries_overdue_due_soon_and_missing_state(tmp_path):
    mod = load_module()
    entries = [
        mod.WatchlistEntry(label="old", url="https://example.com/old", section="X"),
        mod.WatchlistEntry(label="soon", url="https://example.com/soon", section="X"),
        mod.WatchlistEntry(label="fresh", url="https://example.com/fresh", section="X"),
        mod.WatchlistEntry(label="missing", url="https://example.com/missing", section="X"),
    ]
    state = {
        "https://example.com/old": {"last_checked": "2026-06-01", "cadence_days": 14, "priority": "high"},
        "https://example.com/soon": {"last_checked": "2026-06-08", "cadence_days": 14},
        "https://example.com/fresh": {"last_checked": "2026-06-15", "cadence_days": 14},
    }

    brief = mod.build_brief(entries, state, today=mod.parse_date("2026-06-19"), soon_days=3)

    assert [item.entry.label for item in brief.overdue] == ["old"]
    assert brief.overdue[0].days_overdue_as_of(mod.parse_date("2026-06-19")) == 4
    assert [item.entry.label for item in brief.due_soon] == ["soon"]
    assert brief.due_soon[0].days_until_due_as_of(mod.parse_date("2026-06-19")) == 3
    assert [entry.label for entry in brief.missing_state] == ["missing"]


def test_render_markdown_returns_silent_when_nothing_to_report():
    mod = load_module()
    entries = [mod.WatchlistEntry(label="fresh", url="https://example.com/fresh", section="X")]
    state = {"https://example.com/fresh": {"last_checked": "2026-06-18", "cadence_days": 14}}

    brief = mod.build_brief(entries, state, today=mod.parse_date("2026-06-19"), soon_days=3)

    assert mod.render_markdown(brief) == "[SILENT]"


def test_cli_json_output(tmp_path, capsys):
    mod = load_module()
    watchlist = tmp_path / "watchlist.md"
    state_file = tmp_path / "state.json"
    watchlist.write_text("- old — https://example.com/old\n", encoding="utf-8")
    state_file.write_text(
        json.dumps({"items": {"https://example.com/old": {"last_checked": "2026-06-01", "cadence_days": 14}}}),
        encoding="utf-8",
    )

    exit_code = mod.main([
        "--watchlist",
        str(watchlist),
        "--state",
        str(state_file),
        "--today",
        "2026-06-19",
        "--json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["overdue"] == 1
    assert payload["overdue"][0]["label"] == "old"
