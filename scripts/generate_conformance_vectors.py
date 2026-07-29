#!/usr/bin/env python3
"""Conformance-vector generator — the native adapters as executable spec.

Renders a shared corpus (markdown grid + scar tissue + adversarial agent
output) through the NATIVE platform renderers and dumps input→output JSON
vectors, stamped with the oracle commit. The gateway-gateway connector
commits these under conformance/vectors/ and its vitest runner asserts the
CONNECTOR's constructed REST payloads against them — so cross-repo renderer
drift breaks a test instead of a user's formatting.

Oracles (all imported, never reimplemented):
  telegram  plugins.platforms.telegram.adapter.TelegramAdapter.format_message
            (standard markdown → Telegram MarkdownV2)
  slack     plugins.platforms.slack.adapter.SlackAdapter.format_message
            (standard markdown → Slack mrkdwn)
  whatsapp  gateway.platforms.whatsapp_common.WhatsAppBehaviorMixin
            .format_message (standard markdown → WhatsApp formatting)
  discord   plugins.platforms.discord.adapter.DiscordAdapter.format_message
            (GFM tables → bullet groups; otherwise identity)

Expect semantics (consumed by the gg runner):
  parity    connector render must BYTE-EQUAL native_output
            (Slack / WhatsApp — same-dialect ports; most Discord).
  semantic  connector renders a DIFFERENT representation on purpose
            (Telegram: connector sends HTML, native sends MarkdownV2);
            the runner asserts plain-text content equivalence instead.
  divergent documented no-parity (e.g. Discord tables: native converts to
            bullets, connector passes raw markdown through). The runner
            asserts the DOCUMENTED connector behavior named in `note`.

Run:  python scripts/generate_conformance_vectors.py [--out DIR]
Determinism is covered by tests/conformance/test_vector_generator.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GENERATOR_VERSION = 1

# ── corpus ───────────────────────────────────────────────────────────────
# Every entry: (id, category, input). Categories: grid | scar | adversarial.
# Ids are STABLE API — the runner and divergence notes key on them.

GRID: List[tuple] = [
    ("plain-text", "Just a plain sentence."),
    ("bold", "This is **bold** text."),
    ("italic", "This is *italic* text."),
    ("bold-italic", "Mix of **bold** and *italic* in one line."),
    ("strikethrough", "This is ~~struck~~ text."),
    ("inline-code", "Run `pip install hermes` to start."),
    ("fenced-code", "```\nprint('hello')\n```"),
    ("fenced-code-lang", "```python\ndef f(x):\n    return x * 2\n```"),
    ("link", "See [the docs](https://example.com/docs) for more."),
    ("link-parens-url", "See [spec](https://example.com/a_(b)) here."),
    ("header-h1", "# Big Title\nBody follows."),
    ("header-h2", "## Section\nBody follows."),
    ("header-h3", "### Sub-section\nBody follows."),
    ("ul-list", "- first\n- second\n- third"),
    ("ol-list", "1. first\n2. second\n3. third"),
    ("nested-list", "- outer\n  - inner one\n  - inner two\n- outer two"),
    ("blockquote", "> quoted wisdom\nregular line"),
    ("hrule", "above\n\n---\n\nbelow"),
    (
        "table-simple",
        "| name | value |\n|------|-------|\n| a    | 1     |\n| b    | 2     |",
    ),
    ("emoji", "Done ✅ with 🎉 emoji 👀 test."),
    ("cjk", "中文测试：**粗体** 和 `代码` 混排。"),
    ("bare-url", "Visit https://example.com/path?q=1&r=2 today."),
    (
        "mixed-document",
        "## Report\n\nStatus: **green**. Details in `runbook.md`.\n\n"
        "- item *one*\n- item **two**\n\n```sh\nmake deploy\n```\n\n"
        "See [dashboard](https://grafana.example.com/d/x).",
    ),
]

SCAR: List[tuple] = [
    # MarkdownV2 reserved characters in prose — the classic Telegram 400.
    ("mdv2-reserved-chars", "Price is 3.50 (was 4.00) — save ~12%! #deal +tax = win."),
    ("mdv2-underscores", "snake_case_name and file_name.py in prose."),
    ("mdv2-brackets", "Array[0] and dict{key} and (parens) live here."),
    # Slack: **bold** must become *bold*; [t](u) must become <u|t>.
    ("slack-bold-conversion", "**important** word"),
    ("slack-link-conversion", "[click here](https://example.com)"),
    # Slack broadcast-mention escape (model output must not ping @everyone).
    ("slack-broadcast-mention", "Hey <!everyone> and <!channel> and <!here>!"),
    # Fence language tag handling (Slack renders the tag literally).
    ("fence-lang-tag-slack", "```text\nliteral first line issue\n```"),
    # Backslashes inside code must survive doubling rules.
    ("backslash-in-code", "`C:\\Users\\ben\\file.txt` and ```\npath = \"a\\\\b\"\n```"),
    ("backtick-in-fence", "```\nuse `inline` inside fence\n```"),
    # Headers containing bold markers (native strips redundant **).
    ("header-with-bold", "## The **Real** Deal"),
    # Table with CJK cells (display-width alignment scar in Slack).
    (
        "table-cjk",
        "| 名前 | 値 |\n|------|----|\n| 中文 | 42 |\n| b    | 2  |",
    ),
    # Link display text that itself needs escaping.
    ("link-display-escapes", "[v2.0 (beta)](https://example.com/v2)"),
]

ADVERSARIAL: List[tuple] = [
    ("media-tag", "Here you go\nMEDIA:/tmp/output.png\ndone"),
    ("unclosed-fence", "```python\nprint('never closed')"),
    ("pathological-nesting", "**bold *italic ~~struck `code` struck~~ italic* bold**"),
    ("placeholder-injection", "sneaky \x00PH0\x00 token and \x00SL1\x00 too"),
    ("triple-markers", "***what is this*** and ____that____"),
    ("empty-string", ""),
    ("whitespace-only", "   \n\t\n   "),
    ("long-line", "word " * 500),
    ("many-fences", "```\na\n```\nmid\n```\nb\n```\nend ```inline``` tail"),
]


def corpus() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for cid, text in GRID:
        rows.append({"id": cid, "category": "grid", "input": text})
    for cid, text in SCAR:
        rows.append({"id": cid, "category": "scar", "input": text})
    for cid, text in ADVERSARIAL:
        rows.append({"id": cid, "category": "adversarial", "input": text})
    return rows


# ── oracles ──────────────────────────────────────────────────────────────


def _oracles() -> Dict[str, Callable[[str], str]]:
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.slack.adapter import SlackAdapter
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

    wa = object.__new__(WhatsAppBehaviorMixin)  # format_message needs no __init__

    return {
        # These format_message implementations are self-free (asserted by
        # tests/conformance/test_vector_generator.py) — invoked unbound.
        "telegram": lambda s: TelegramAdapter.format_message(None, s),  # type: ignore[arg-type]
        "slack": lambda s: SlackAdapter.format_message(None, s),  # type: ignore[arg-type]
        "discord": lambda s: DiscordAdapter.format_message(None, s),  # type: ignore[arg-type]
        "whatsapp": wa.format_message,
    }


# Per-platform expect overrides (default: parity, except telegram=semantic).
# Keyed by vector id; value = (expect, note).
_EXPECT_OVERRIDES: Dict[str, Dict[str, tuple]] = {
    "telegram": {
        # Native wraps pipe tables into row groups; the connector's HTML lane
        # renders tables as <pre>. Same content, structurally different enough
        # that plain-text comparison is noise — documented divergence.
        "table-simple": ("divergent", "native wraps tables into row groups; connector renders <pre> — content preserved, layout differs"),
        "table-cjk": ("divergent", "same as table-simple (CJK width alignment is native-only)"),
        "unclosed-fence": ("divergent", "unterminated fence: native escapes as prose, connector HTML may close the block — degraded either way, never a 400"),
        "placeholder-injection": ("divergent", "NUL placeholder tokens are renderer-internal; each side neutralizes its own pattern"),
        "whitespace-only": ("divergent", "native collapses to empty-ish prose, connector HTML preserves — cosmetic"),
    },
    "slack": {
        "placeholder-injection": ("divergent", "\\x00SL tokens are the native renderer's own placeholder alphabet; connector uses a different scheme"),
        "unclosed-fence": ("divergent", "unterminated fence handling differs; both degrade without dropping content"),
    },
    "whatsapp": {
        "placeholder-injection": ("divergent", "placeholder alphabets are renderer-internal"),
        "unclosed-fence": ("divergent", "unterminated fence handling differs; both degrade without dropping content"),
    },
    "discord": {
        "table-simple": ("divergent", "native converts GFM tables to bullet groups; connector passes raw markdown through (port deferred — parity report Phase 4/oracle section)"),
        "table-cjk": ("divergent", "same as table-simple"),
    },
}


def _expect_for(platform: str, vector_id: str) -> tuple:
    default = ("semantic", "") if platform == "telegram" else ("parity", "")
    return _EXPECT_OVERRIDES.get(platform, {}).get(vector_id, default)


def _oracle_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def generate(out_dir: Path) -> Dict[str, Any]:
    """Render the corpus through every oracle; write one JSON per platform."""
    oracles = _oracles()
    commit = _oracle_commit()
    rows = corpus()
    summary: Dict[str, Any] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for platform, render in sorted(oracles.items()):
        vectors = []
        for row in rows:
            expect, note = _expect_for(platform, row["id"])
            entry: Dict[str, Any] = {
                "id": row["id"],
                "category": row["category"],
                "expect": expect,
                "input": row["input"],
                "native_output": render(row["input"]),
            }
            if note:
                entry["note"] = note
            vectors.append(entry)
        doc = {
            "$comment": (
                "GENERATED — do not hand-edit. Regenerate with "
                "hermes-agent scripts/generate_conformance_vectors.py; the "
                "native renderers are the oracle (executable spec)."
            ),
            "oracle": {
                "repo": "NousResearch/hermes-agent",
                "commit": commit,
                "generator": "scripts/generate_conformance_vectors.py",
                "generator_version": GENERATOR_VERSION,
            },
            "platform": platform,
            "vectors": vectors,
        }
        path = out_dir / f"{platform}.json"
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary[platform] = len(vectors)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "tests" / "conformance" / "vectors"),
        help="Output directory for <platform>.json vector files",
    )
    args = parser.parse_args()
    summary = generate(Path(args.out))
    for platform, count in sorted(summary.items()):
        print(f"{platform}: {count} vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
