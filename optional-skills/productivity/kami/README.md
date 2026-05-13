# Kami · 紙

**The paper your deliverables land on.**

Kami is a Hermes Agent skill for typesetting professional documents: resumes, one-pagers, white papers, letters, portfolios, slide decks, equity reports, and invoices. One design language across eight document types — warm parchment canvas, ink-blue accent, serif-led hierarchy, tight editorial rhythm.

Part of `Kaku · Waza · Kami` — Kaku writes code, Waza drills habits, **Kami delivers documents.**

## Document Types

| Type | CN Template | EN Template |
|------|------------|-------------|
| One-Pager | `one-pager.html` | `one-pager-en.html` |
| Long Doc | `long-doc.html` | `long-doc-en.html` |
| Letter | `letter.html` | `letter-en.html` |
| Portfolio | `portfolio.html` | `portfolio-en.html` |
| Resume | `resume.html` | `resume-en.html` |
| Slides | `slides-weasy.html` | `slides-weasy-en.html` |
| Equity Report | `equity-report.html` | `equity-report-en.html` |
| Changelog | `changelog.html` | `changelog-en.html` |
| Invoice | `invoice-fr.html` | — |

## Diagrams

14 diagram primitives (architecture, flowchart, quadrant, bar/line/donut charts, candlestick, waterfall, state machine, timeline, swimlane, tree, layer stack, venn) — all SVG-based, embeddable in any document.

## Quick Start

```bash
# Build and verify all templates
python3 scripts/build.py --verify

# macOS Apple Silicon — use the convenience wrapper
bash scripts/kami-build.sh --verify resume-en

# Ensure fonts are present
bash scripts/ensure-fonts.sh
```

## Fonts

- **Chinese:** TsangerJinKai02 (commercial) — excluded from repo, fetched via `ensure-fonts.sh`
- **English:** Charter (system-bundled on macOS/iOS)
- **Japanese:** YuMincho (best-effort, system fallback chain)

## Structure

```
kami/
├── SKILL.md              # Main skill instructions
├── CHEATSHEET.md         # Quick reference
├── .gitignore
├── references/           # Design, writing, production specs
│   ├── design.md
│   ├── writing.md
│   ├── production.md
│   ├── diagrams.md
│   ├── anti-patterns.md
│   └── brand-profile.md
├── assets/
│   ├── templates/        # HTML templates (CN + EN)
│   ├── diagrams/         # SVG diagram templates
│   ├── fonts/            # Font files (JetBrains Mono, TsangerJinKai)
│   └── examples/         # Built PDF/PPTX examples
└── scripts/              # Build, verify, package utilities
    ├── build.py
    ├── shared.py
    ├── stabilize.py
    ├── package-skill.sh
    ├── kami-build.sh
    └── ensure-fonts.sh
```

## Contributing

Kami is designed to be language-agnostic. The skill matches the user's language and routes to the appropriate template (CN/EN/CJK). To contribute:

1. **New document types:** Add templates to `assets/templates/`, update `SKILL.md` routing table
2. **New diagrams:** Add SVG templates to `assets/diagrams/`, update `references/diagrams.md`
3. **Bug fixes:** Patch templates or scripts, run `build.py --verify` to confirm
4. **Translations:** New language templates follow the `-lang.html` convention

## License

MIT — see [LICENSE](LICENSE)
