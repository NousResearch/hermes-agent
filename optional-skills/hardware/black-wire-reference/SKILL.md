---
name: black-wire-reference
description: "Find board pinouts and GPIO maps in the Black Wire guide."
version: 0.1.0
author: Richard (rahlquist), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Hardware, Pinout, GPIO, Electronics, Embedded, Reference]
    related_skills: [grounded-citations]
---

# Black Wire Reference Skill

Look up physical pinouts, GPIO maps, schematics, wiring guides and power profiles for
dev boards and maker modules in the Black Wire Maker's Technical Reference Guide
(BWM). Covers the live catalog, the collection repository and the offline library.
It is a source index, not an electrical authority: "Reviewed source" means categorized
against a cited source, never bench-verified, and no listing is a complete-device
approval.

## When to Use

- "Which pin is GPIO5 on <board>?", "find the pinout for <board>", "what does pin 7 do?"
- "Does <board> have a schematic / datasheet / board page?"
- Wiring and protocol questions covered by the 15 BWM guides (RJ45/T568, telephone,
  UART, RS-232, RS-485, CAN, I2C, SPI, SWD, JTAG, RFID/NFC).
- Supply and current questions for one of the 13 power profiles.
- Deciding whether a board's documentation exists before wiring it up.

Don't use for: live measurements, purchasing decisions, or as a substitute for the
manufacturer datasheet. Chip-package diagrams are labeled separately and are not
physical board pinouts.

## Prerequisites

- Python 3.10+ (standard library only).
- Network access to the live catalog (`library/catalog.json`, ~16 MB), or a local
  clone/extracted copy of `valleytechsolutions/black-wire-pinouts` (~1.8 GB).
- No API key. The script caches the downloaded catalog under the system temp dir.

## How to Run

```bash
python3 scripts/bwm.py search "ESP32-S3-DevKitC-1"
python3 scripts/bwm.py show <board-id>
python3 scripts/bwm.py pins <board-id> --connector "P1"
python3 scripts/bwm.py resolve <board-id> --board-paths catalog/board-paths.json
python3 scripts/bwm.py wiring "RS-485"
python3 scripts/bwm.py stats
```

`--catalog PATH` uses a local `library/catalog.json`; `--json` gives machine-readable
output; `--refresh` re-downloads the live catalog.

## Quick Reference

| Surface | Location |
|---|---|
| Live catalog | `https://valleytech-black-wire-guide.pages.dev/library/catalog.json` |
| Web app / search | `https://valleytech-black-wire-guide.pages.dev/` |
| Board deep link | `https://valleytech-black-wire-guide.pages.dev/?board=<board-id>` |
| Collection repo | `https://github.com/valleytechsolutions/black-wire-pinouts` |
| Raw library file | `https://raw.githubusercontent.com/valleytechsolutions/black-wire-pinouts/main/library/<file>` |
| Board page | `boards/<brand>/<family>/<slug>/README.md` in the collection repo |
| Browsable index | `BROWSE.md` in the collection repo |
| Wiki | `https://valleytech-black-wire-guide.pages.dev/wiki/boards/` |
| Desktop app | `https://github.com/valleytechsolutions/BWM-Technical-Reference-Guide/releases` |

Catalog fields that matter: `id`, `name`, `brand`, `family`, `processor`, `aliases`,
`revision`, `pinoutCoverage.status`, `assets[].type/file/url/rights/review`,
`pinConnectors`, `documentationCoverage`, `powerIds`.

`pinoutCoverage.status`: `pinout-source` (physical map exists), `functions-only` (pin
functions, no map), `reference-only` (supporting docs only), `missing`.

## Procedure

1. Search the exact model. Run `search` with the printed model code, not a family
   name. `--exact` keeps identifier matches only. Done when the top hit's brand and
   processor match the board in hand.
2. Confirm identity. Check `revision` (often "Not identified"), `coverage` and
   `partial`. Variants are separate records - match the exact board, not the family.
   Done when the record's `id` is the one you cite.
3. Read the pinout. `resolve` prints the `library/media/<sha256>.<ext>` path, the raw
   URL and the live URL for every asset. Open the image with `vision_analyze` or the
   PDF with `web_extract`; `board-paths.json` adds the human-readable board page.
   Done when you have viewed the physical pinout for the exact record.
4. Read pin functions. `pins` prints transcribed `pinConnectors` (position, label,
   functions) per connector. Transcription is `ocr-checked` or `single/double-entry`,
   not electrical verification. Done when every pin you need is accounted for.
5. Wire and protocol questions. `wiring <term>` returns the guide's contact table,
   steps, checks and sources. Done when the guide's `scope` covers the exact connector.
6. Report with provenance. Cite the source URL, the asset's `review` and `rights`, the
   board `revision`, and the coverage gaps from `pinoutCoverage.missing`. Done when the
   answer states what is sourced, what is transcribed and what is unverified.

## Pitfalls

- The catalog is ~16 MB. Never `read_file` it whole; always go through `scripts/bwm.py`.
- `library/media/` filenames are SHA-256 content hashes; `library/manifest.json` lists
  every file in the collection. A missing hash means the file is not in that snapshot.
- "Reviewed source" and "ocr-checked" are not electrical approval. `completeDeviceApproved`
  is false across the collection; never state a pin is safe to wire.
- Do not infer pin functions from a similar board, family or processor. Different
  revisions and variants carry different mappings.
- Rights: manufacturer artwork keeps its own rights; `assets[].rights` is often "Not
  established". Cite, do not redistribute.
- The collection is a snapshot: check `stats`/edition and `generatedAt` before calling
  coverage current. The live catalog and the repository can be a snapshot apart.
- The web app's search needs JavaScript; the wiki pages, `BROWSE.md` and raw files do not.
- `searchText` includes asset labels and original file paths, so a query can match a
  document instead of a board. Check the hit's `id` before trusting it.
- `boardIdAliases` maps a few duplicate records to one canonical id; the script resolves
  both directions.

## Verification

- `python3 scripts/bwm.py search "RAK13002"` ranks the RAK13002 record first and does
  not return RAK13003 for that query.
- `python3 scripts/bwm.py resolve <board-id>` prints a raw URL that returns HTTP 200 and
  a `?board=<id>` deep link.
- `python3 scripts/bwm.py pins <board-id>` prints position/label rows for a board with
  `pinConnectors`, or lists pinout assets when none are transcribed.
- `python3 scripts/bwm.py wiring "RS-485"` prints the guide's contact table and sources.
