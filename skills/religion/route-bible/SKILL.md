---
name: route-bible
description: "Reference Bible passages via app-agnostic links and QR codes."
version: 1.0.0
author: Dylan Shade (dpshde)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Bible, Religion, Scripture, Routing, API]
    related_skills: [ocr-and-documents]
---

# route.bible

Reference Bible passages and generate app-agnostic links or QR codes. route.bible acts as a universal router, allowing users to open scriptures in their preferred Bible application (YouVersion, BibleGateway, Logos, etc.).

## Quick Reference

| Action | Pattern |
|--------|---------|
| Generate link | `https://route.bible/GEN.1.1` |
| Generate QR code | `https://route.bible/GEN.1.1?qr=true` |
| Local parsing | Use `grab-bcv` or similar for OSIS ID generation |

## Usage

The routing layer follows a stable URL contract using standard book abbreviations and OSIS-style references.

### Canonical Link Generation

To generate a universal link for a passage, use the following URL pattern:
`https://route.bible/<BOOK>.<CHAPTER>.<VERSE>`

Examples:
- `https://route.bible/GEN.1.1`
- `https://route.bible/JHN.3.16`
- `https://route.bible/PSA.23`

### QR Code Integration

To generate a QR code pointing to a passage, append `?qr=true` to the canonical link.
- `https://route.bible/ROM.12.1-2?qr=true`

## Implementation Details

### Identification (OSIS IDs)

When generating links, use standard 3-letter abbreviations (e.g., `GEN`, `EXO`, `MAT`, `MRK`).

### Local BCV Parsing

For high-volume local processing or command-line integration, use the `grab-bcv` utility (if available) to normalize references before routing.

```bash
# Example: Normalize a human reference to an OSIS ID
echo "John 3:16" | python3 -c "import sys; print('JHN.3.16') # Placeholder for grab-bcv logic"
```

## Verification

Confirm a link is correctly formatted by checking the book abbreviation against the standard list.
- `https://route.bible/ACT.1.8` is valid.
- `https://route.bible/ACTS.1.8` should be normalized to `ACT.1.8`.
