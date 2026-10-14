# Wayback Machine — Internet Archive CDX

## 1. Summary

The Internet Archive's Wayback Machine has captured ~900B+ web pages since
1996. The CDX server API indexes those captures by URL, timestamp, and
content hash. Free, anonymous, no auth.

## 2. Access Methods

- **CDX server:** `https://web.archive.org/cdx/search/cdx`
- **Wayback URL:** `https://web.archive.org/web/<timestamp>/<url>`
- **Save Page Now (write):** `https://web.archive.org/save/<url>` (different API)
- **Auth:** None
- **Rate limit:** Generous; be polite (~1 req/s)

## 3. Data Schema

Key fields emitted by `fetch_wayback.py`:

| Column | Type | Description |
|--------|------|-------------|
| `url` | str | Original URL captured |
| `timestamp` | str | YYYYMMDDHHMMSS (CDX format) |
| `wayback_url` | str | Direct replay URL |
| `mimetype` | str | Content-type at capture |
| `status` | str | HTTP status (typically 200) |
| `digest` | str | SHA1 of capture content (collapse-friendly) |
| `length` | str | Byte length of capture |

## 4. Coverage

- 1996 → present
- ~900B+ captures across ~700M domains
- Updated continuously by automated crawls + manual saves
- Some domains have aggressive coverage (news), others sparse (private)

## 5. Cross-Reference Potential

- **Wikipedia** ↔ Reverse-lookup pages cited as references that have since
  disappeared
- **News URLs** ↔ Original article content when present-day URLs 404
- **Corporate websites** ↔ Historical "About" pages, executive bios that
  have been scrubbed

The Wayback CDX is most useful as a **content-recovery** layer when other
sources point to URLs that no longer exist.

## 6. Data Quality

- robots.txt-blocked domains may have spotty or no coverage
- Captures vary in completeness (HTML may be saved without CSS/JS)
- Some content is excluded by domain owner request (DMCA, etc.)
- Coverage of "deep links" (URLs with query strings) is uneven
- Time resolution is per-capture, not continuous — gaps are common

## 7. Acquisition Script

Path: `scripts/fetch_wayback.py`

```bash
# All captures of a specific URL
python3 SKILL_DIR/scripts/fetch_wayback.py --url "https://example.com/page" \
    --out data/wb.csv

# All captures of a host
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --out data/wb.csv

# All captures of a domain + subdomains
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match domain --out data/wb.csv

# Only unique-content captures within a date window
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --collapse digest \
    --from-date 2020-01-01 --to-date 2023-12-31 \
    --out data/wb.csv
```

## 8. Legal & Licensing

- Internet Archive captures are made under fair-use research provisions
- Replay URLs are stable references — citing them is encouraged
- Internet Archive non-profit terms of use govern content
- Some content is rights-restricted; replay may be blocked even if the
  CDX entry shows it as captured

## 9. References

- CDX server docs: https://github.com/internetarchive/wayback/blob/master/wayback-cdx-server/README.md
- Wayback API: https://archive.org/help/wayback_api.php
- Internet Archive: https://archive.org/
