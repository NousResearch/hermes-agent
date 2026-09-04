---
name: qr-code
description: "Generate and decode QR codes for URLs, text, and contacts."
version: 1.0.0
author: Nolan (nolanchic)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Productivity, QR, Encoding, Media, Utilities]
    related_skills: [ocr-and-documents]
---

# QR Code Skill

Generate QR codes (encode text/URLs/Wi-Fi/vCards to PNG or terminal art) and decode
them (read a QR image back into text). No API key, no network calls — runs entirely
on the local machine. This skill does **not** invent QR payloads; it encodes data you
provide and decodes images you point it at.

## When to Use

- Share a URL, message, or command so a phone can scan it.
- Encode Wi-Fi credentials, a vCard, or a calendar event for quick onboarding.
- Decode a QR image saved locally (screenshot, photo, downloaded PNG).
- Batch-generate QR labels for a list of values.

## Prerequisites

All backends are **optional** — pick whichever is installed. The helper script auto-detects
the best available encoder/decoder and tells you what is missing.

| Capability | Tool | Install |
|------------|------|---------|
| Encode (recommended) | `qrencode` | `brew install qrencode` · `apt install qrencode` · `choco install qrencode` |
| Encode (fallback) | Python `qrcode` | `pip install qrcode[pil]` |
| Decode (recommended) | `zbarimg` | `brew install zbar` · `apt install zbar-tools` |
| Decode (fallback) | Python `pyzbar` + Pillow | `pip install pyzbar pillow` |

On Windows, `zbarimg` is uncommon — prefer the Python `pyzbar` decoder there. Check what
is available before first use:

```bash
python3 scripts/qr_code.py doctor
```

## How to Run

All actions go through one helper, invoked via the `terminal` tool:

```bash
# Encode a URL to a PNG file
python3 scripts/qr_code.py encode "https://example.com" -o site.png

# Encode straight to the terminal as text art (no file)
python3 scripts/qr_code.py encode "Hello world" --terminal

# Encode Wi-Fi credentials
python3 scripts/qr_code.py wifi --ssid "MyNetwork" --password "s3cret" -o wifi.png

# Encode a vCard (contact)
python3 scripts/qr_code.py vcard --name "Ada Lovelace" --phone "+15551234" -o ada.png

# Decode a QR image back to text
python3 scripts/qr_code.py decode image.png

# Decode and pipe the payload somewhere
python3 scripts/qr_code.py decode image.png --raw

# Batch encode one PNG per line of an input file
python3 scripts/qr_code.py batch values.txt --outdir qr/
```

Generated files land in the working directory unless `-o` / `--outdir` says otherwise.
Use `write_file` to save a payload list, and `read_file` to inspect a decoded result.

## Quick Reference

| Action | Command |
|--------|---------|
| Check installed backends | `python3 scripts/qr_code.py doctor` |
| Encode text → PNG | `python3 scripts/qr_code.py encode "TEXT" -o out.png` |
| Encode text → terminal | `python3 scripts/qr_code.py encode "TEXT" --terminal` |
| Wi-Fi QR | `python3 scripts/qr_code.py wifi --ssid S --password P -o w.png` |
| vCard QR | `python3 scripts/qr_code.py vcard --name N --phone P -o v.png` |
| Decode image | `python3 scripts/qr_code.py decode img.png` |
| Batch encode | `python3 scripts/qr_code.py batch list.txt --outdir qr/` |

## Procedure

1. **Check backends** once: `python3 scripts/qr_code.py doctor`. Note what is installed.
2. **Encode** a payload. For anything scannable by a phone, write a PNG with `-o`.
   Use `--terminal` only for short, screen-only payloads.
3. **Decode** by pointing at an image path. The helper prints the payload; add `--raw`
   to get the bare string for piping.
4. **Verify** the round-trip with the command in `## Verification` below.

### Payload format notes

- **Wi-Fi**: encoded as `WIFI:T:WPA;S:<ssid>;P:<password>;;` — Android and iOS camera
  apps prompt to join the network directly.
- **vCard**: minimal 2.1 record. Add more fields by extending the template in the script
  if you need email/address/note.
- **Error correction**: default level `M` (~15% recovery). Pass `--ec H` for damaged
  or densely printed codes (~30% recovery).

## Pitfalls

- **`zbarimg` returns exit 1 on a clean "no code found"** — the helper treats this as a
  normal "could not decode" result, not a crash. A corrupted or non-QR image yields the
  same outcome.
- **`pyzbar` on Linux needs the `libzbar0` shared library**, not just the pip package.
  Install `zbar-tools` (Debian) / `zbar` (Homebrew) if you see `Unable to find zbar`.
- **Terminal art rendering is font-dependent.** Use a monospace font; very large payloads
  produce oversized blocks that may not fit a small terminal — prefer a PNG file instead.
- **Special characters** in payloads (spaces, `;`, `:`) are fine — the helper passes the
  argument as a single value, no shell interpolation. Do **not** wrap the payload in extra
  quotes inside the argument.
- **Maximum capacity** per QR depends on error-correction level and character set:
  ~2,953 ASCII bytes at the lowest density. URLs are usually fine; full documents are not.

## Verification

Confirm encode + decode round-trip end to end:

```bash
python3 scripts/qr_code.py encode "hermes-qr-ok" -o /tmp/qrtest.png && \
  python3 scripts/qr_code.py decode /tmp/qrtest.png --raw
# Expected output: hermes-qr-ok
```

If that prints `hermes-qr-ok`, both an encoder and a decoder are installed and working.
