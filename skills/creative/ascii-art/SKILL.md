---
name: ascii-art
description: Generate ASCII art text banners via pyfiglet (571 fonts) and search 11,000+ pre-made ASCII artworks from asciiart.eu. Falls back to LLM-generated art using Unicode characters.
version: 2.0.0
author: 0xbyt4, Hermes Agent
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [ASCII, Art, Banners, Creative, Unicode, Text-Art, pyfiglet, figlet]
    related_skills: [excalidraw]

---

# ASCII Art Skill

Three modes: text banners via pyfiglet, searching pre-made art from asciiart.eu, and LLM-generated custom art.

## Mode 1: Text Banners (pyfiglet)

Use pyfiglet to render text as large ASCII art banners. 571 fonts available, no API key needed.

### Setup (one-time)

```bash
pip install pyfiglet --break-system-packages -q
```

### Generate a banner

```bash
python3 -m pyfiglet "YOUR TEXT" -f slant
```

### List all available fonts

```bash
python3 -m pyfiglet --list_fonts
```

### Recommended fonts by style

| Style | Font | Best for |
|-------|------|----------|
| Clean & modern | `slant` | Project names, headers |
| Bold & blocky | `doom` | Titles, logos |
| Big & readable | `big` | Banners |
| Classic banner | `banner3` | Wide displays |
| Compact | `small` | Subtitles |
| Cyberpunk | `cyberlarge` | Tech themes |
| 3D effect | `3-d` | Splash screens |
| Rounded | `rounded` | Friendly text |
| Gothic | `gothic` | Dramatic text |
| Lean italic | `lean` | Stylish headers |

### Tips

- Preview 2-3 fonts and let the user pick their favorite
- Short text (1-8 chars) works best with detailed fonts like `doom` or `block`
- Long text works better with compact fonts like `small` or `mini`
- Use `python3 -m pyfiglet "TEXT" -f font_name -w 80` to set output width

## Mode 2: Search Pre-Made ASCII Art (asciiart.eu)

The ASCII Art Archive at asciiart.eu has 11,000+ artworks organized by category. Use `web_extract` to fetch them.

### Browse by category

Common categories (use as URL paths):
- `animals/cats`, `animals/dogs`, `animals/birds`, `animals/horses`
- `animals/dolphins`, `animals/dragons`, `animals/insects`
- `space/rockets`, `space/stars`, `space/planets`
- `vehicles/cars`, `vehicles/ships`, `vehicles/airplanes`
- `food-and-drinks/coffee`, `food-and-drinks/beer`
- `computers/computers`, `electronics/robots`
- `people/faces`, `people/body-parts/hands`
- `art-and-design/hearts`, `art-and-design/skulls`
- `plants/flowers`, `plants/trees`
- `mythology/dragons`, `mythology/unicorns`

```
web_extract(urls=["https://www.asciiart.eu/animals/cats"])
```

### Search by keyword

```
web_extract(urls=["https://www.asciiart.eu/search?q=rocket"])
```

### Tips

- The extracted content includes multiple art pieces — pick the best one for the user
- Preserve artist initials/signatures (e.g., `jgs`, `hjw`) — this is important etiquette
- If search returns nothing good, fall back to Mode 3 (LLM generation)

## Mode 3: LLM-Generated Custom Art (Fallback)

When pyfiglet and asciiart.eu don't have what's needed, generate ASCII art directly using these Unicode characters:

### Character Palette

**Box Drawing:**
```
╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬
┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼
╭ ╮ ╰ ╯
```

**Block Elements:**
```
░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞
```

**Geometric & Symbols:**
```
◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧
◀ ▶ ◁ ▷ ⬡ ⬢ ⟐ ⌂ ⎔ ⏣
```

### Style Guide

1. **Block Banner**: Use `█` and `╗╔╝╚` for large letter forms
2. **Shadow**: Add depth with `▄ ▀` half-blocks
3. **Gradient**: Use block density `░▒▓█` for effects
4. **Decorative Frame**: Combine box-drawing with symbols

### Rules

- Max width: 60 characters per line (terminal-safe)
- Max height: 15 lines for banners, 25 for scenes
- Monospace only: output must render correctly in fixed-width fonts
- Center-align banners by default

## Decision Flow

1. **User wants text as a banner** → Mode 1 (pyfiglet)
2. **User wants art of a thing** (cat, rocket, dragon) → Mode 2 (asciiart.eu search)
3. **User wants something custom/creative** → Mode 3 (LLM generation)
4. **Mode 2 returns nothing good** → Fall back to Mode 3
