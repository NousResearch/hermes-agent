---
name: photo-alchemy
description: Transform any photo into surrealist AI art. Uses Claude to write a story about your photo, then Gemini generates a reimagined version in one of 35+ visual styles. Deep Apple Photos integration — pick from albums, save back, schedule as an Apple TV screensaver.
version: 1.0.0
author: hbmartin
license: Apache-2.0
metadata:
  hermes:
    tags: [AI, Images, Claude, Gemini, Apple-Photos, Creative, macOS]
    homepage: https://github.com/hbmartin/imagemine
---

# photo-alchemy

Transform any photo into surrealist AI art via a two-step pipeline:
1. Claude writes a surrealist story + image prompt about your photo
2. Gemini generates a brand-new image from that description in a random visual style

Built by Harold Martin. Installable via uvx — no setup required beyond API keys.

## Prerequisites

- Python 3.14+
- Anthropic API key (`ANTHROPIC_API_KEY`)
- Google Gemini API key (`GEMINI_API_KEY`)

## Installation

### One-off (no install)
```bash
uvx imagemine path/to/photo.jpg
```

### Permanent install
```bash
curl -LsSf uvx.sh/imagemine/install.sh | sh
```

## API Key Setup

Keys are resolved in order: SQLite DB → env vars → interactive prompt (saved to DB on first entry).

```bash
# Via environment
export ANTHROPIC_API_KEY=***
export GEMINI_API_KEY=***

# Or via interactive config wizard (saves to ~/.imagemine.db)
imagemine --config
```

## Basic Usage

```bash
# Transform a single photo
imagemine photo.jpg

# Pick a random photo from a macOS Photos album
imagemine --input-album "Camera Roll"

# Save output to a specific directory
imagemine photo.jpg --output-dir ~/Desktop/output

# Use a specific visual style instead of random
imagemine photo.jpg --style "Ukiyo-e woodblock print, bold outlines, flat color"

# Tune creativity (default 1.0 for both)
imagemine photo.jpg --desc-temp 1.5 --img-temp 0.8

# Pick style interactively from a numbered table
imagemine photo.jpg --choose-style

# Blend multiple styles (enter comma-separated numbers)
imagemine photo.jpg --choose-style  # then enter: 1,5,12

# JSON output (for scripting)
imagemine photo.jpg --json

# Show recent run history with timing sparklines
imagemine --history
```

## Visual Styles

35+ built-in styles including: Watercolor, 8-Bit Pixel Art, Ukiyo-e Woodblock, Neon Noir,
Tarot Card, Vaporwave, Glitch Art, Renaissance Painting, and more.

```bash
imagemine --list-styles          # show all styles with usage count
imagemine --add-style            # add a custom style interactively
imagemine --remove-style         # remove styles interactively
```

## Apple Photos Integration

imagemine reads face-detection names from Photos albums and uses them in prompts.
Use character mappings to rename people before they reach the AI (e.g. "John" → "Captain America").

```bash
# Pick input from album, save generated image back to another album
imagemine --input-album "Camera Roll" --destination-album "AI Art"

# Manage character name mappings
imagemine --add-character-mapping
imagemine --list-character-mappings
imagemine --remove-character-mapping
```

## Apple TV Screensaver (killer feature)

Run photo-alchemy on a schedule to continuously generate new art into a shared Photos album,
then set that album as your Apple TV screensaver for a living, ever-changing art display.

### Setup

1. Create two macOS Photos albums: one for source photos, one for output (make output a Shared Album so Apple TV can see it)
2. Configure albums:
```bash
imagemine --config
# Set INPUT_ALBUM and DESTINATION_ALBUM
```

3. Schedule via launchd (runs every N minutes):
```bash
imagemine --launchd 30
# Writes ~/Library/LaunchAgents/imagemine.plist and prints the launchctl command
launchctl load ~/Library/LaunchAgents/imagemine.plist
```

4. On Apple TV: Photos app → Shared Albums → select your output album → Set as Screen Saver

### Apple TV Pitfalls
- Output album must be a **Shared Album** for Apple TV to see it
- New images appear in the screensaver automatically once added to the shared album
- Use `--config-path` if you want a non-default DB location with launchd

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for story/prompt generation |
| `GEMINI_MODEL` | `gemini-3-pro-image-preview` | Gemini model for image generation |
| `DEFAULT_DESC_TEMP` | `1.0` | Claude sampling temperature |
| `DEFAULT_IMG_TEMP` | `1.0` | Gemini sampling temperature |
| `INPUT_ALBUM` | — | macOS Photos album to pick input from |
| `DESTINATION_ALBUM` | — | macOS Photos album to save output to |
| `ASPECT_RATIO` | `4:3` | Output image aspect ratio (1:1, 3:4, 4:3, 9:16, 16:9) |

## Pipeline Steps (what happens under the hood)

1. Resize input to max 1024px (preserving aspect ratio), save to disk
2. Send resized image to Claude via Files API → generates surrealist story + image prompt
3. Pick a random visual style from the style library (or use `--style`)
4. Pass story + style + resized image to Gemini → generates the new image
5. Save run metadata to `~/.imagemine.db` (SQLite)
6. Optionally import generated image into a macOS Photos album

## Run History & Database

Every run is recorded in `~/.imagemine.db` with input path, generated story, style used,
model names, per-step timing (ms), and output path.

```bash
imagemine --history              # show recent runs with timing sparklines
imagemine --config-path ~/custom.db  # use a different DB location
```

## Notes

- A new Claude description is generated on every run — no caching of stories
- `--fresh` picks from least-used styles (good for variety over time)
- `--session-svg` saves a styled SVG of the terminal session alongside the output image
- macOS only for Photos/launchd features; core image generation works on any platform
