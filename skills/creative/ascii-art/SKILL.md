---
name: ascii-art
description: Generate creative ASCII art banners, logos, and text art using Unicode box-drawing and block characters. Supports multiple styles and custom text.
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [ASCII, Art, Banners, Creative, Unicode, Text-Art]
    related_skills: [excalidraw]

---

# ASCII Art Generator

Generate creative ASCII art banners, logos, and decorative text using Unicode characters. No external tools needed -- pure text output.

## Capabilities

- **Text Banners**: Large stylized text using block characters
- **Decorative Borders**: Frames and boxes around content
- **Mini Logos**: Compact symbolic art (icons, emblems)
- **Scene Art**: Small ASCII scenes and illustrations

## Character Palette

Use these Unicode characters for rich output:

### Box Drawing
```
╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬
┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼
```

### Block Elements
```
░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞
```

### Geometric & Symbols
```
◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧
◀ ▶ ◁ ▷ ⬡ ⬢ ⟐ ⌂ ⎔ ⏣
```

### Decorative
```
╭ ╮ ╰ ╯ ─ ═ ⟨ ⟩ « » ‹ › ∙ • ·
```

## Style Guide

### 1. Block Banner Style
Best for short text (1-8 characters). Use full block characters (`█`) to form letters:

```
██╗  ██╗██╗
██║  ██║██║
███████║██║
██╔══██║██║
██║  ██║██║
╚═╝  ╚═╝╚═╝
```

### 2. Shadow Style
Add depth with half-blocks (`▄ ▀`) below or beside letters:

```
 ▄▄▄▄▄  ▄▄▄▄▄
█     █ █     █
█  ▀▀▀  █  ▀▀▀
█  ▄▄▄  █  ▄▄▄
█     █ █     █
 ▀▀▀▀▀   ▀▀▀▀▀
```

### 3. Outlined Style
Use box-drawing for clean outlines:

```
╔═══════════════════════╗
║   YOUR TEXT HERE      ║
╚═══════════════════════╝
```

### 4. Gradient Style
Use block density for gradient effects:

```
░░▒▒▓▓██ TEXT ██▓▓▒▒░░
```

### 5. Decorative Frame Style
Combine multiple character sets:

```
╭─────────────────────╮
│  ★  YOUR TEXT  ★    │
│  ◆ subtitle here ◆  │
╰─────────────────────╯
```

## Rules

1. **Max width**: 60 characters per line (terminal-safe)
2. **Max height**: 15 lines for banners, 25 for scenes
3. **Monospace only**: Output must look correct in monospace fonts
4. **No external tools**: Generate everything inline as text
5. **Alignment**: Center-align banners by default
6. **Creativity**: Mix styles freely, add decorative elements

## Examples

**User**: "Make a banner for my project called NOVA"

**Response**:
```
╔══════════════════════════════════════╗
║                                      ║
║   ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗  ║
║   ████╗  ██║██╔═══██╗██║   ██║██╔══██╗ ║
║   ██╔██╗ ██║██║   ██║██║   ██║███████║ ║
║   ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║ ║
║   ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║ ║
║   ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝ ║
║                                      ║
║       ★ The Future is Bright ★       ║
╚══════════════════════════════════════╝
```

**User**: "Draw a small rocket"

**Response**:
```
      ▲
     ╱ ╲
    ╱   ╲
   │  ◉  │
   │     │
   │  H  │
   │  E  │
   │  R  │
  ╱│  M  │╲
 ╱ │  E  │ ╲
╱  │  S  │  ╲
▔▔▔╘═════╛▔▔▔
   ╱ ▓▓▓ ╲
  ╱ ░▓▓▓░ ╲
     ░▓░
      ░
```
