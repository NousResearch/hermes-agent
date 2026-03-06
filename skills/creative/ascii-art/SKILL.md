---
name: ascii-art
description: Generate ASCII art using pyfiglet (571 fonts), cowsay, boxes, toilet, image-to-ascii conversion, and search curated art from emojicombos.com and asciiart.eu (11,000+ artworks). Falls back to LLM-generated art.
version: 3.1.0
author: 0xbyt4, Hermes Agent
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [ASCII, Art, Banners, Creative, Unicode, Text-Art, pyfiglet, figlet, cowsay, boxes]
    related_skills: [excalidraw]

---

# ASCII Art Skill

Multiple tools for different ASCII art needs. All tools are local CLI programs — no API keys required.

## Tool 1: Text Banners (pyfiglet)

Render text as large ASCII art banners. 571 built-in fonts.

### Setup

```bash
pip install pyfiglet --break-system-packages -q
```

### Usage

```bash
python3 -m pyfiglet "YOUR TEXT" -f slant
python3 -m pyfiglet "TEXT" -f doom -w 80    # Set width
python3 -m pyfiglet --list_fonts             # List all 571 fonts
```

### Recommended fonts

| Style | Font | Best for |
|-------|------|----------|
| Clean & modern | `slant` | Project names, headers |
| Bold & blocky | `doom` | Titles, logos |
| Big & readable | `big` | Banners |
| Classic banner | `banner3` | Wide displays |
| Compact | `small` | Subtitles |
| Cyberpunk | `cyberlarge` | Tech themes |
| 3D effect | `3-d` | Splash screens |
| Gothic | `gothic` | Dramatic text |

### Tips

- Preview 2-3 fonts and let the user pick their favorite
- Short text (1-8 chars) works best with detailed fonts like `doom` or `block`
- Long text works better with compact fonts like `small` or `mini`

## Tool 2: Cowsay (Message Art)

Classic tool that wraps text in a speech bubble with an ASCII character.

### Setup

```bash
sudo apt install cowsay -y    # Debian/Ubuntu
# brew install cowsay         # macOS
```

### Usage

```bash
cowsay "Hello World"
cowsay -f tux "Linux rules"       # Tux the penguin
cowsay -f dragon "Rawr!"          # Dragon
cowsay -f stegosaurus "Roar!"     # Stegosaurus
cowthink "Hmm..."                  # Thought bubble
cowsay -l                          # List all characters
```

### Available characters (50+)

`beavis.zen`, `bong`, `bunny`, `cheese`, `daemon`, `default`, `dragon`,
`dragon-and-cow`, `elephant`, `eyes`, `flaming-skull`, `ghostbusters`,
`hellokitty`, `kiss`, `kitty`, `koala`, `luke-koala`, `mech-and-cow`,
`meow`, `moofasa`, `moose`, `ren`, `sheep`, `skeleton`, `small`,
`stegosaurus`, `stimpy`, `supermilker`, `surgery`, `three-eyes`,
`turkey`, `turtle`, `tux`, `udder`, `vader`, `vader-koala`, `www`

### Eye/tongue modifiers

```bash
cowsay -b "Borg"       # =_= eyes
cowsay -d "Dead"       # x_x eyes
cowsay -g "Greedy"     # $_$ eyes
cowsay -p "Paranoid"   # @_@ eyes
cowsay -s "Stoned"     # *_* eyes
cowsay -w "Wired"      # O_O eyes
cowsay -e "OO" "Msg"   # Custom eyes
cowsay -T "U " "Msg"   # Custom tongue
```

## Tool 3: Boxes (Decorative Borders)

Draw decorative ASCII art borders/frames around any text. 70+ built-in designs.

### Setup

```bash
sudo apt install boxes -y    # Debian/Ubuntu
# brew install boxes         # macOS
```

### Usage

```bash
echo "Hello World" | boxes                    # Default box
echo "Hello World" | boxes -d stone           # Stone border
echo "Hello World" | boxes -d parchment       # Parchment scroll
echo "Hello World" | boxes -d cat             # Cat border
echo "Hello World" | boxes -d dog             # Dog border
echo "Hello World" | boxes -d unicornsay      # Unicorn
echo "Hello World" | boxes -d diamonds        # Diamond pattern
echo "Hello World" | boxes -d c-cmt           # C-style comment
echo "Hello World" | boxes -d html-cmt        # HTML comment
echo "Hello World" | boxes -a c               # Center text
boxes -l                                       # List all 70+ designs
```

### Combine with pyfiglet

```bash
python3 -m pyfiglet "HERMES" -f slant | boxes -d stone
```

## Tool 4: TOIlet (Colored Text Art)

Like pyfiglet but with ANSI color effects and visual filters. Great for terminal eye candy.

### Setup

```bash
sudo apt install toilet toilet-fonts -y    # Debian/Ubuntu
# brew install toilet                      # macOS
```

### Usage

```bash
toilet "Hello World"                    # Basic text art
toilet -f bigmono12 "Hello"            # Specific font
toilet --gay "Rainbow!"                 # Rainbow coloring
toilet --metal "Metal!"                 # Metallic effect
toilet -F border "Bordered"             # Add border
toilet -F border --gay "Fancy!"         # Combined effects
toilet -f pagga "Block"                 # Block-style font (unique to toilet)
toilet -F list                          # List available filters
```

### Filters

`crop`, `gay` (rainbow), `metal`, `flip`, `flop`, `180`, `left`, `right`, `border`

**Note**: toilet outputs ANSI escape codes for colors — works in terminals but may not render in all contexts (e.g., plain text files, some chat platforms).

## Tool 5: Image to ASCII Art

Convert images (PNG, JPEG, GIF, WEBP) to ASCII art.

### Option A: ascii-image-converter (recommended, modern)

```bash
# Install via snap or Go
sudo snap install ascii-image-converter
# OR: go install github.com/TheZoraiz/ascii-image-converter@latest
```

```bash
ascii-image-converter image.png                  # Basic
ascii-image-converter image.png -C               # Color output
ascii-image-converter image.png -d 60,30         # Set dimensions
ascii-image-converter image.png -b               # Braille characters
ascii-image-converter image.png -n               # Negative/inverted
ascii-image-converter https://url/image.jpg      # Direct URL
ascii-image-converter image.png --save-txt out   # Save as text
```

### Option B: jp2a (lightweight, JPEG only)

```bash
sudo apt install jp2a -y
jp2a --width=80 image.jpg
jp2a --colors image.jpg              # Colorized
```

## Tool 6: Search Pre-Made ASCII Art (Web APIs)

Search curated ASCII art databases via `web_extract`. No API keys needed.

### Source A: emojicombos.com (recommended first)

Huge collection of ASCII art, dot art, kaomoji, and emoji combos. Modern, meme-aware, user-submitted content. Great for pop culture, animals, objects, aesthetics.

**URL pattern:** `https://emojicombos.com/{term}-ascii-art`

```
web_extract(urls=["https://emojicombos.com/cat-ascii-art"])
web_extract(urls=["https://emojicombos.com/rocket-ascii-art"])
web_extract(urls=["https://emojicombos.com/dragon-ascii-art"])
web_extract(urls=["https://emojicombos.com/skull-ascii-art"])
web_extract(urls=["https://emojicombos.com/heart-ascii-art"])
```

**Tips:**
- Use hyphenated search terms: `hello-kitty-ascii-art`, `star-wars-ascii-art`
- Returns a mix of classic ASCII, Braille dot art, and kaomoji — pick the best style for the user
- Includes modern meme art and pop culture references
- Great for kaomoji/emoticons too: `https://emojicombos.com/cat-kaomoji`

### Source B: asciiart.eu (classic archive)

11,000+ classic ASCII artworks organized by category. More traditional/vintage art.

**Browse by category** (use as URL paths):
- `animals/cats`, `animals/dogs`, `animals/birds`, `animals/horses`
- `animals/dolphins`, `animals/dragons`, `animals/insects`
- `space/rockets`, `space/stars`, `space/planets`
- `vehicles/cars`, `vehicles/ships`, `vehicles/airplanes`
- `food-and-drinks/coffee`, `food-and-drinks/beer`
- `computers/computers`, `electronics/robots`
- `art-and-design/hearts`, `art-and-design/skulls`
- `plants/flowers`, `plants/trees`
- `mythology/dragons`, `mythology/unicorns`

```
web_extract(urls=["https://www.asciiart.eu/animals/cats"])
web_extract(urls=["https://www.asciiart.eu/search?q=rocket"])
```

**Tips:**
- Preserve artist initials/signatures (e.g., `jgs`, `hjw`) — this is important etiquette
- Better for classic/vintage ASCII art style

### Source C: GitHub Octocat API (fun easter egg)

Returns a random GitHub Octocat with a quote. No auth needed.

```bash
curl -s https://api.github.com/octocat
```

## Tool 7: LLM-Generated Custom Art (Fallback)

When tools above don't have what's needed, generate ASCII art directly using these Unicode characters:

### Character Palette

**Box Drawing:** `╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬ ┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ╭ ╮ ╰ ╯`

**Block Elements:** `░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞`

**Geometric & Symbols:** `◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧ ◀ ▶ ◁ ▷ ⬡ ⬢ ⌂`

### Rules

- Max width: 60 characters per line (terminal-safe)
- Max height: 15 lines for banners, 25 for scenes
- Monospace only: output must render correctly in fixed-width fonts

## Fun Extras

### Star Wars in ASCII (via telnet)

```bash
telnet towel.blinkenlights.nl
```

### Useful Resources

- [asciiart.eu](https://www.asciiart.eu/) — 11,000+ artworks, searchable
- [patorjk.com/software/taag](http://patorjk.com/software/taag/) — Web-based text-to-ASCII with font preview
- [asciiflow.com](http://asciiflow.com/) — Interactive ASCII diagram editor (browser)
- [awesome-ascii-art](https://github.com/moul/awesome-ascii-art) — Curated resource list

## Decision Flow

1. **Text as a banner** → pyfiglet (or toilet for colored output)
2. **Wrap a message in fun character art** → cowsay
3. **Add decorative border/frame** → boxes (can combine with pyfiglet)
4. **Art of a thing** (cat, rocket, dragon) → emojicombos.com first, then asciiart.eu
5. **Kaomoji / emoticons** → emojicombos.com (`{term}-kaomoji`)
6. **Convert an image to ASCII** → ascii-image-converter or jp2a
7. **Something custom/creative** → LLM generation with Unicode palette
8. **Any tool not installed** → install it, or fall back to next option
