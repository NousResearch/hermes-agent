```
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░    ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░  

                A U T O L Y C U S
              v e r s i o n  0 . 0 . 3
```

---

## Release Notes — Autolycus v_0.0.3

**Release Date:** April 9, 2026  
**Platform:** FreeBSD only (native)  
**License:** LGPL v2.1

### About This Release

Autolycus v_0.0.3 delivers a complete **TUI rebranding and accessibility overhaul**. The interface now features a distinctive blue/teal color scheme optimized for colorblind users, consistent fox emoji branding 🦊, and professional Autolycus identity throughout all user-facing components.

This release transforms the visual experience while maintaining 100% functional compatibility with v_0.0.2.

---

### What's New

#### 🎨 Complete TUI Rebranding

**New Color Scheme:**
- **Teal (#00CED1, #00d4aa)** - Primary accent color replacing gold/yellow
- **Royal Blue (#4169E1)** - Secondary branding color
- **Dark Navy (#0a3d62)** - High-contrast borders and labels
- **Bright Purple (#b163db)** - Model name display in status bar
- **Sky Blue (#00BFFF)** - Tools/skills summary

**Before (Hermes-inspired):** Gold, yellow, red palette  
**After (Autolycus):** Teal, blue, navy palette

#### ♿ Colorblind Accessibility Improvements

The new color scheme was specifically designed for accessibility:
- **Deuteranopia-friendly** - Teal/blue distinguishable from green/red confusion
- **High contrast ratios** - All text meets WCAG AA standards (4.5:1 minimum)
- **Orange warnings (#ff8c00)** - Replaced yellow which is problematic for many colorblind users
- **White body text (#ffffff)** - Maximum readability against dark backgrounds

#### 🦊 Consistent Fox Branding

The fox emoji now appears throughout the interface:
- Goodbye message: `"Goodbye! 🦊"`
- Response box label: `" 🦊 Autolycus "`
- Execute code output (replaced duck emoji 🐍)
- Skin branding strings across all built-in themes

#### 🔧 Bug Fixes

**Duplicate Logo Issue:**
- Removed redundant `AUTOLYCUS_LOGO` from inside TUI frame panel
- Logo now appears only in top banner (cleaner layout)

**Box-Drawing Continuity:**
- Fixed border rendering issues in response panels
- All box-drawing characters now properly aligned across terminals

#### 📝 Updated Built-in Skins

All three built-in skins rebranded with Autolycus identity:

| Skin | Description | Colors |
|------|-------------|--------|
| **default** | "Autolycus — high-contrast teal and blue for FreeBSD (colorblind-friendly)" | Teal/Blue/Navy |
| **mono** | "Monochrome — clean grayscale" with Autolycus branding | Grayscale |
| **slate** | "Cool blue — developer-focused" with Autolycus branding | Blue/Slate |

*Note: Thematic skins (Ares, Poseidon) retain their unique identities.*

---

### Technical Changes

#### Files Modified (4 commits, 243 lines changed)

| File | Lines Changed | Description |
|------|---------------|-------------|
| `hermes_cli/banner.py` | +87/-87 | New ASCII logo, color scheme, version string |
| `cli.py` | +58/-58 | Response box colors, fox emoji, status bar updates |
| `hermes_cli/skin_engine.py` | +56/-56 | All built-in skins rebranded with Autolycus identity |
| `hermes_cli/main.py` | +4/-2 | Version string update (v0.8.0 → v0.0.2) |
| `hermes_cli/providers.py` | +36/-36 | Provider display colors updated |
| `agent/display.py` | +2/-1 | Fox emoji in execute_code output |

#### Commit History

```
72d3b9b3 Update TUI colors and branding for Autolycus theme
a359a891 TUI rebranding part 3: high-contrast colors for colorblind accessibility
ff55012e TUI rebranding part 2: welcome message, skin colors, fix duplicate logo
eaaab868 TUI rebranding: Autolycus banner, blue/teal colors, version update
```

---

### What Works (Inherited from v_0.0.2)

| Feature | Status | Notes |
|---------|--------|-------|
| Terminal execution | ✅ Working | PTY-based, interrupt/timeout support |
| File operations | ✅ Working | Read, write, edit, search files |
| Web browsing | ✅ Working | Firecrawl and Parallel web tools |
| GitHub integration | ✅ Working | Issues, PRs, repo search |
| Memory system | ✅ Working | Persistent memory with FTS5 search |
| Skills system | ✅ Working | 71 built-in skills available |
| CLI interface | ✅ Working | Full TUI with streaming output (now rebranded!) |
| Model providers | ✅ Working | LM Studio, Ollama, OpenRouter, etc. |
| Voice transcription | ❌ Unavailable | faster-whisper has no FreeBSD wheels |

---

### Known Limitations (Unchanged)

1. **Voice tools unavailable** — `faster-whisper` depends on `ctranslate2`, which has no FreeBSD wheels. Use cloud-based alternatives:
   - Set `GROQ_API_KEY` for Groq-powered STT
   - Set `VOICE_TOOLS_OPENAI_KEY` for OpenAI Whisper API

2. **Clipboard support requires xclip/xsel** — Install with `pkg install xclip` for clipboard tools to work

3. **uv must be built from source** — No FreeBSD binary available; expect ~4 minute compile time with `cargo install uv`

---

### Installation (Updated)

```bash
git clone https://github.com/waym0reom3ga/autolycus-agent.git
cd autolycus-agent
cargo install uv  # ~4 minutes
export PATH="$HOME/.cargo/bin:$PATH"
uv venv venv --python 3.11
. venv/bin/activate
uv pip install -e ".[modal,daytona,messaging,cron,cli,dev,tts-premium,slack,honcho,mcp]"
mkdir -p ~/.local/bin && ln -sf $(pwd)/venv/bin/hermes ~/.local/bin/hermes
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Then run `hermes setup` to configure your model provider and API keys.

---

### Upgrade from v_0.0.2

If you already have v_0.0.2 installed:

```bash
cd autolycus-agent
git pull
. venv/bin/activate  # or your venv name
uv pip install -e ".[modal,daytona,messaging,cron,cli,dev,tts-premium,slack,honcho,mcp]" --upgrade
```

The new color scheme and branding will be applied automatically on next startup!

---

### Visual Comparison

#### Before (v_0.0.2 - Hermes-inspired):
```
╔══════════════════════════════════════════════════════════════╗
║  ⚕ HERMES - AI Agent Framework              ║
║  Built on Nous Research architecture         Nous Research ║
╚══════════════════════════════════════════════════════════════╝
Colors: Gold (#FFD700), Yellow (#FFBF00)
```

#### After (v_0.0.3 - Autolycus):
```
╔══════════════════════════════════════════════════════════════╗
║  ⚕ AUTOLYCUS - AI Agent Framework              ║
║  The World's First AI Agent for FreeBSD    Technetia Inc   ║
╚══════════════════════════════════════════════════════════════╝
Colors: Teal (#00CED1), Blue (#4169E1)
```

---

### Credits

Built on the [Hermes Agent](https://github.com/NousResearch/hermes-agent) architecture by Nous Research.  
An independent project by **Technetia Inc**.

Special thanks to:
- The FreeBSD community for maintaining a world-class Unix operating system
- All contributors who made v_0.0.2's PTY fix possible

---

### What's Next?

**v_0.0.4 and beyond:**
- Hermes script compatibility test suite
- Automated cross-platform testing (FreeBSD/Linux)
- Additional FreeBSD-specific optimizations
- Community-contributed skins

Stay tuned! 🦊

---

*The world's first AI agent for FreeBSD.*  
*Now with colorblind-friendly accessibility!*
