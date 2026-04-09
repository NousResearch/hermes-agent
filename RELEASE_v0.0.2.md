```
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░    ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░  

                A U T O L Y C U S
              v e r s i o n  0 . 0 . 2
```

---

## Release Notes — Autolycus v_0.0.2

**Release Date:** April 9, 2026  
**Platform:** FreeBSD only (native)  
**License:** LGPL v2.1

### About This Release

Autolycus v_0.0.2 marks our first **functional release**. The agent can now execute terminal commands reliably on FreeBSD, read and write files, browse the web, interact with GitHub, and perform intelligent automation tasks.

This release fixes a critical terminal execution bug that prevented all shell commands from running, making Autolycus a fully operational AI agent for FreeBSD systems.

---

### What's New

#### 🎉 Terminal Execution Now Works

The core issue blocking all command execution has been resolved:

- **PTY-based subprocess handling** — Replaced `subprocess.Popen` with `ptyprocess` for reliable pseudo-terminal allocation
- **Cross-platform compatibility** — The fix works on FreeBSD, Linux, and macOS (though Autolycus is FreeBSD-only)
- **Interrupt support maintained** — Ctrl+C still cancels running commands gracefully
- **Timeout enforcement** — Commands exceeding their timeout are properly terminated

Before this release, every command timed out with exit code 124. Now they work as expected:

```bash
$ hermes
> What's my current directory?
🔧 Using tool: terminal_tool
$ pwd
/home/user/project
✅ Your current directory is /home/user/project
```

#### 📦 Dependency Updates

- **ptyprocess** moved from optional `[pty]` extra to core dependencies — required for FreeBSD operation
- Removed the now-redundant `[pty]` optional dependency group
- All 126 packages install cleanly on FreeBSD 15.0 without voice extras

#### 📝 Documentation Improvements

- Updated installation instructions with accurate package names and build times
- Added notes about uv's ~4 minute compile time from source
- Documented clipboard support requirement (xclip/xsel)
- Clarified voice transcription limitations on FreeBSD

---

### What Works

| Feature | Status | Notes |
|---------|--------|-------|
| Terminal execution | ✅ Working | PTY-based, interrupt/timeout support |
| File operations | ✅ Working | Read, write, edit, search files |
| Web browsing | ✅ Working | Firecrawl and Parallel web tools |
| GitHub integration | ✅ Working | Issues, PRs, repo search |
| Memory system | ✅ Working | Persistent memory with FTS5 search |
| Skills system | ✅ Working | 71 built-in skills available |
| CLI interface | ✅ Working | Full TUI with streaming output |
| Model providers | ✅ Working | LM Studio, Ollama, OpenRouter, etc. |
| Voice transcription | ❌ Unavailable | faster-whisper has no FreeBSD wheels |

---

### Known Limitations

1. **Voice tools unavailable** — `faster-whisper` depends on `ctranslate2`, which has no FreeBSD wheels. Use cloud-based alternatives:
   - Set `GROQ_API_KEY` for Groq-powered STT
   - Set `VOICE_TOOLS_OPENAI_KEY` for OpenAI Whisper API

2. **Clipboard support requires xclip/xsel** — Install with `pkg install xclip` for clipboard tools to work

3. **uv must be built from source** — No FreeBSD binary available; expect ~4 minute compile time with `cargo install uv`

---

### Installation

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

### Upgrade from v_0.0.1

If you already have v_0.0.1 installed:

```bash
cd autolycus-agent
git pull
. venv/bin/activate  # or your venv name
uv pip install -e ".[modal,daytona,messaging,cron,cli,dev,tts-premium,slack,honcho,mcp]" --upgrade
```

The ptyprocess package will be installed automatically as part of the core dependencies.

---

### Credits

Built on the [Hermes Agent](https://github.com/NousResearch/hermes-agent) architecture by Nous Research.  
An independent project by **Technetia Inc**.

Special thanks to the FreeBSD community for maintaining a world-class Unix operating system.

---

*The world's first AI agent for FreeBSD.*
