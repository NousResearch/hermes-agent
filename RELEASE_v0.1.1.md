```
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓██████▓▒░░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░    ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░  

                A U T O L Y C U S
              v e r s i o n  0 . 1 . 1
```

---

## Release Notes — Autolycus v0.1.1

**Release Date:** May 17, 2026

> Unified vision pipeline — the auxiliary model paradigm is retired. With qwen3.5+ and qwen3.6, LLMs and vision models are no longer separate. Autolycus now processes images directly through the main model, eliminating redundant API calls, secondary model loading, and credit drain. Fully compliant with Hermes 0.14.0.

---

## ✨ Highlights

- **Unified Vision Pipeline** — The 4700-line auxiliary client chain has been replaced with a 9KB self-contained module. All image processing now routes through the main model.

- **Hermes 0.14.0 Compliance** — Autolycus is fully synchronized with the latest upstream Hermes Agent release, inheriting all stability improvements, bug fixes, and feature additions.

- **Zero Information Loss** — The main model sees actual pixels, not a lossy text description. No more "describe this image" round-trips.

---

## 👁️ Unified Vision

### The Problem

The Hermes Agent architecture maintained a separate "auxiliary vision" backend (typically Gemini Flash) for image processing. When the agent called `vision_analyze`, it went through a 4700-line resolution chain to hit a secondary model, then returned a text description to the main model. This meant:
- Extra API credits burned on a separate backend
- Secondary model loading (causing VRAM OOM during context compression)
- Information loss from pixel-to-text-to-pixel translation

### The Solution

Modern multimodal models (qwen3.5+, qwen3.6) handle images natively. The lycus branch now routes everything through the main model.

**Before:**
```
User image → vision_analyze_tool()
  → async_call_llm(task="vision")
    → resolve_vision_provider_client() [4700 lines]
      → Gemini Flash / separate vision backend
        → text description returned to agent
```

**After:**
```
User image → vision_analyze_tool()
  → _lycus_async_vision_call()
    → resolve_runtime_provider(requested="custom")
      → Same main model (qwen3.6) handles everything
        → Native image understanding, no information loss
```

### Technical Implementation

- **New file:** `agent/lycus_vision.py` (9KB) — self-contained unified vision module
- **Patched:** `cli.py` (2 additions) — import at startup, patch after class definition
- **4 injection points patched at runtime:**
  1. `agent.auxiliary_client.async_call_llm` → `_lycus_async_vision_call`
  2. `tools.vision_tools.async_call_llm` → `_lycus_async_vision_call`
  3. `agent.image_routing.decide_image_input_mode` → always returns `"native"`
  4. `cli.ChatConsole._preprocess_images_with_vision` → simplified fallback

### Why This Survives Upstream Changes

- `agent/lycus_vision.py` is a **new file** — upstream merges won't touch it
- The 2 cli.py additions are single-line imports/patches, easy to re-apply
- Runtime monkey-patching works regardless of upstream function signatures
- Graceful fallback: if lycus_vision import fails, upstream behavior is preserved

---

## 🛠 Technical Changes

- Added `agent/lycus_vision.py` — unified vision pipeline module
  - `_lycus_async_vision_call()` — simplified async LLM call (~50 lines)
  - `patch_cli_preprocess()` — deferred CLI patch for ChatConsole
  - Auto-applies all patches at import time

- Modified `cli.py`:
  - Line 48: `import agent.lycus_vision` (triggers patches at startup)
  - Line 2323: `agent.lycus_vision.patch_cli_preprocess(ChatConsole)` (applies after class definition)

- Updated `vision_analyze` tool schema description — no longer references "auxiliary vision model"

---

## 📋 Compatibility

- **Requires:** Multimodal-capable main model (qwen3.5+, qwen3.6, or equivalent)
- **Hermes 0.14.0:** Fully compliant — all upstream features inherited
- **Graceful degradation:** If lycus_vision import fails, upstream behavior is preserved
- **Config:** No schema changes — `auxiliary.vision` settings are simply ignored

---

## 📝 Notes

This release marks a philosophical shift: the separation between "language model" and "vision model" is obsolete. Autolycus embraces this reality, treating the main model as a true multimodal engine. The auxiliary paradigm is retired for the lycus branch while remaining available upstream for users who still need it.

**One model. One pipeline. No separation.**
