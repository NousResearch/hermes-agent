---
name: token-juice
description: Compress tool outputs before LLM processing to save 40-60% tokens. Use this whenever you read files, search the web, browse pages, receive terminal output, or process any tool output larger than 2K characters — especially before large files, search results, web extracts, and verbose terminal commands. Automatically strip boilerplate, deduplicate URLs, summarize verbose sections, and preserve critical data (errors, paths, keys). CJK/emoji-safe.
---

# TokenJuice — Output Compression

Goal: compress every tool output before it enters LLM context. Inspired by openhuman's TokenJuice engine.

## Rules (applied mentally BEFORE processing any output)

### 1. Terminal Output (>3K chars)
- Keep: last 80 lines + error lines + exit code
- Drop: repetitive logs, progress bars, npm/pip install lists
- Crash/stacktrace: keep full traceback, drop everything else

### 2. Web Search Results
- Deduplicate by URL (keep first occurrence)
- If >5 results, keep top 3 by description length + 1 wildcard
- Trim descriptions to 150 chars each

### 3. Web Extracts / Browsed Pages (>3K chars)
- Strip: nav, footer, sidebar, cookie banners, ads
- Keep: main content, headings, code blocks
- If still >3K: ask "is every paragraph contributing to the task?" — drop those that aren't

### 4. File Reads (>200 lines single file)
- Skip: blank lines, comment blocks >5 lines, import sections >10 lines (after noting "standard imports: x")
- Keep: function signatures, exports, business logic
- Flag: if file is config/template, read full — compression kills accuracy here

### 5. Multi-File Operations
- After reading 3+ files: mentally group into "core files" (read full) and "supporting files" (compress)
- Skip files that are obvious boilerplate (`.gitignore`, `package-lock.json`, etc.)

### 6. General
- ALWAYS preserve: file paths, error codes, API keys (redacted), URLs, line numbers
- Preserve CJK characters and emoji grapheme-by-grapheme
- When in doubt, keep it — truncation is cheaper than wrong truncation

## The Compression Script

When processing output programmatically, use `scripts/compress.py`:

```bash
# Pipe terminal output
cat output.txt | python3 ~/.hermes/skills/token-juice/scripts/compress.py --type terminal

# Compress web extract
cat page.md | python3 ~/.hermes/skills/token-juice/scripts/compress.py --type web

# Compress search results JSON
python3 ~/.hermes/skills/token-juice/scripts/compress.py --type search --input results.json
```

Types: `terminal`, `web`, `search`, `file`, `auto` (detect from content)

## Anti-Patterns to Avoid
- Don't compress files you're about to edit
- Don't compress error messages or stack traces
- Don't compress when the user says "show me everything"
- Don't use the script for outputs under 2K chars (overhead > savings)
