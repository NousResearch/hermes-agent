# Known Pitfalls — Local Models (Gemma 4, etc.)

Local models (Gemma 4, others < 10B) have specific weaknesses with the extended memory system. **READ BEFORE ANY MODIFICATION.**

## 🐛 Observed bug (Gemma 4, Aug 2026)

Gemma 4 was asked to "extract preferences.md info into a file called writing". It:
1. **Hallucinated** the creation of `writing.md` (the file did not exist)
2. **Confused index and detail**: tried to patch `extended/preferences.md` with a line that looked like a MEMORY.md index entry
3. **Declared successful reads** that never happened (fictional state)
4. Added a **parasite entry** in MEMORY.md (a full system description instead of a title)

## Safety rules for weak models

1. **ALWAYS verify real state BEFORE modifying:**
   ```bash
   ls -la memories/extended/
   cat memories/MEMORY.md
   ```
   Never trust the model's internal "memory" of what exists.

2. **NEVER put long content in MEMORY.md** — only one-line titles with `→ see extended/<file>.md`. A parasite entry is a bug.

3. **Accent-free filenames** (`writing.md`, not `écriture.md`) — accents break under MSYS/Windows.

4. **Two distinct files:**
   - `memories/MEMORY.md` = the INDEX (pointers only)
   - `memories/extended/*.md` = the DETAIL
   Never patch an extended/ file with an index line. Never put detail in MEMORY.md.

5. **After each write**, re-read the written file to verify (`cat`).

6. **If a patch fails**, do not retry with an "old string seen in memory" — re-read the file first (real content may differ from what you believe).

## Correct workflow for "moving info"

1. `read_file` the source file (e.g. preferences.md)
2. `write_file` the new file (e.g. writing.md) with the extracted content
3. `write_file` the cleaned source file (without the moved content)
4. `write_file` MEMORY.md with the correct index (1 line per topic)
5. `cat` MEMORY.md + `ls extended/` to verify
