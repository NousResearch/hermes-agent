# PR 3/3: chore/wiki-lifecycle-daily-helpers

**Branch:** `chore/wiki-lifecycle-daily-helpers`
**Base:** `main` @ `4d39a60`
**Scope:** Wiki lifecycle daily helpers (single commit `7fc9d618`).
**Source:** Izdvojeno iz #42341 (commit `7fc9d61810d0ecbecea8276b9ebf4bdb37369f1d`).

## Files
- `scripts/wiki_lifecycle.py` (+186) — CLI helper
- `tools/wiki_lifecycle.py` (+383) — core lifecycle engine

## Što radi
- Daily maintenance za lokalni wiki (vault-style markdown KB):
  - index refresh
  - orphan detection
  - backlink rebuild
  - archive rotation

## Ovisnosti
- **Nema** dependency-ja na PR 1 ili PR 2. Čist chore PR.

**Preporuka:** Može ići paralelno s PR 1 i PR 2, neovisno o njima.

## Acceptance za maintainere
- `python scripts/wiki_lifecycle.py --help` radi
- `python tools/wiki_lifecycle.py --dry-run` izvještava bez write-ova
- Postoji unit test za archive rotation (trenutno scope ovog PR-a samo tools/scripts)

## Notes
- Wiki lifecycle je generički utility — ne ovisi o Agents OS runtime-u.
- Naslov je `chore/` jer je maintenance utility, ne nova funkcionalnost.
