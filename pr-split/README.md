# PR Split Plan — #42341 → 3 PR-a

## Izvor
- PR: https://github.com/NousResearch/hermes-agent/pull/42341
- Title: feat: add local Agents OS Mission Control panels
- Branch: `feat/agents-os-runtime-control-plane`
- Commits: 11
- Files: 17 (promijenjenih)
- Razlog split-a: scope drift, preširok za jedan PR, velika šansa za odbijanje.

## PR Split

| # | Branch | Scope | Commits | Files | Merge redoslijed |
|---|---|---|---|---|---|
| 1 | `feat/agents-os-runtime-control-plane-clean` | Originalni control plane + današnji fix | 2 | 9 | Prvi |
| 2 | `feat/jarvis-stt-tts-contracts` | Jarvis/STT/TTS scope | 8 | 6 | Drugi (zavisan o 1) |
| 3 | `chore/wiki-lifecycle-daily-helpers` | Wiki restore | 1 | 2 | Paralelno s 1 i 2 |

## Prednosti split-a
- Lakši review (max 9 fajlova / 8 commit-ova po PR-u)
- Veća šansa za merge
- Mogu ići (djelomično) paralelno — PR 3 je neovisan
- Maintaineri mogu odbiti dio bez da odbace cijeli set
- Lakše pratiti regression ako PR 2 padne

## Akcijski plan (za sljedeću sesiju kad se token oporavi)
1. Klonirati PR branch lokalno: `git fetch origin pull/42341/head:pr-42341`
2. Za svaki PR:
   - Kreirati branch iz `main`
   - Cherry-pick odgovarajuće commitove iz `pr-42341`
   - Za PR 1: ručno izdvojiti samo `agents_os.py` / `agents_os_tui.py` / `agents_os_web.py` (fix) / `commands.py` / `main.py` / launch script + njihove testove; **NE uključivati** `agents_os_idea_factory.py` i `agents_os_seo.py`
   - Za PR 2: cherry-pick 8 Jarvis/STT/TTS commitova
   - Za PR 3: cherry-pick 1 wiki commit
3. Pushati 3 grane na `goran1mikac-ux/hermes-agent` fork
4. Otvoriti 3 PR-a prema `NousResearch/hermes-agent:main`
5. U svakom PR opisu linkati na #42341 i obrazložiti zašto je izdvojen
6. Zatvoriti #42341 s komentarom "Superseded by #X, #Y, #Z"

## Artefakti
- PR1 opis: `./PR1-control-plane.md`
- PR2 opis: `./PR2-jarvis-stt-tts.md`
- PR3 opis: `./PR3-wiki-lifecycle.md`

## Token-used pri splitu
- Split opis kreiran: ~52/50k (over budget ali radnja završena)
- GitHub push i PR kreacija: odgođeno za sljedeću sesiju
