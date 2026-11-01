# PR 2/3: jarvis-stt-tts-contracts

**Branch:** `feat/jarvis-stt-tts-contracts`
**Base:** `main` @ `4d39a60`
**Scope:** Jarvis Mission Control paneli, STT/TTS kontrakti, ideja-factory i SEO panels.
**Source:** Izdvojeno iz #42341 (commits `ad2fdc6f` do `cdab2abc`, 8 commitova).

## Files
- `hermes_cli/agents_os_idea_factory.py` (+151) — Jarvis ideja-factory panel
- `hermes_cli/agents_os_seo.py` (+81) — SEO panel
- Mission Control paneli za Jarvis (ugrađeni u `agents_os_web.py`, +~700 linija)
- `tests/hermes_cli/test_agents_os_complete.py` (+388)
- `tests/hermes_cli/test_agents_os_idea_factory.py` (+86)
- `tests/hermes_cli/test_agents_os_seo.py` (+75)

## Commit sekvenca
1. `ad2fdc6f` feat: add Mission Control Jarvis and SEO panels
2. `cfe8703` feat: add Jarvis command preview scaffold
3. `e4e4376` test: gate Jarvis security scan previews
4. `ca3c18e` feat: add Jarvis STT advisor boundary
5. `c974fdf` feat: add Jarvis local STT adapter
6. `4206fd5` feat: add Jarvis transcript cleanup contract
7. `b55f230` feat: add Jarvis voice reply contract
8. `cdab2ab` feat: add Hume Octave TTS draft contract

## Što radi
- Mission Control paneli: Jarvis command preview, security scan gate, STT advisor + local STT adapter, transcript cleanup + voice reply contract, Hume Octave TTS draft
- Ideja-factory panel: prijedlozi za content/SEO ideje
- SEO panel: page-level SEO snapshot

## Ovisnosti
- PR 1 (control plane) — ova PR gradi **na vrhu** kontrolne ravnine. Ako merge PR 2 prije PR 1, testovi padaju jer importaju `agents_os` runtime module.

**Preporuka:** Merge PR 1 → PR 2 → PR 3.

## Acceptance za maintainere
- `pytest tests/hermes_cli/test_agents_os_complete.py tests/hermes_cli/test_agents_os_idea_factory.py tests/hermes_cli/test_agents_os_seo.py -q` prolazi
- Jarvis paneli u Mission Control-u (port 18790) se renderiraju
- STT/TTS kontrakti imaju contract test fixture (provider=None okruženje)

## Notes
- TTS draft koristi Hume Octave kao draft provider — nije hard dependency. Adapter je pluggable.
- STT advisor boundary čuva sigurnosni gate oko transkripta prije slanja u model.
