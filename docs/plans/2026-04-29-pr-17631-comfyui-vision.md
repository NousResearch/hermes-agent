# PR #17631 – Worum es geht + Vision für die Lösung

## Kurzfassung
PR #17631 verschiebt den `comfyui`-Skill von:
- `optional-skills/creative/comfyui`

nach:
- `skills/creative/comfyui`

Es ist ein reines `git mv` (kein inhaltlicher Code-/Skill-Change), damit ComfyUI als built-in Skill ausgeliefert wird.

## Was ist das eigentliche Problem?
Nicht die Skill-Logik war falsch, sondern die Einordnung:
- ComfyUI war unter `optional-skills/`,
- sollte aber laut Diskussion zu #17610 bei den built-ins liegen (`skills/creative/`).

Dadurch gab es Inkonsistenz in der Skill-Struktur und potenziell Verwirrung bei Discovery/Erwartung ("ist das jetzt standardmäßig da oder optional?").

## Meine Vision für die "saubere" Lösung
Die Verschiebung in #17631 ist korrekt. Die nachhaltige Lösung ist jetzt:

1) klare Regel im Repo definieren
- Wann gehört ein Skill in `skills/` vs. `optional-skills/`?
- Kriterien dokumentieren (Dependencies, Runtime-Kosten, Zielgruppe, Stabilität).

2) Guardrails in CI einbauen
- Test/Check, der verhindert, dass ein geplanter built-in Skill versehentlich unter `optional-skills/` landet.
- Optional: Mapping-Liste "expected built-ins" + Validierung.

3) Doku synchronisieren
- Alle Referenzen auf den alten Pfad prüfen und korrigieren.
- Migration-Hinweis für bestehende interne Links/Automationen.

4) Smoke-Test für Skill-Discovery
- Sicherstellen, dass `skills_list`/`skill_view` den Skill an erwarteter Stelle finden.
- Besonders wichtig nach reinen Move-PRs.

## Konkreter Vorschlag für Follow-up PR
- `docs/`:
  - kurze Policy "built-in vs optional" ergänzen.
- Skill-Validierung:
  - kleiner Check-Script/CI-Check für erlaubte/verbotene Pfade bestimmter Skills.
- Regression-Schutz:
  - ein Testfall, der bestätigt, dass `creative/comfyui` in built-ins enthalten ist.

## Ergebnisbild (Zielzustand)
- ComfyUI bleibt built-in.
- Die Regel ist dokumentiert.
- CI schützt vor Rückfall.
- Discovery und Doku sind konsistent.

---
Stand: 2026-04-29
Bezug: https://github.com/NousResearch/hermes-agent/pull/17631
