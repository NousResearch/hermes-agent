PR-Titel
chore(skills): codify built-in vs optional placement and add guardrails for ComfyUI path

PR-Body (Entwurf)
## Summary
Follow-up zu #17631: ComfyUI wurde korrekt nach `skills/creative/comfyui` verschoben.
Dieser PR macht die Einordnung dauerhaft robust:
- dokumentiert klare Regeln für `skills/` vs `optional-skills/`
- ergänzt einen Guardrail-Check gegen falsche Skill-Platzierung
- ergänzt einen kleinen Regression-Check für ComfyUI-Discovery

## Why
#17631 hat die akute Inkonsistenz behoben (reiner Move), aber ohne Guardrails kann die gleiche Fehl-Einordnung später wieder passieren.

## Changes
- docs: Policy für Skill-Platzierung ergänzt
- ci/checks: Validierung für erwartete Skill-Pfade ergänzt
- tests: Regression-Check, dass `creative/comfyui` als built-in aufgelöst wird

## Test Plan
- lokale Skill-Checks laufen grün
- neuer Pfad-Guardrail schlägt bei absichtlich falscher Platzierung fehl
- Regression-Test bestätigt `skills/creative/comfyui`

## Backward Compatibility
Keine Laufzeitänderung der Skill-Inhalte, nur Struktur-/Validierungsverbesserung.

## Linked
- follow-up to #17631
- context: #17610

Taskliste für Umsetzung
1) Doku-Policy ergänzen (built-in vs optional)
2) Guardrail-Check-Skript hinzufügen
3) CI um Guardrail erweitern
4) Regression-Test für ComfyUI-Discovery ergänzen
5) kurze Release-Note/Changelog-Zeile ergänzen
