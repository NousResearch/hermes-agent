# OpenClaw Runtime Session Forensics

Comment retrouver et prouver un incident conversationnel réel à partir des sessions
runtime OpenClaw — établi lors de CLOE_MEMORY_TOOL_PROTOCOL_AND_LOCAL_TIME_GROUNDING_REPAIR_V1
(incident 2026-08-04 23:19 America/Guadeloupe).

## Où vivent les preuves

- Sessions de l'agent : `~/.openclaw/agents/<agent>/sessions/<sessionId>.jsonl`
  (conversation : rôles, contenu, stopReason, responseId, timestamps, idempotencyKeys).
- Trace par run : `<sessionId>.trajectory.jsonl` (events `session.started`, `trace.metadata`,
  `context.compiled`, `prompt.submitted`, `model.completed`, `trace.artifacts`, `session.ended`).
- Registre des sessions : `~/.openclaw/agents/<agent>/sessions/sessions.json`.
- Config provider : `~/.openclaw/openclaw.json` → `models.providers.cloe.baseUrl`
  (ex. `http://127.0.0.1:4000`, `api: openai-completions`).

## Localiser la session de l'incident

Convertir l'heure locale du transcript (ex. 23:19–23:20 America/Guadeloupe = UTC-4)
→ 03:19–03:20 UTC → chercher les fichiers `*.jsonl`/`*.trajectory.jsonl` dont le mtime
tombe dans cette fenêtre. Le fichier `.jsonl` (conversation) porte le mtime de la
dernière écriture — c'est lui qui matche le plus précisément l'incident.

## Ce que le trajectory prouve (labels de provenance)

| Événement / champ | Preuve |
|---|---|
| `model.completed` → `messagesSnapshot.*.stopReason` | `stop` (pas `tool_calls`) → le markup textuel est du CONTENU, pas un appel natif |
| `messagesSnapshot.*.responseId` | format `chatcmpl-<ts>-<rand>` = la réponse vient du bridge de CE repo |
| `trace.artifacts` → `itemLifecycle.startedCount` / `toolMetas` | 0 = le dispatcheur n'a exécuté AUCUN outil (pas de reinjection, pas de 2e tour) |
| `context.compiled` → `data.tools[].name` | inventaire des tools envoyés (ex. 34 tools incl. `memory_search`) |
| `.jsonl` message `role:"user"` + idempotencyKey | fragment = saisie utilisateur réelle, PAS une corruption (ex. « salt Cloe ») |
| `trace.metadata` → `harness.version`, `model.provider`, `model.modelApi` | version OpenClaw, provider, API |

Utiliser les labels : PROVEN_FROM_RUNTIME / PROVEN_FROM_CODE / PROVEN_FROM_REPRODUCTION /
UNPROVEN_HISTORICAL_DETAIL. Ne pas étiqueter un fragment comme corruption mémoire tant que
sa source n'est pas prouvée (un fragment user-typed est un message utilisateur légitime).

## Règles de lecture

- Toujours redacter secrets/tokens/contenus mémoire privés lors de l'extraction
  (regex sur `sk-*`, Bearer, `*_API_KEY` avant affichage).
- Les sessions sont redactées par le runtime (`redaction.config.mode`), mais le fichier
  brut peut contenir des valeurs sensibles — ne pas les imprimer.
- Comparer la chaîne complète (payload provider → responseId → messagesSnapshot) avant de
  conclure quelle couche a produit le symptôme.

## Piège

Ne pas « normaliser silencieusement » un transcript d'incident : si le transcript montre un
fragment étrange, vérifier dans la session runtime s'il s'agit d'un message utilisateur
(role=user) avant de le traiter comme un artefact système. La correction du transcript
fait partie du diagnostic.
