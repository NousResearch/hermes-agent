# Provider Raw Tool-Markup Protection + Temporal Grounding Pattern

Établi lors de CLOE_MEMORY_TOOL_PROTOCOL_AND_LOCAL_TIME_GROUNDING_REPAIR_V1 (2026-08-05),
incident réel du 2026-08-04 23:19 America/Guadeloupe (session OpenClaw `4f646a9e-*`).

## La classe de panne

Un provider (DeepSeek ici) renvoie du **markup de contrôle d'outil textuel** dans
`message.content` au lieu de `message.tool_calls` natifs, avec `finish_reason="stop"` :

```json
{"message": {"role": "assistant", "content": "<tool_calls>\n<invoke name=\"memory_search\">\n<parameter name=\"query\" string=\"true\">récent</parameter>\n</invoke>\n</tool_calls>"}, "finish_reason": "stop"}
```

Conséquences observées en chaîne :
1. Le bridge mappe `content` verbatim → l'utilisateur VOIT le markup brut (réponse `chatcmpl-<ts>-<rand>` = format du bridge).
2. `stopReason=stop`, pas de `tool_calls` natif → le dispatcheur OpenClaw n'exécute RIEN (`itemLifecycle.startedCount=0`, `toolMetas=[0]`) → pas de reinjection de résultat d'outil → pas de second tour provider → session terminée sur le markup (aucune réponse naturelle finale).

## Invariant obligatoire

`NO_RAW_TOOL_CONTROL_MARKUP_CAN_REACH_USER_VISIBLE_ASSISTANT_CONTENT`

## Détection : détecteur STRUCTUREL étroit, pas un ban de sous-chaîne

Un texte bénin mentionnant « DSML » ou des balises HTML isolées ne doit PAS être bloqué.
Exiger la structure imbriquée :

```
hasToolCallsOpener = /<\s*tool_calls\s*>/i
hasInvoke          = /<\s*invoke\b[^>]*\bname\s*=\s*["'][^"']+["']/i
hasParameter       = /<\s*parameter\b[^>]*\bname\s*=\s*["'][^"']+["']/i
hasClosers         = /<\s*\/\s*(?:tool_calls|invoke)\s*>/i
match = (hasToolCallsOpener && (hasInvoke || hasClosers))
     || (hasInvoke && (hasParameter || hasClosers))
```

Un seul tag isolé (`<tool_calls>` seul, `<b>bold</b>`) ne matche PAS. « DSML » sans balise ne matche PAS.

## Réparation (fail-closed, jamais d'exécution inventée)

- **Markup brut + PAS de tool_calls natif** → remplacer `content` par un fallback sûr déterministe
  (jamais le markup), poser `_cloe.protocol_failure = { code: 'PROVIDER_RAW_TOOL_MARKUP', exposed_raw_markup: false }`.
- **Markup brut + tool_calls natifs présents** → le call structuré natif reste autoritaire ;
  le content pollué est droppé à `null` (jamais affiché), `finish_reason` conservé, même code observé.
- **Jamais** fabriquer de `tool_calls` à partir du texte (augmenterait le risque d'injection).
- **Garde terminale** : `content` null/vide + `finish_reason=stop` + pas de tool call →
  fallback déterministe + `_cloe.terminal_fallback = { code: 'PROVIDER_EMPTY_TERMINAL_CONTENT' }`
  (un tour « stop » ne doit jamais être silencieusement vide).
- Appliquer l'invariant aux DEUX points d'entrée : bridge natif (`buildOpenAiResponse`)
  ET chemin non-tool (`buildBrainAskResponse` → mapping `answer`).
- Ne PAS construire un convertisseur permissif texte→tool_calls sauf preuve contractuelle officielle.

## Temporal grounding (Phase F) — contexte serveur déterministe

Le modèle inventait jour/heure (aucun `Intl.DateTimeFormat`, aucun IANA dans les payloads).

- Module pur : `buildTemporalAuthorityContext({ now, timezone, env })` via `Intl.DateTimeFormat`
  IANA + `formatToParts` (weekday + `timeZoneName:'longOffset'` → `-04:00`).
- **Horloge injectable** (`now`) pour tests déterministes F1–F3 ; zone canonique `America/Guadeloupe`,
  override env `CLOE_LOCAL_TIMEZONE`.
- Zone invalide → fallback sûr `UTC` + code observable `CLOE_TZ_INVALID_FALLBACK` (pas de throw).
- Marqueur `CLOE_TEMPORAL_AUTHORITY` dans le bloc → anti-duplication F5 (ne pas ré-injecter
  quand un system message de l'historique porte déjà le marqueur).
- Injecter dans les DEUX payloads : message system préfixé (chemin tool) ET préfixe de
  `systemContent` (chemin non-tool). Bloc :

```
Current temporal authority:
* UTC datetime: <ISO>
* Local datetime: <YYYY-MM-DDTHH:MM:00±HH:MM>
* Timezone: <IANA>
* Local weekday: <Weekday>
Instruction: use ONLY this authority ... Do not guess a different date/time ...
<CLOE_TEMPORAL_AUTHORITY_MARKER>
```

## Vérification (shape éprouvée)

- Fixtures B1–B7 : tool call natif préservé / DSML brut jamais visible / mixte malformé
  (natif autoritaire, markup masqué) / mention « DSML » inoffensive non bloquée /
  cycle 2e tour complet (IDs tool_call appariés, pas de 3e loop) / résultat d'outil vide ou
  malformé → réponse honnête / continuation ambiguë (« oui comme toujours ») → réponse complète.
- Garde terminale E : null/empty + stop → fallback.
- Temporel F1–F6 (dont boundary minuit qui change le weekday, F5 non-dup).
- Sécurité G : markup absent de tout content JSON retourné, arguments jamais exécutés depuis
  le texte, texte utilisateur ressemblant à du DSML ne déclenche pas d'exécution.
- **Phase H live-shape non-mutante** : mock provider DeepSeek local + serveur réel du worktree
  sur port éphémère (PORT=4111, ADMIN_API_KEY de test), replay du format OpenClaw (34 tools
  incl. `memory_search`) — prouve le chemin HTTP réel sans déploiement ni provider payant.

## Variante obfusquée (2026-08-05, déploiement b39c2a8) — BYPASS CONNU du détecteur

Le provider (deepseek-v4-flash) a émis une variante ÉVASIVE du même DSML, en insérant
des paires de FULLWIDTH REVERSE SOLIDUS + infixe « DSML » dans chaque balise :

```
<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>
<\uff5c\uff5cDSML\uff5c\uff5cinvoke name="memory_search">
...
```

- `detectRawToolControlMarkup` retourne `false` sur cette forme : les regex
  `<\s*tool_calls\s*>` etc. ne matchent pas `<＼＼DSML＼＼tool_calls>` (chars : `3c ff5c ff5c 44 53 4d 4c ff5c ff5c ...`).
- Prouvé : serveur réel du SHA déployé + mock provider → le markup passe en JSON ET en SSE,
  aucun fallback, `_cloe.protocol_failure: null`. Observé en production (session store msg[7],
  run b6a56b5a, 1 `model.completed`, 0 tool event).
- Classification honnête : DEPLOYED_AND_LIVE_VERIFIED_WITH_EXTERNAL_RUNTIME_LIMITATION
  (le déploiement est sain ; le guard couvre la forme classique ; la variante obfusquée est
  un gap de couverture → mission de réparation séparée, PAS un rollback : l'ancien SHA n'avait
  aucun guard). Réparation recommandée : normaliser/stripper la famille obfusquée (U+FF5C,
  séparateurs doublés, infixe DSML) dans le chemin JSON ET le chemin SSE, avec tests des deux.
- **Trap d'inspection** : ne JAMAIS écrire la chaîne littérale `<invoke name="memory_search">`
  dans un argument d'outil (write_file/terminal/grep) — le harnais l'interprète comme un appel
  d'outil indisponible (« Tool 'memory_search' does not exist ») et casse le run. Construire
  les chaînes par concaténation/codes (`'inv'+'oke'`, `'\uFF5C'`) ; lire le JSONL en Python ;
  grep avec fragments (`tool_calls|invoke` sans chevrons).

Détail complet + séquence de déploiement manuel quand le deployer est bloqué par image prunée :
`references/cloe-tool-markup-obfuscation-and-deploy-fallback.md`.

## Piège test-harness rencontré

Nouveau script standalone `.mjs` dans `test/` (sans suffixe `.test.`) → le test
`node-test-discovery-boundary.test.js` le signale comme « unexpected » SAUF s'il est ajouté à
`STANDALONE_TEST_SCRIPTS` dans `tools/ci/node-test-discovery.mjs`. Ne pas le renommer en
`.test.mjs` (il démarre des serveurs et fait `process.exit`). Même mécanique que
`cloe-end-to-end-certification.mjs` etc.

## Phase I — Single-retry recovery synthesis (2026-08-07, CLOE_POST_TOOL_RAW_DSML_SINGLE_RETRY_RECOVERY_V1, deploy 5404046)

Le fallback déterministe empêchait la fuite mais perdait la réponse. Extension bornée :
**exactement UN appel de recovery** quand la réponse post-tool est HTTP/JSON OK, zéro
tool_call natif, markup DSML détecté. Payload recovery : contexte déjà groundé +
directive finale explicite (`injectProtocolRecoveryDirective` : « Produce the final
user-facing answer only. Do not emit tool calls, DSML, XML-like control markup… »),
`tools: undefined`, `tool_choice: 'none'`, `parallel_tool_calls: false`, budget post-tool
résolu (`resolveEffectiveTimeoutMs` + `planDeadlines` — 70s par défaut, overrides gagnent).
`MAX_PROTOCOL_RECOVERY_RETRIES = 1` — aucun loop, aucune récursion, aucun 2e retry.
Tout échec recovery (timeout/HTTP/JSON/markup/tool_calls/shape) → fallback déterministe
existant ; classification `PROVIDER_RAW_TOOL_MARKUP` préservée via `_cloe.protocol_failure`.
Observabilité : `_cloe.protocol_recovery` (provider_round_type, effective_timeout_ms,
provider_duration_ms, raw_tool_markup_detected, raw_tool_markup_kind,
native_tool_calls_count, protocol_recovery_attempted/success/duration_ms,
protocol_recovery_failure_code, final_fallback_used). L'éligibilité est restreinte au
round post-tool — les rounds standard gardent le fallback immédiat (préserve les tests B2
existants).

### Pitfall 1 (critique) — juger la propreté du recovery sur le provider value BRUT, AVANT buildOpenAiResponse
`buildOpenAiResponse` convertit déjà un 2e payload markup en fallback déterministe (ou
null) : juger la propreté APRÈS le builder fait passer l'échec pour un succès
(`protocol_recovery_success=true` sur du markup — attrapé par le test T3). Vérifier sur
`recoveryResult.value.choices[0].message` (content + tool_calls) avant d'appeler le builder.

### Pitfall 2 — classifyRawToolMarkupKind doit d'abord gater sur detectRawToolControlMarkup
Sans le gate, `classifyRawToolMarkupKind('plain text')` retourne `'classic_dsml'` au lieu
de `null`. Gate : `if (!detectRawToolControlMarkup(content)) return null;` puis classer
`classic_dsml` vs `obfuscated_unicode` (fullwidth U+FF5C/U+FF3C ou infixe DSML).

### Pitfall 3 — patch-tool double-échappement des chaînes `\u2019`
En patchant des chaînes contenant des séquences unicode (`'Je n\u2019ai…'`), le patch
peut doubler le backslash (`\\u2019` réel) : la chaîne passe au rendu mais l'apostrophe
courbe devient littérale. Vérifier après chaque patch : `node --check` + import runtime
affichant `JSON.stringify(EMPTY_TERMINAL_FALLBACK)`.

### Harness réel (REAL PROVIDER PROOF) — boucle round-par-round obligatoire
- S1 contrôle post-tool : une synthèse réelle propre → ok, zéro fallback, budget 70s.
- S2 replay de la mission d'origine : boucle VRAIE round-par-round (le premier appel
  renvoie des tool_calls de décision, PAS la réponse finale — un appel one-shot échoue
  avec `tool_calls>0, answer_chars=0`). Pousser assistant tool_calls + résultats
  `role:'tool'` (tous les appels parallèles dans UN message assistant, un result par
  call), rappeler jusqu'à réponse finale (max rounds borné). Mocker UNIQUEMENT
  l'exécution d'outil read-only (memory_search → JSON de registre réaliste), garder le
  provider réel.
- Preuve de non-régression full-suite : lancer le sous-ensemble échouant sur un worktree
  baseline même SHA SANS le patch, diff des noms de tests échoués (delta 0 = pré-existant).
