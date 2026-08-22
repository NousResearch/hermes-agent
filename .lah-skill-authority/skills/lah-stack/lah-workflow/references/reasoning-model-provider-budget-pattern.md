# Reasoning Model Provider Budget Pattern

Quand le provider LLM utilise un modèle à raisonnement (ex: DeepSeek v4 Flash, Claude Sonnet Reasoning), le budget de tokens doit tenir compte du fait que **le modèle consomme des tokens en `reasoning_content` avant d'émettre `content`**.

Sans séparation explicite, le budget `max_tokens` peut être entièrement consommé par le raisonnement, laissant 0 tokens pour le JSON de sortie.

## Modèle à trois niveaux

Utiliser trois budgets distincts dans `WorkerBudgetSnapshot` :

```
max_provider_completion_tokens   # (ex: 1536) Budget total de génération
                                 # envoyé comme max_tokens au provider
                                 # couvre raisonnement + sortie

max_provider_output_tokens       # (ex: 512)  Budget attendu du JSON canonique
                                 # uniquement pour la sortie, pas envoyé au provider

max_provider_response_bytes      # (ex: 8192) Plafond dur de la réponse HTTP
                                 # toujours cohérent avec completion_tokens
```

### Règle de cohérence

```
max_provider_response_bytes >= max_provider_completion_tokens × 4 + ~500 (enveloppe JSON)
```

## Classification des réponses

`finish_reason` et `content` déterminent ensemble la classification. **Toujours inspecter le premier `choice` avant d'appeler `_unwrap_chat_completion_payload`**.

| HTTP status | finish_reason | content | reasoning_content | Classification |
|------------|--------------|---------|-------------------|---------------|
| 200 | stop | non-vide | * | `SUCCEEDED` — normal |
| 200 | length | vide | non-vide | `OUTPUT_TRUNCATED` — budget épuisé par le raisonnement |
| 200 | length | vide | vide | `OUTPUT_TRUNCATED` — budget épuisé sans raisonnement |
| 200 | stop | vide | * | `SCHEMA_INVALID` — réponse malformée (≠ exhaustion) |
| 200 | length | vide | * + HTTP body > 8192 | `UNKNOWN_OUTCOME` — réponse oversized (catch séparé) |
| 4xx/5xx | * | * | * | `REJECTED` / `RATE_LIMITED` / `UNAVAILABLE` |

## Circuit breaker

`OUTPUT_TRUNCATED` est une **soft failure** : elle n'ouvre PAS le circuit breaker.

- Ce n'est pas une erreur de transport (le provider a répondu)
- Ce n'est pas une erreur de schéma (le format de réponse est correct)
- Le breaker la traite comme un état « neutre » — rien n'est incrémenté

## Propagation au diagnostic

Le `rejection_reason` doit contenir une chaîne canonique comme `"PROVIDER_COMPLETION_EXHAUSTED: finish_reason=length, empty content"`. L'orchestrateur peut ainsi distinguer :

- Complétion épuisée (`OUTPUT_TRUNCATED`)
- Provider injoignable (`UNAVAILABLE`)
- Réponse trop volumineuse (`UNKNOWN_OUTCOME`)
- Schéma invalide (`SCHEMA_INVALID`)
- Budget d'appels épuisé (`UNKNOWN_OUTCOME` + `"worker provider budget exhausted"`)

## Métadonnées préservées

Même en cas d'`OUTPUT_TRUNCATED`, préserver :

- `provider` (nom du provider)
- `provider_identifier`
- `model_id` (du config, pas du provider — jamais faire confiance au champ `model` de la réponse)
- `endpoint`
- `request_id` (du champ `id` de la réponse JSON)
- `status` = `OUTPUT_TRUNCATED`

Ne jamais persister `reasoning_content` ni `content` brut.

## Multi-turn tool loop : DeepSeek thinking-mode EXIGE le retour de `reasoning_content`

Observé 2026-08-05 (production smoke tests PR #701, CLOE_RETRIEVAL_TOOLS_PRESERVATION_PR701_END_TO_END_MERGE_DEPLOY_AND_PRODUCTION_CERTIFICATION_V1). Le pattern ci-dessus (« ne jamais persister reasoning_content ») vaut pour les diagnostics single-turn. En multi-turn avec un modèle thinking (deepseek-v4-flash), la règle est INVERSE au niveau de l'API :

- **Symptôme** : le premier tour émet des `tool_calls` structurés (finish_reason=tool_calls). Le second tour (historique + message `tool` résultat) est rejeté par DeepSeek avec `HTTP 400 invalid_request_error` : `The reasoning_content in the thinking mode must be passed back to the API.`
- **Preuve d'isolation** : l'appel DIRECT à l'API DeepSeek (curl, sans passer par le service) reproduit le même 400 — ce n'est pas un défaut du bridge/adapter, c'est une exigence du provider.
- **Contournement de diagnostic** : ajouter `"thinking": {"type": "disabled"}` au payload du second tour → HTTP 200 + synthèse finale. Confirme que le 400 vient du mode thinking.
- **Conséquence architecturale** : le runtime qui gère la continuation (ex: boucle tool OpenClaw) doit soit (a) stocker et re-injecter `reasoning_content` du tour précédent dans le message assistant du tour suivant, soit (b) désactiver le thinking. Ne pas patcher le bridge /chat/completions pour « réparer » ce 400 — c'est une frontière runtime, pas une régression du service.
- **Trap de classification** : un second tour qui 400 avec ce message ne doit PAS être diagnostiqué comme « tools perdus » ou « réponse cassée » — vérifier d'abord le message d'erreur exact du provider avant de conclure.

## Pièges

| Piège | Symptôme | Correctif |
|-------|----------|----------|
| **Tous les tokens consommés par reasoning** | `content: ""`, `finish_reason: length`, réponse 200 | Augmenter `max_provider_completion_tokens` — pas `max_provider_output_tokens` |
| **Budget output = budget completion** | Les tests échouent car le modèle reasoning consomme tout | Toujours avoir `completion_tokens > output_tokens` avec marge pour le raisonnement |
| **Classification comme UNAVAILABLE** | Provider injoignable ET réponse vide classifiés pareil | Détecter `finish_reason=length` AVANT l'unwrap — statut distinct `OUTPUT_TRUNCATED` |
| **reasoning_content persisté** | Fuite de contenu brut via diagnostics | Vérifier que `reasoning_content` n'apparaît nulle part dans `ProviderMetadata` ou `ProviderCallResult` |

## Utilisation dans la stack

Établi lors de `LOT_F_REASONING_TOKEN_BUDGET_AND_EMPTY_CONTENT_CLASSIFICATION_CLOSURE_V1` (commit 778775d, worktree `cloe-diagnostic-orchestrator-lot-f`).

Implémentations :
- `WorkerBudgetSnapshot` dans `worker/models.py` (champ `max_provider_completion_tokens`)
- Détection early dans `DeepSeekProviderClient.invoke()` dans `providers/transport.py`
- Envoi de `completion_tokens` comme `max_tokens` dans `IntelligentProposalService.propose()` dans `worker/service.py`
