# Hermes Production Prompt Pack

Package minimal pour brancher un agent Hermes orienté `function calling` sur un runtime orchestré.

## Contenu

- `prompts/system_prompt.production.md`
  Prompt système opérationnel orienté supervision, sources primaires, outil obligatoire et blocage.
- `contracts/tool_contracts.v1.json`
  Registre machine-lisible des outils, avec contrats d'arguments, enveloppes de résultat et règles de déclenchement.
- `policies/escalation_policy.v1.json`
  Politique d'escalade exécutable en style JSON Logic.
- `schemas/final_response.schema.json`
  Schéma JSON de la réponse finale validable côté orchestrateur.

## Intégration minimale

1. Charger `prompts/system_prompt.production.md` comme system prompt après substitution des variables runtime.
2. Enregistrer les outils exposés au modèle à partir de `contracts/tool_contracts.v1.json`.
3. Avant chaque réponse finale, construire un `policy_context` puis évaluer `policies/escalation_policy.v1.json`.
4. Si une règle déclenche `stop_generation=true`, appeler l'outil d'escalade requis et interdire toute conclusion finale.
5. Valider la réponse finale avec `schemas/final_response.schema.json`.

## Variables runtime attendues

- `{{JURISDICTION}}`
- `{{DEFAULT_CURRENCY}}`
- `{{SEUIL_ESCALADE_MATERIALITE_EUR}}`
- `{{DELAI_ESCALADE_ECHEANCE_JOURS}}`
- `{{DATE_SYSTEME}}`
- `{{OUTIL_*}}` si le runtime renomme les fonctions exposées

## Hypothèse d'orchestration

Le runtime conserve pour chaque tour :

- `classification`
- `facts`
- `tool_results`
- `source_verification`
- `risk_flags`
- `user_identity_context`
- `execution_permissions`

Sans cet état structuré, la politique d'escalade ne peut pas être évaluée correctement.
