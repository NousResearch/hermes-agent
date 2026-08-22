# Pattern : Ajouter un nouveau type d'action au pipeline d'exécution gouverné CLOE

## Quand utiliser ce pattern

Quand une mission doit ajouter un nouveau type d'action (ex: `CAMPAIGN_CREATE_PAUSED`, `VARIATION_CREATE_VISIBLE`, etc.) au pipeline d'exécution gouverné CLOE dans `lah-openclaw-mvp`.

## Chaîne complète (6 points de modification)

### 1. Execution policy — `src/services/execution-policy.js`

Ajouter l'action à `ALLOWED_ACTION_TYPES[]` (validation gate) ET `IMPLEMENTED_ACTION_TYPES` (dispatch set) :

```javascript
export const ALLOWED_ACTION_TYPES = Object.freeze([
  // ...
  'MON_NOUVEAU_TYPE',
]);

export const IMPLEMENTED_ACTION_TYPES = new Set([
  // ...
  'MON_NOUVEAU_TYPE',
]);
```

### 2. Validation — `src/services/executor.js` — `validateExecutePayload()`

Ajouter les champs target requis pour le nouveau type :

```javascript
if (action_type === 'MON_NOUVEAU_TYPE') {
  const required = ['champ1', 'champ2', 'champ3'];
  for (const field of required) {
    if (!target[field]) errors.push(`target.${field} is required for ${action_type}`);
  }
}
```

### 3. Implémentation — `src/services/executor.js` — `executeAction()`

Ajouter le handler d'exécution. Respecter les invariants suivants :

```javascript
if (action_type === 'MON_NOUVEAU_TYPE') {
  // Appel API ExoClick ou autre provider
  const result = await maFonction(payload.target);
  
  // Normaliser la réponse
  const normalized = normalizeExoClickCampaign(result, payload.action_id);
  
  // Construire le reçu
  return buildExecuteReceipt({
    action_id: payload.action_id,
    status: normalized.ok ? 'SUCCESS' : 'PROVIDER_ERROR',
    provider: 'exoclick',
    live_sent: normalized.live_sent,
    provider_campaign_id: normalized.provider_campaign_id,
    execution: { action_type, ...normalized.metadata },
  });
}
```

**Règle live gate** : Si le type d'action doit contourner le live gate (comme `CAMPAIGN_CREATE_PAUSED`), ajouter l'exemption :

```javascript
// Dans la fonction qui vérifie le live gate
if (action_type === 'MON_NOUVEAU_TYPE') {
  // Commentaire expliquant pourquoi ce type est exempté
} else if (process.env.EXOCLICK_LIVE_ENABLED !== 'true') {
  return { ok: false, status: 'DRY_RUN_BLOCKED', ... };
}
```

### 4. Détection d'intention — `src/services/cloe-governed-action-packet.js`

Ajouter le pattern de détection dans `detectSpecificActionType()` :

```javascript
if (/(?:pattern|regex|keywords)/i.test(lower)) {
  return 'MON_NOUVEAU_TYPE';
}
```

### 5. Execute-adapter — `src/services/cloe-governed-action-execute-adapter.js`

**Cinq choses à modifier :**

**a.** Ajouter à `SUPPORTED_ACTION_TYPES` :

```javascript
const SUPPORTED_ACTION_TYPES = Object.freeze([
  'CAMPAIGN_PAUSE',
  'CAMPAIGN_PLAY',
  'MON_NOUVEAU_TYPE',  // ← ajout
]);
```

**b.** Ajouter la validation des champs target (section 6 de `buildExecutePayloadFromGovernedAction`) :

```javascript
if (actionType === 'MON_NOUVEAU_TYPE') {
  const required = ['champ1', 'champ2'];
  const missing = required.filter(f => !target?.[f]);
  if (missing.length > 0) {
    targetError = `MON_NOUVEAU_TYPE target requires: ${missing.join(', ')}`;
  }
}
```

**c.** Ajouter la construction du payload (section "Build canonical execute payload") :

```javascript
if (packet.action_type === 'MON_NOUVEAU_TYPE') {
  const fields = ['champ1', 'champ2', 'champ3'];
  for (const field of fields) {
    if (packet.target?.[field] !== undefined && packet.target[field] !== null) {
      payload.target[field] = packet.target[field];
    }
  }
}
```

**d.** Mettre à jour la JSDoc des checks fail-closed (étape 6).

**e.** Si le type nécessite un traitement spécial pour `approved_by_human` ou `approval_id`, ajuster.

### 6. Orchestrateur/point d'entrée (optionnel)

Si le nouveau type fait partie d'un flux conversationnel multi-tours, ajouter l'intégration dans l'orchestrateur concerné.

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Oubli SUPPORTED_ACTION_TYPES** | `buildExecutePayloadFromGovernedAction` retourne `UNSUPPORTED_ACTION_TYPE` bien que l'executor supporte l'action | Vérifier les DEUX listes : `ALLOWED_ACTION_TYPES` dans execution-policy.js ET `SUPPORTED_ACTION_TYPES` dans l'adapter |
| **target.campaign_id requis par défaut** | L'ancien code avait `if (!target.campaign_id)` en dur. Les nouveaux types peuvent ne PAS avoir de campaign_id | Remplacer le check blanket par un `if/else` action-type-aware (voir section 5b) |
| **JSDoc pas à jour** | Les commentaires JSDoc listent `CAMPAIGN_PAUSE, CAMPAIGN_PLAY` mais pas le nouveau type | Mettre à jour la JSDoc de `buildExecutePayloadFromGovernedAction` |
| **Payload builder trop minimal** | Le payload ne propage que `campaign_id`. Un nouveau type a besoin de champs différents | Ajouter un bloc conditionnel pour propager les champs spécifiques (section 5c) |
| **validateCompleteness dans le packet** | `validatePacketCompleteness()` dans `cloe-governed-action-packet.js` peut aussi avoir un check blanket `campaign_id` | Vérifier et rendre action-type-aware si nécessaire |

## Exemple concret : `CAMPAIGN_CREATE_PAUSED`

Voir le commit `7a36a83` sur `openclaw-runtime` pour l'implémentation complète.

Résumé des modifications pour `CAMPAIGN_CREATE_PAUSED` :

| Fichier | Changement |
|---------|------------|
| `execution-policy.js` | Déjà présent (`ALLOWED_ACTION_TYPES` + `IMPLEMENTED_ACTION_TYPES`) |
| `executor.js` | Déjà présent (`validateExecutePayload` + `executeAction` handler + live gate exemption) |
| `cloe-governed-action-packet.js` | `detectSpecificActionType` retourne `CAMPAIGN_CREATE_PAUSED` sur pattern "create"/"créer"/"nouveau" |
| `cloe-governed-action-execute-adapter.js` | Ajout à SUPPORTED_ACTION_TYPES + validation 9 champs target + payload builder propagation |
