# Provider-Boundary Crash-Window Proof Pattern

## Quand l'utiliser

Quand un système effectue des mutations chez un provider externe (API HTTP, base de données, service réseau) et que le processus peut crasher entre l'envoi de la requête et la persistance du reçu terminal. Ce pattern résout le problème de classification correcte : l'état du provider est ambigu après le crash.

## Le problème

```
CLAIMED → EXECUTING → [APPEL PROVIDER] → EXECUTED/FAILED (persistance)
                            ^                              ^
                    crash possible ici         crash possible ici aussi
```

Si le processus crashe après que le provider a accepté la mutation mais avant que EXECUTED soit persisté :

- Le reçu local reste en EXECUTING
- Au redémarrage, recoverOrphans() le marque FAILED_ORPHANED
- Le système pense que la mutation a échoué — faux négatif
- Le système pourrait relancer la mutation — double mutation

## La solution en 4 parties

### Partie A — Champs d'evidence provider-boundary

Avant l'appel provider, persister des timestamps dans le reçu :

| Champ | Timing | Responsable | Utilité |
|-------|--------|-------------|---------|
| provider_request_started_at | Avant toute préparation | Executor | Preuve que la tentative a commencé |
| provider_request_dispatched_at | Juste avant `fetch()` | **Adapter** (via `provider_dispatch_at`) | Preuve réelle qu'une requête est partie |
| provider_response_received_at | Après la réponse provider | Executor | Preuve que le provider a répondu |
| provider_operation | Avant l'appel | Executor | Type d'opération |
| expected_provider_state | Avant l'appel | Executor | État attendu après mutation |
| reconciliation_required | Après crash/exception | Executor | Flag pour le moteur de réconciliation |
| reconciliation_reason | Après crash/exception | Executor | Raison (ex: PROVIDER_CALL_EXCEPTION) |

### Règle critique : le timestamp de dispatch appartient à l'adapter

Le `provider_request_dispatched_at` ne doit PAS être défini dans l'executor avant d'appeler l'adapter. Sinon, un échec pré-dispatch (token manquant, validation) laisse un faux timestamp dans le reçu, ce qui fait croire à orphan recovery que le provider a été contacté.

**Où le mettre** : l'adapter définit un timestamp `provider_dispatch_at` juste avant `fetch()` et le retourne dans le résultat structuré. L'executor le persiste dans le reçu seulement si présent.

**Signature de l'adapter** :

```javascript
async function providerAdapter(input) {
  // Pre-dispatch checks (no throw — return structured errors)
  if (!token) return { ok: false, dispatch_attempted: false, verdict: 'MISSING_TOKEN', ... };

  // Dispatch boundary — timestamp set right before the network call
  const providerDispatchAt = new Date().toISOString();
  try {
    const res = await fetch(endpoint, { ... });
    return { ok: res.ok, dispatch_attempted: true, provider_dispatch_at: providerDispatchAt, live_sent: true, ... };
  } catch (err) {
    return { ok: false, dispatch_attempted: true, provider_dispatch_at: providerDispatchAt, live_sent: true, ... };
  }
}
```

### Partie B — Classification des échecs (sur preuve réelle)

```
Pre-dispatch failure (dispatch_attempted=false)
→ FAILED
→ provider_write_attempted=false
→ provider_request_dispatched_at: absent
→ reconciliation_required=false
→ error code: MISSING_TOKEN, MISSING_CAMPAIGN_IDS, LOGIN_FAILED, etc.

Explicit provider rejection (HTTP response with status_code)
→ FAILED
→ provider_write_attempted=true
→ live_sent=true
→ provider_request_dispatched_at: present
→ provider_response_received_at: present
→ reconciliation_required=false
→ error code: EXOCLICK_API_ERROR / EXOCLICK_REJECTED

Ambiguous transport (timeout, reset, socket hangup, no status_code)
→ UNKNOWN_OUTCOME
→ provider_write_attempted=true
→ live_sent=true
→ provider_request_dispatched_at: present
→ provider_response_received_at: absent
→ reconciliation_required=true
→ error code: PROVIDER_TRANSPORT_AMBIGUOUS

Confirmed success (HTTP 2xx)
→ EXECUTED
→ provider_write_attempted=true
→ provider_request_dispatched_at: present
→ provider_response_received_at: present
→ reconciliation_required=false
```

### Partie C — Réconciliation read-only

```
async function reconcileExecutionOutcome(receipt, deps):
  1. Vérifier éligibilité (UNKNOWN_OUTCOME uniquement)
  2. Vérifier type d'action supporté
  3. Lire l'état provider (GET uniquement, jamais POST/PUT)
  4. Comparer état observé vs état attendu
  5. Si match → RECONCILED_EXECUTED, mise à jour du reçu
  6. Si mismatch → RECONCILED_NOT_EXECUTED ou STILL_UNKNOWN
  7. Jamais de mutating provider call
  8. Jamais de second execution receipt
```

### Partie D — Récupération des orphelins

Dans recoverOrphans(), classifier selon la présence d'evidence provider :

```
if (receipt.provider_request_dispatched_at || receipt.provider_request_started_at)
  → UNKNOWN_OUTCOME, reconciliation_required = true
else
  → FAILED_ORPHANED (sûr)
```

## Preuve hermétique du crash-window

### Architecture de la preuve

```
LAHB Mock (approvals)    Provider Sentinel (mutation+read)    Child Process (real code)
  127.0.0.1:N              127.0.0.1:M                       
       │                        │                              │
       └────────────────────────┼──────────────────────────────┘
                                │
                         [receipts.json]
                         (fichier partagé)
```

### Scénario de preuve

1. Démarrer LAHB mock + Provider sentinel (compteurs mutations/lectures)
2. Lancer un processus enfant qui utilise les vrais modules de production
3. L'enfant : CLAIM → EXECUTING → evidence provider → appelle provider
4. Le sentinel accepte la mutation (compteur +1) et change l'état
5. L'enfant écrit un signal "mutation acceptée" puis crash (process.exit(0))
6. Aucun EXECUTED persistance — le reçu reste EXECUTING
7. Le parent importe recoverOrphans() — détecte l'orphelin
8. Vérifie que l'orphelin est classé UNKNOWN_OUTCOME (pas FAILED_ORPHANED)
9. Le parent importe reconcileExecutionOutcome() — lit l'état provider
10. Vérifie que le reçu devient EXECUTED après réconciliation
11. Vérifie que le compteur de mutations provider = 1 (pas de duplication)

### Pièges connus

| Piège | Symptôme | Fix |
|-------|----------|-----|
| Chemin du script enfant | Le import() dans le child process échoue car __dirname est relatif au script, pas au repo | Utiliser un script inline avec des chemins absolus, ou écrire le script dans un fichier temp avec des imports corrects |
| Format de réponse mock provider | verifyExoClickCampaign reçoit PROVIDER_UNKNOWN car le format est imbriqué (data.campaign.status) mais le code attend data.status | Vérifier que data.data (ou le resolved object) a status à la racine |
| Timing des orphelins | recoverOrphans() ne trouve pas l'orphelin car claimed_at est trop récent | Régler EXECUTION_ORPHAN_THRESHOLD_MS très bas (50ms) et forcer claimed_at à une date passée |
| Module-level env constant | const X = process.env.X || DEFAULT bloque l'isolation de test | Remplacer par un getter: function getX() { return process.env.X || DEFAULT; } |

## Intégration dans le workflow LAH

Ce pattern s'applique à la Phase 5 (Implementation) des missions MIXED, spécifiquement à la Phase C (Runtime Proof). Il est le complément des patterns de preuve HTTP route et cross-process.

Références connexes : http-route-runtime-proof-pattern.md, cross-process-proof-pattern.md, provider-bridge-propagation-proof-pattern.md
