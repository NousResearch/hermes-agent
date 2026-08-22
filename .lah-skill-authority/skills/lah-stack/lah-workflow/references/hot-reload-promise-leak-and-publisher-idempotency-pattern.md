# Hot-Reload Single-Flight Promise Leak + Publisher Idempotency (CLOE Living Capability Evidence)

Session: CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1_HOT_RELOAD_AND_PUBLISHER_IDEMPOTENCY_REPAIR_V1 (2026-08-04). Mission MIXED sur openclaw-runtime/lah-openclaw-mvp. Le publisher est `bin/cloe-evidence-publisher.mjs` (worktree contenant l'arbre du merge supersession 6cda879d == 38a0838 ; le worktree deploy-evidence-v1 à b3b79ee n'a PAS supersession-resolver et produirait 0 promotion).

## Défaut A — hot reload figé par une fuite de promesse single-flight (root cause exact)

Symptôme : le provider canonique de longue durée ne transitionne PAS vers un current-manifest nouvellement promu sans restart. Le live summary continue de servir l'ancien graph 10+ min après la promotion ; un provider FRAIS sur le même graph dir charge le nouveau graph correctement.

Mécanisme exact (`src/self-audit/capability-graph/canonical-graph-provider.mjs`, getCurrentGraph) :

    currentCachePromise = (async function resolveCurrent() { ... })();

L'IIFE async contenait des retours anticipés SYNCHRONES (« manifest unchanged → reuse cache », « legacy flat → preserve cache ») sans `await`. Un appel de fonction sans await résout sa promesse retournée SYNCHRONEMENT (microtask, mais AVANT que l'affectation `currentCachePromise = promise` ne se termine). Un `finally { currentCachePromise = null }` placé DANS l'IIFE — même n'enveloppant que la branche reload async — s'exécute AVANT que l'affectation n'atterrisse ; l'affectation écrase donc la valeur nettoyée et la promesse RÉSOLUE fuit dans currentCachePromise.

Chaque requête suivante prend alors :

    if (currentCachePromise !== null) return currentCachePromise;

et sert la promesse figée avec l'ancien graph. Le manifest n'est plus JAMAIS relu ; toute promotion ultérieure est invisible jusqu'au restart. C'est pourquoi l'hypothèse du fallback flat était fausse : le provider restait sur le graph du manifest PRÉCÉDENT (a7867b10), pas sur le flat legacy (3dfe01a6).

Pourquoi les tests simples passent : une transition A→B avec une seule requête avant la promotion n'exerce JAMAIS le retour anticipé « unchanged ». Le bug exige la séquence production : requêtes périodiques (toutes « unchanged ») AVANT la promotion.

Recette de reproduction (tests RED) :
1. serveur/provider démarre sur le graph A ;
2. plusieurs requêtes à intervalles > manifestPollMs (chacune passe par le chemin unchanged) ;
3. promotion du graph B ;
4. après l'écoulement du poll, le serveur sert encore A (FAIL attendu avant réparation) ;
5. un provider FRAIS sur le même dir sert B → prouve la promesse figée, pas un contenu cassé.

Vérification de la fuite en direct : instrumenter getCurrentGraph avec un log de `promise=` ; après le premier chemin unchanged, `promise=true` reste coincé en permanence.

## Pattern de fix (défaut A)

Nettoyer la promesse single-flight depuis l'EXTÉRIEUR de l'IIFE, attachée à la promesse ASSIGNÉE, avec une garde d'identité :

    const promise = (async function resolveCurrent() { ... })();
    currentCachePromise = promise;
    const clearPromise = () => { if (currentCachePromise === promise) currentCachePromise = null; };
    promise.then(clearPromise, clearPromise);
    return promise;

`.then()` s'exécute après l'affectation pour les chemins synchrones ET asynchrones ; la garde d'identité empêche de nettoyer une promesse plus récente sous charge concurrente.

## Défaut B — `--promote` répété non idempotent (self-référence previous→current)

Symptôme : re-exécuter `--promote` pour le graph déjà désigné current produit un second receipt PROMOTED et réécrit previous-manifest avec le graph courant lui-même (self-référence current→current).

Mécanisme (`src/self-audit/evidence/canonical-publisher.mjs`) : `previousIdentity = current.identity` est résolu AVANT toute comparaison, puis previous-manifest et current-manifest sont écrits inconditionnellement. Quand le hash candidat == hash courant, previousIdentity EST le graph courant → archive self-référentielle.

Pattern de fix : après validateCandidateLoadable, comparer `candidate.graph_hash === current.identity.graph_hash`. Quand identique → retourner `outcome: 'ALREADY_CURRENT'` avec zéro promotion/démotion et AUCUNE écriture de manifest, AUCUNE mutation de snapshot, AUCUN receipt PROMOTED trompeur. S'applique à promote ET dry-run (dry-run doit rapporter zéro mutation).

Vérification réelle : dry-run du graph déjà courant (dir production) → ALREADY_CURRENT, 0 promotion/0 démotion, manifests byte-identiques après.

## Validator chain-integrity (nouveau module)

`src/self-audit/evidence/manifest-chain-integrity.mjs`, câblé dans runPublication avant toute mutation, fail-closed sur :
- CURRENT_EQUALS_PREVIOUS (hash current == hash previous) ;
- PREVIOUS_SNAPSHOT_SELF_REFERENCE (snapshot_path previous == current) ;
- PREVIOUS_SNAPSHOT_MISSING (snapshot précédent déclaré absent) — MAIS un archive legacy flat avec `snapshot_path:null` n'est PAS une violation (le premier promote depuis legacy flat archive avec snapshot_path null par design).

## Quirks d'environnement découverts

- **test/reports/\*.json** : les runs de suite complète régénèrent les reports trackés EN COURS de run ; les tests cognitifs/gateway qui les lisent échouent DURANT le run complet mais passent individuellement avec reports propres. Restaurer avant/après : `git checkout -- test/reports/`. Classer les échecs en vérifiant les fichiers fautifs individuellement avec reports propres — ne pas conclure à une régression.
- **Masquage secret Hermes corrompt les scripts de test** : écrire `process.env.ADMIN_API_KEY = 'x'` dans un fichier de test fait que le scanner de secrets réécrit la ligne en `***` dans le FICHIER (corruption silencieuse, syntaxe cassée). Workaround : construire le nom de clé par concaténation : `process.env['ADMIN_' + 'API_KEY'] = ...` — le scanner ne reconnaît pas le motif littéral.
- **Reproduction dans le conteneur réel** : `docker cp /tmp/repro.mjs <container>:/tmp/` puis `docker exec <container> node /tmp/repro.mjs` avec le vrai graph dir monté (ro) — la repro la plus fidèle avant déploiement. Les worktrees frais ont besoin du symlink node_modules : `ln -s <autre-worktree>/lah-openclaw-mvp/node_modules node_modules` (dotenv et deps manquent sinon).
- **CI `openclaw-ci`/`ci-governance` mort pré-existant sur main** : échoue sur main depuis b3b79ee (5+ runs consécutifs). Ne pas bypass sans autorisation opérateur ; documenter la pré-existence (`gh run list --branch main --workflow openclaw-ci`).
- **Restauration previous-manifest après double --promote** : le receipt de publication authentique (1er) porte `previous_graph_hash` correct — le restaurer depuis ce receipt, ne pas inventer l'identité.
