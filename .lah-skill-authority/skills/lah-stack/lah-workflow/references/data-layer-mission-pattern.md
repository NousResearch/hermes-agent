# Data-layer missions on OpenClaw opportunity-reports (GEO/AI-Companion pipeline)

Classe de missions gated READ_ONLY_AUDIT qui produisent des ARTEFACTS DE DONNÉES (JSON/MD) sous
`/home/deploy/.hermes/openclaw/opportunity-reports/<projet>/<mission>-v1/` — PAS des repos Git.
Exemples réels : pilote GEO (schema v0.1→v0.2), remédiation WB, matrice réglementaire, concurrence commerciale.
Chaque mission ajoute UNE couche au dataset pilote et laisse les couches précédentes intactes.
Workflow réellement suivi : preflight réconciliation → schema validé → collecte bornée → simulation opérateur
→ mise à jour dataset isolée par couche → rapports + SHA256 → verdict de la liste autorisée.

## Gate 0 — routing
- Le routeur renvoie UNRESOLVED (INSUFFICIENT_STRUCTURAL_EVIDENCE) pour les racines de données : c'est NORMAL.
  Appliquer MANUAL_OVERRIDE : workspace = `<projet>/<mission>-v1/`, write_forbidden_roots = lah-stack-repos,
  lah-stack-worktrees, /opt/data (inexistant — résoudre tout chemin /opt/data/openclaw/... vers
  /home/deploy/.hermes/openclaw/...). Écrire le receipt dans /tmp et le propager.

## Preflight réconciliation (obligatoire avant chaque nouvelle mission)
1. Snapshot immuable + SHA256SUMS des artefacts existants AVANT toute modification (chmod a-w le dossier).
2. Autorités par précédence : dataset + cell reports (1) > continuity JSON (2) > rapport final (3) > rapports secondaires (4).
3. Chercher les mentions obsolètes (grep BLOCKED / INJOIGNABLE / vieilles valeurs) : corriger le SCRIPT GÉNÉRATEUR
   (pas seulement le fichier), régénérer, re-vérifier le grep = vide. Sinon la régénération réintroduit le stale.
4. Prouver la couche économique intacte : hashes de la couche archivés dans evidence/ de la mission.
5. Livrer PILOT_ARTIFACT_RECONCILIATION_REPORT.md (contradictions, autorité retenue, STALE_ARTIFACT, IDs orphelins).
6. ID source orphelin (ex. eu_dsa_avmsd absent du registre) : régulariser avec URL officielle vérifiée HTTP 200
   (EUR-Lex : https://eur-lex.europa.eu/eli/reg/2022/2065/oj, /eli/dir/2018/1808/oj) — jamais laisser d'orphelin.

## Schema-first
- Le spec mission donne le schéma cible : le VALIDER programmatiquement avant collecte (JSON valide + enums +
  alignement cross-fichiers des dimensions — ex. 10 dims identiques schema/mapping/contract).

## Collecte bornée (sources gratuites, read-only)
- PROBE court par source (curl, timeout 6s, 1 retry) AVANT la collecte : si injoignable → evidence BLOCKED, on continue.
- Budget temporel par source + wrapper `timeout N` ; ne jamais lancer 100+ requêtes sans probe.
- Piège urllib : le timeout socket ne couvre PAS le hang DNS getaddrinfo (process bloqué en do_poll).
  Si un hôte pend, tester avec curl d'abord, ou court-circuiter la source et utiliser le fallback.
- Fallback : réutiliser les valeurs vérifiées antérieurement (même source, retrieved_at antérieur) — method=evidence_reuse,
  documenté, jamais présenté comme nouveau fetch.
- WB API : code pays UK→GB (ISO-2 "UK" invalide). Vérifier le comptage par géo après collecte (14/15 = geo manquante).
- Eurostat JSON-stat (API dissemination) :
  - Les clés value[] sont des INDICES PLATS, pas "T2024" — décoder par strides (tailles des dimensions dans l'ordre de `id`)
    + dimension.time.category.index ; le code time EST l'année (pas la position).
  - isoc_ci_in_en2 = ENTREPRISES (faux pour internet_penetration) → utiliser isoc_ci_ifp_iu (I_IU3, IND_TOTAL, PC_IND).
  - ilc_di01 : filtrer statinfo=TC (SHARE = pourcentages absurdes) ; quant_inc D5 = proxy documenté du revenu médian
    (pas de P50 dans la série) ; unité PPS.
  - UK : income absent post-Brexit → la dernière valeur non nulle (2020) est le comportement correct.

## Recherche par subagents (Gate 2)
- Lane trop large = timeout : 3 lanes × 7 pays × 7 dimensions ont timeout à 600s (0 JSON écrit).
  Découper : 1-2 pays ou UNE plateforme par lane ; demander d'écrire les JSON PARTIELS tôt (pas seulement à la fin) ;
  exiger des URLs vérifiées par curl (HTTP 200 ou 403/000 documenté) et relire le contenu avant de citer un constat.
- Les pages sauvegardées par un subagent (evidence/research/*/_pages/) restent des preuves réutilisables.

## Mise à jour du dataset isolée par couche
- Chaque mission ajoute SA couche ; ASSERTION d'intégrité : comparer dict par dict les blocs des couches précédentes
  (économique : coverage, pilot_dimensions, offer/acquisition/consumer_economics, payment, market_signals,
  data_readiness_status, critical_missing, contradictions ; réglementaire : regulatory_* ) avant/après par cellule.
- Snapshot dataset avant (evidence/dataset_before_<mission>.json) + diff md.
- Recalculer UNIQUEMENT les champs de la mission (réglementaire : market_hypothesis + simulation_readiness ;
  concurrence : 5 champs sur les géos cibles seulement). Une dimension pilote (pilot_dimensions.x) peut rester MISSING
  même quand le champ cellule est rempli — la couche de données est figée.
- Régénérer cell reports + SHA256SUMS. Piège : patcher un fichier APRÈS l'avoir hashé crée une entrée dupliquée
  (ancien + nouveau hash) → mismatch à la validation : régénérer tout le fichier SHA256SUMS proprement.

## Simulation opérateur (behavioral)
- Réutiliser le MODULE RÉEL (CellBuilder / variant_verdict / registre) avec entrées mockées ; 10 scénarios ;
  une ligne par scénario ; evidence JSON + rapport.
- Assertions naïves par sous-chaîne = FAUX NÉGATIFS (piège connu) : tester structurellement (pas de "$", pas de
  champ budget) ou avec marqueur de contexte négatif ("non inférable", "aucun", "ne déduis") à ~40 chars, pas la présence nue.

## Verdict ladder
- La mission définit les verdicts autorisés et les critères d'arrêt : vérifier NUMÉRIQUEMENT les critères d'arrêt
  avant de choisir (ex. "<50% des dimensions couvertes sur la majorité des géos" : 7/15 <50% ≠ majorité → non déclenché ;
  1 source bloquée < 3 → non déclenché) et documenter pourquoi.
- READY_WITH_LIMITATIONS quand : sources à accès restreint (403/000 documentés), confiance < 0.6, variantes à risque,
  données de transaction absentes (paid_demand_evidence = PROXY_ONLY).

## Particularités par mission (déjà rencontrées)
- GEO pilote : règle data_readiness = covered ≥ 8 ET ZÉRO contradiction → les cellules 9/10 avec contradiction
  documentée internet (Eurostat vs WB) sont DATA_PARTIAL alors que des 8/10 sans contradiction sont DATA_READY.
- Réglementaire : AE HYPOTHESIS_BLOCKED était une hypothèse non sourcée → réévaluer depuis les sources officielles :
  explicite BLOCKED prouvé, généraliste opérable → consolidé HIGH_RISK, hypothèse WEAK. Ne garder BLOCKED que si prouvé.
- Concurrence : iTunes Search API = preuve réelle (notes + nb d'avis par store, devise locale) ; prix web souvent JS
  non extraits (limite documentée) ; Play page 200 mais titres JS ; Meta Ad Library NOT_EXECUTED sans token — jamais un blocage.
