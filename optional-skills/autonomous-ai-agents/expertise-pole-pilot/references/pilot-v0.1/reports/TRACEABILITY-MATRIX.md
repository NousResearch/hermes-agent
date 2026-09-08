# Matrice de traçabilité

| Exigence | Document | Test/revue | Preuve | Statut |
|---|---|---|---|---|
| Inventaire/périmètre/risques/décisions | PHASE-0-INVENTORY | contrôle adversarial local | observations locales/blocage | couvert, validation Vincent requise |
| Carte capacités/dépendances/risques | CAPABILITIES | audit + CASE | SEO-C01…SEO-C06 | couvert structurellement |
| Taxonomie sans confusion | DOCUMENT-TAXONOMY | audit local | définitions exclusives | expérimental |
| Neuf documents SEO distincts | domains/seo neuf canoniques | `test_audit_pack` | responsabilités exclusives | passé : RUN-001 |
| Six profils | agents/ | `test_audit_pack` | mission/limite/I-O/autonomie | passé structurellement |
| Contrats/handoffs/escalades | CONTRACTS + WORKFLOW | CASE-02/03/04 | états contrôlés | couvert fixture |
| Sources primaires + OSS | SOURCES | transparence | recherche bloquée, aucun faux résultat | bloqué sans autorisation |
| Skills vérifiables | skills/* | `test_audit_pack` | six rubriques obligatoires | passé structurellement |
| Six scénarios/régression | evaluations/ | CASE + RUN-001 | outputs locaux + 6 tests OK | passé fixture |
| Revue contradictoire indépendante | INDEPENDENT-REVIEW | critique distincte | blocage NO-SPEND documenté | **bloqué** |
| Maturation | MATURITY-DECISION | revue finale | critères/limites | expérimental, aucune promotion |
| Pas d'action interdite | EXECUTION-LOG | audit local | déclaration + guard | couvert |
