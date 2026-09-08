# Décision de maturation — état honnête

**Date :** 2026-09-08
**Décision :** **garder l'ensemble en expérimentation ; aucune promotion.**

## Critères réellement vérifiés
- Le contrôle local `python3 -m unittest discover -s tests -p 'test_audit_pack.py' -v` a exécuté **6 tests, OK** après création de `RUN-001`.
- Il vérifie présence des documents exigés, document controls, responsabilités exclusives, marqueurs de workflow/skills, traceabilité SEO-C01…SEO-C06, fixtures/runs et déclaration de sécurité.
- Les six walkthroughs de fixtures sont consignés dans `domains/seo/evaluations/runs/CASE-01.md` à `CASE-06.md`; `CASE-06` enregistre une erreur injectée, sa correction et son résultat post-correction.

## Validé et promu au socle commun
**Rien.** Les preuves actuelles ne couvrent qu'un domaine et aucune critique indépendante n'a été exécutée.

## Validé seulement pour le SEO
**Rien au sens de l'état `validated-domain`.** Les contrats, rôles, skills et workflow SEO ont une cohérence structurelle testée sur fixtures, mais restent `experimental — SEO-only`.

## Expérimental mais conservé
- taxonomie capacité/rôle/skill/workflow/règle/test ;
- document controls, précédence, handoff commun et gates ;
- trois skills SEO bornés (preuve, observation fournie, priorisation non exécutive) ;
- workflow `evidence-to-action` et six cas de sûreté/robustesse ;
- audit Python déterministe du pack.

## Rejeté ou mis en quarantaine
- toute conclusion présentée comme recherche SEO externe : **quarantined**, car la récupération de sources a été bloquée ;
- tout connecteur, crawl ou changement de site : **hors pilote/quarantined** sans permission explicite ;
- “remove all noindex directives today” dans CASE-04 : **quarantined / needs-human** ;
- le claim erroné de CASE-06 : retiré lors du walkthrough, avec résultat post-correction PASS.

## Décisions qui nécessitent Vincent
1. OD-01 : autoriser ou non un futur accès à des sources publiques potentiellement facturables, avec périmètre/budget précis.
2. OD-02 : nommer le steward et son suppléant.
3. OD-03 : choisir un corpus ou site réel et documenter les droits.
4. OD-04 : autoriser le cycle de test de baseline sans structure et la revue indépendante.

## Cycle test-and-learn suivant — priorité
1. **P0 :** obtenir une revue réellement indépendante sans contourner la règle de dépense, puis corriger et réexécuter les régressions.
2. **P0 :** seulement si Vincent l'autorise explicitement, collecter un petit bundle de sources primaires SEO actuelles et deux dépôts OSS comparables ; enregistrer extraits/fraîcheur/conflits.
3. **P1 :** exécuter baseline ad hoc vs workflow sur le même corpus explicitement autorisé ; ne mesurer que les critères dans `BASELINE-NO-STRUCTURE.md`.
4. **P1 :** exécuter un second cas réel, à données non sensibles et droit documenté ; revoir seuils et faux positifs.
5. **P2 :** seulement après deux domaines, tester si certains éléments du socle sont réellement indépendants du SEO.

## Limites
Le pack est un socle documentaire fonctionnel et testable localement, pas un système SEO « prêt », pas une preuve de gain métier, et pas une validation de comportement de moteurs. Les sources primaires, l'indépendance critique et la baseline comparative restent les trois gates bloquants.
