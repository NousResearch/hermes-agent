# Hermes Update Plan — mise à jour sans risque

**Goal:** Mettre à jour Hermes sans casser l’installation actuelle, en préservant les patchs locaux critiques pour Telegram/WhatsApp et la surveillance scolaire de Robin.

**Contexte actuel vérifié:**
- Version installée : Hermes Agent v0.8.0 (2026.4.8)
- Dépôt local : `/home/manu/.hermes/hermes-agent`
- Branche : `main`
- Retard sur l’amont : 59 commits
- Modifications locales détectées :
  - `gateway/run.py`
  - `scripts/whatsapp-bridge/bridge.js`
  - `tests/gateway/test_background_process_notifications.py`
  - `package-lock.json`

**Pourquoi il faut être prudent :**
Les correctifs locaux touchent le cœur du gateway et du bridge WhatsApp. Une mise à jour directe peut écraser :
- le silence sur les DM WhatsApp non autorisés
- le correctif des événements synthétiques Telegram/DM
- le support `WHATSAPP_MONITOR_GROUPS`
- l’écriture passive dans `group_events.jsonl`
- la logique de surveillance du groupe de Robin

---

## Résultat visé

À la fin, on doit avoir :
- Hermes mis à jour ou au moins évalué proprement
- tous les patchs locaux préservés
- le gateway toujours fonctionnel
- le bridge WhatsApp toujours connecté
- le moniteur Robin toujours opérationnel
- zéro régression visible côté Telegram/WhatsApp

---

## Stratégie recommandée

Ne pas faire `hermes update` directement.

Procéder en 3 phases :
1. sauvegarde complète de l’état actuel
2. test de mise à jour dans un espace contrôlé
3. validation fonctionnelle avant bascule réelle

---

## Phase 1 — Sauvegarde complète avant toute action

### Étape 1 — Capturer l’état git exact
Objectif : pouvoir revenir précisément à l’état actuel.

Commandes :
```bash
cd /home/manu/.hermes/hermes-agent
git status --short
git rev-parse HEAD
git branch --show-current
```

À conserver dans une note :
- commit actuel
- branche actuelle
- liste des fichiers modifiés

### Étape 2 — Sauvegarder les diffs locaux critiques
Objectif : conserver les patchs même si une update écrase les fichiers.

Commandes :
```bash
cd /home/manu/.hermes/hermes-agent
mkdir -p /home/manu/.hermes/backups/hermes-update-2026-04-09
git diff -- gateway/run.py > /home/manu/.hermes/backups/hermes-update-2026-04-09/gateway-run.patch
git diff -- scripts/whatsapp-bridge/bridge.js > /home/manu/.hermes/backups/hermes-update-2026-04-09/whatsapp-bridge.patch
git diff -- tests/gateway/test_background_process_notifications.py > /home/manu/.hermes/backups/hermes-update-2026-04-09/test-background.patch
git diff -- package-lock.json > /home/manu/.hermes/backups/hermes-update-2026-04-09/package-lock.patch
```

### Étape 3 — Sauvegarder les fichiers complets modifiés
Objectif : avoir aussi une copie brute des fichiers, pas seulement les patches.

Commandes :
```bash
cp /home/manu/.hermes/hermes-agent/gateway/run.py /home/manu/.hermes/backups/hermes-update-2026-04-09/run.py
cp /home/manu/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js /home/manu/.hermes/backups/hermes-update-2026-04-09/bridge.js
cp /home/manu/.hermes/hermes-agent/tests/gateway/test_background_process_notifications.py /home/manu/.hermes/backups/hermes-update-2026-04-09/test_background_process_notifications.py
cp /home/manu/.hermes/hermes-agent/package-lock.json /home/manu/.hermes/backups/hermes-update-2026-04-09/package-lock.json
```

### Étape 4 — Sauvegarder la config Hermes utilisée en production
Objectif : pouvoir restaurer le comportement exact.

Commandes :
```bash
cp /home/manu/.hermes/config.yaml /home/manu/.hermes/backups/hermes-update-2026-04-09/config.yaml
cp /home/manu/.hermes/.env /home/manu/.hermes/backups/hermes-update-2026-04-09/.env
```

### Étape 5 — Sauvegarder les scripts et états liés à Robin
Objectif : ne rien perdre sur la surveillance scolaire.

Commandes :
```bash
cp /home/manu/WhatsApp/whatsapp_ecole_robin_monitor.mjs /home/manu/.hermes/backups/hermes-update-2026-04-09/whatsapp_ecole_robin_monitor.mjs
cp /home/manu/WhatsApp/EcoleRobin_state.json /home/manu/.hermes/backups/hermes-update-2026-04-09/EcoleRobin_state.json
cp /home/manu/.config/systemd/user/whatsapp-ecole-robin-monitor.service /home/manu/.hermes/backups/hermes-update-2026-04-09/whatsapp-ecole-robin-monitor.service
```

---

## Phase 2 — Préparer une zone de test sans toucher à la prod

### Étape 6 — Créer une branche locale de sécurité
Objectif : figer l’état actuel avant expérimentation.

Commande :
```bash
cd /home/manu/.hermes/hermes-agent
git switch -c local/hermes-stable-before-update-2026-04-09
```

### Étape 7 — Créer une seconde branche de test pour l’update
Objectif : tester la mise à jour sans perdre la branche de sécurité.

Commande :
```bash
cd /home/manu/.hermes/hermes-agent
git switch -c test/hermes-update-eval-2026-04-09
```

### Étape 8 — Vérifier si les patchs locaux peuvent être commités proprement
Objectif : transformer les changements actuels en historique local lisible.

Commande :
```bash
cd /home/manu/.hermes/hermes-agent
git diff --stat
```

Décision :
- si les changements sont propres et voulus, faire un commit local de sauvegarde
- sinon, garder les patches + copies de fichiers et ne pas commit

### Étape 9 — Option recommandée : commit local de sécurité
Objectif : rendre la restauration ultra simple.

Commande :
```bash
cd /home/manu/.hermes/hermes-agent
git add gateway/run.py scripts/whatsapp-bridge/bridge.js tests/gateway/test_background_process_notifications.py package-lock.json
git commit -m "local: preserve telegram/whatsapp monitoring patches before update"
```

Si le commit ne passe pas ou si tu préfères éviter, sauter cette étape.

---

## Phase 3 — Évaluer l’update sans bascule immédiate

### Étape 10 — Récupérer l’amont sans fusionner à l’aveugle
Objectif : voir exactement ce qui change.

Commandes :
```bash
cd /home/manu/.hermes/hermes-agent
git fetch origin
git log --oneline HEAD..origin/main | head -n 50
```

### Étape 11 — Comparer les zones sensibles avant update
Objectif : savoir si l’amont a déjà intégré certains correctifs.

Fichiers à comparer en priorité :
- `gateway/run.py`
- `scripts/whatsapp-bridge/bridge.js`
- `tests/gateway/test_background_process_notifications.py`

Commandes utiles :
```bash
cd /home/manu/.hermes/hermes-agent
git diff HEAD..origin/main -- gateway/run.py scripts/whatsapp-bridge/bridge.js tests/gateway/test_background_process_notifications.py
```

### Étape 12 — Mettre à jour seulement dans la branche de test
Objectif : tester l’intégration sans toucher au comportement final en prod.

Commande :
```bash
cd /home/manu/.hermes/hermes-agent
git merge origin/main
```

Si conflit :
- ne pas improviser
- résoudre uniquement après comparaison avec les patchs sauvegardés

---

## Phase 4 — Réappliquer consciemment les patchs locaux

### Étape 13 — Réappliquer uniquement ce qui reste nécessaire
Objectif : éviter de reposer des patchs déjà intégrés en amont.

Méthode :
- comparer le fichier mis à jour avec la sauvegarde locale
- réintroduire uniquement les comportements encore absents

Priorité fonctionnelle :
1. silence sur les DM WhatsApp non autorisés
2. correction des événements synthétiques Telegram/DM
3. `WHATSAPP_MONITOR_GROUPS`
4. `group_events.jsonl`
5. logique de groupe monitoré dans `bridge.js`

### Étape 14 — Vérifier la validité syntaxique locale
Objectif : éviter les erreurs évidentes avant redémarrage.

Commandes :
```bash
cd /home/manu/.hermes/hermes-agent
python3 -m py_compile gateway/run.py
node --check scripts/whatsapp-bridge/bridge.js
node --check /home/manu/WhatsApp/whatsapp_ecole_robin_monitor.mjs
```

---

## Phase 5 — Validation fonctionnelle avant production

### Étape 15 — Vérifier les services systemd liés
Objectif : confirmer ce qui tourne avant et après test.

Commandes :
```bash
systemctl --user status hermes-gateway.service --no-pager
systemctl --user status whatsapp-ecole-robin-monitor.service --no-pager
```

### Étape 16 — Vérifier la santé du bridge
Objectif : s’assurer que WhatsApp reste connecté.

Commande :
```bash
curl -s http://127.0.0.1:3000/health
```

Résultat attendu : état connecté, sans erreur manifeste.

### Étape 17 — Vérifier que le tap file continue de grandir
Objectif : confirmer que la surveillance passive fonctionne toujours.

Commandes :
```bash
test -f /home/manu/.hermes/whatsapp/group_events.jsonl && tail -n 5 /home/manu/.hermes/whatsapp/group_events.jsonl
```

### Étape 18 — Vérifier le moniteur Robin
Objectif : confirmer que les logs et alertes restent possibles.

Commandes :
```bash
systemctl --user is-active whatsapp-ecole-robin-monitor.service
journalctl --user -u whatsapp-ecole-robin-monitor.service -n 30 --no-pager
```

### Étape 19 — Vérifier les régressions critiques
Checklist manuelle :
- Telegram répond normalement
- les traces techniques ne réapparaissent pas si `tool_progress: off`
- les messages du groupe WhatsApp sont encore journalisés
- les alertes Robin continuent d’arriver
- le nom de l’institutrice apparaît toujours comme « Mme Marie-Paule »
- aucun pairing parasite n’est envoyé à un inconnu en DM WhatsApp

---

## Phase 6 — Bascule ou retour arrière

### Étape 20 — Si tout est bon, seulement alors basculer en prod
Options possibles :
- garder la branche mise à jour comme nouvelle base
- ou reproduire les mêmes changements sur `main`

### Étape 21 — Si quelque chose casse, rollback immédiat
Objectif : revenir à l’état stable sans réfléchir.

Option A — retour git :
```bash
cd /home/manu/.hermes/hermes-agent
git switch local/hermes-stable-before-update-2026-04-09
```

Option B — restauration fichiers :
```bash
cp /home/manu/.hermes/backups/hermes-update-2026-04-09/run.py /home/manu/.hermes/hermes-agent/gateway/run.py
cp /home/manu/.hermes/backups/hermes-update-2026-04-09/bridge.js /home/manu/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js
cp /home/manu/.hermes/backups/hermes-update-2026-04-09/config.yaml /home/manu/.hermes/config.yaml
```

Puis redémarrage :
```bash
systemctl --user restart hermes-gateway.service
systemctl --user restart whatsapp-ecole-robin-monitor.service
```

---

## Décision recommandée

### Si ton objectif principal est la stabilité
Ne fais pas la mise à jour tout de suite. Commence uniquement par la Phase 1 et la Phase 2.

### Si tu veux avancer sans risque inutile
Le bon compromis est :
- sauvegarder maintenant
- créer les branches maintenant
- analyser la fusion ensuite
- ne basculer que si tous les tests de Phase 5 passent

---

## Résumé très court

Ne fais pas `hermes update` en direct.

Fais plutôt :
1. sauvegarde complète
2. branche de sécurité
3. branche de test
4. fetch + diff + merge en test
5. réapplication sélective des patchs
6. validation Telegram/WhatsApp/Robin
7. bascule seulement si tout est bon

---

## Commande à ne pas lancer tout de suite
```bash
hermes update
```

## Première commande sûre à lancer
```bash
mkdir -p /home/manu/.hermes/backups/hermes-update-2026-04-09
```
