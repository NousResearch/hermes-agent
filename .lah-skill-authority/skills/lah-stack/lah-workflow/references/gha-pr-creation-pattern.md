# GHA-Style PR Creation (lah-stack-tools)

## Contexte

Créer une PR à partir d'une branche locale qui a des commits d'historique, en ne sélectionnant que le périmètre voulu (excluant les commits AUTO/BIZ).

## Problème

La branche `orchestration-4-publication-governance` contient un mélange de :
- Governance core (BW29-BW32, OC30) — à inclure dans PR1
- Autonomy pipeline (AUTO1-3, BW33-35, OC32-33) — PR2
- Business assets (BIZ1-6, BIZX) — PR3

On ne peut pas cherry-picker tous les commits car chaque commit peut contenir des fichiers de scope différent.

## Solution : branche propre depuis master + cherry-pick sélectif

```bash
# 1. Stash le working tree sale (artefacts BIZX générés)
git stash push -m "dirty generated metadata"

# 2. Partir de master
git checkout master
git pull --ff-only

# 3. Créer une feature branch propre
git checkout -b gha3-governance-core-regularization

# 4. Cherry-pick les commits approuvés (un par un)
git cherry-pick 0f75f8b   # ORCH4 governance core

# 5. Pour les commits qui mélangent fichiers in/out, n'appliquer que le fichier souhaité
git show 0cc178e -- .gitignore | git apply
git add .gitignore
git commit -m "GHA2 tools cleanup: add DO_NOT_PUSH patterns"

# 6. Vérifier qu'aucun contenu non désiré n'a fuité
git diff --name-status master...HEAD
grep -E 'biz|adult-cam|ai-companion|hentai|dating' <(git diff --name-only master...HEAD) \
  && echo "ALERTE: contenu BIZ détecté!" || echo "Clean ✓"

# 7. Vérifier les tests et le diff
node --test tools/control-plane/tests/bw29*.test.mjs tools/control-plane/tests/bw30*.test.mjs [...]
git diff --check

# 8. Push et créer PR
git push -u origin gha3-governance-core-regularization
gh pr create --title "Titre" --body-file /tmp/pr-body.md
```

## Vérifications post-création

```bash
gh pr view <NUMERO> --json number,state,mergeable,title,url
git status --short --branch              # doit être clean (sauf untracked préexistants)
git log --oneline origin/<branch>..HEAD  # doit être vide (ahead=0)
```

## Vérification de scope PR — nuance sur le mot "auto"

Quand tu vérifies qu'aucun contenu AUTO/BIZ n'a fuité dans la PR, **ne pas**
utiliser un simple `grep -i "auto"` — le mot "auto" apparaît dans des noms de
fichier légitimes comme `bw32-policy-based-auto-publish-preflight.mjs` et
`autonomy-governance-audit.mjs`. Utiliser une boucle avec des patterns précis :

```bash
for pat in biz adult-cam ai-companion hentai dating business-asset; do
  matches=$(git diff --name-only master...HEAD | grep -i "$pat" || true)
  if [ -n "$matches" ]; then echo "ALERTE: '$pat' trouvé dans: $matches"; fi
done
```

## Lancement des tests avec --test-concurrency=1

Les tests du control-plane partagent le répertoire `tools/control-plane/data/`.
Quand plusieurs fichiers de test s'exécutent en parallèle, un test peut nettoyer
le kill switch qu'un autre test est en train de lire, causant des échecs
intermittents.

**Toujours utiliser** `--test-concurrency=1` pour les suites multi-fichiers :

```bash
node --test --test-concurrency=1
```

## Modèle de corps de PR

```markdown
## Scope
Description concise.

### Included Missions
- **[MISSION]** — description courte

### Included Files (N)
N fichiers.

### Excluded Work
- **[CONTENU]** — raison

### Safety
Boundaries FastSafe.

### Test Results
- Targeted (N): N/N pass
- Full suite: N/N pass
- Known failure(s)
```

## Stash pop

Après avoir fini le travail de PR, revenir sur la branche originale et restaurer le stash :

```bash
git checkout orchestration-4-publication-governance
git stash pop
```

## Piège : stash pop sur la mauvaise branche → fichiers "DU"

Si tu fais `git stash pop` sur une branche qui ne contient **pas** les fichiers
du stash, ceux-ci apparaissent en statut `DU` (Deleted/Unmerged) :

```
DU docs/business-assets/.../fichier.meta.json
```

**Solution** :
```bash
git rm --cached <fichiers-en-DU>
```

**Ne pas** faire `git checkout -- .` — cela échoue sur les fichiers unmerged.

## Piège : `gh pr merge` échoue avec index sale

Si un stash est présent ou a laissé des fichiers "DU" :
```
failed to run git: error: you need to resolve your current index first
```

**Solution** :
```bash
git stash drop
git rm --cached <fichiers-du>
gh pr merge --squash --delete-branch
```

## DO_NOT_PUSH (.gitignore)

Ces patterns DOIT être dans `.gitignore` pour éviter de pusher des artefacts locaux :

```
tools/runs/
tools/control-plane/continuity-*.json
.codegraph-repair/
```
