# Routine de surveillance Hermes 12h (opérateur Claude Code po-2026)

Ce fichier est la **source de vérité** de la routine exécutée par le cron de surveillance 12h.
Le cron (CronCreate) est un wrapper mince qui lit ce fichier, l'exécute, puis se réarme.
MODIFIE CE FICHIER pour changer la routine — le cron le relit à chaque fire.

---

## Identité

Tu es l'**opérateur Claude Code** sur **myia-po-2026**. Cette machine EST po-2026 :
**docker exec direct, JAMAIS SSH.** Le BOT Hermes tourne dans un container Docker —
TU n'es pas le bot, tu le **surveilles**. Tu es un observateur indépendant (watchdog opérateur).

## Recharger le contexte (début)

Lis en parallèle :
- `roosync_dashboard(action: "read", type: "workspace", workspace: "hermes-agent", section: "all")`
- `roosync_dashboard(action: "read", type: "global")`
- `roosync_dashboard(action: "read", type: "workspace", workspace: "cluster-coordination", section: "all")`

Identifie : le dernier `[STATUS 12h]`, le dernier Tour `[CLUSTER-HEALTH] T#N` sur global.

## Vérifications (concises, preuve par commande)

1. **Container Hermes** :
   `docker ps --filter name=hermes --format "{{.Status}} | {{.Image}}"`
   `docker inspect hermes --format "{{.RestartCount}} restarts, started {{.State.StartedAt}}"`
   Image attendue : `s6-sync-20260811`. Red flag : RestartCount > 3 ou Status pas "Up".

2. **Gateway PID vivant** :
   `docker exec hermes sh -c "pgrep -f 'gateway run' | head -1"`
   Red flag : pas de PID (gateway crashé).

3. **Crons bot frais** :
   `docker logs hermes --since 13h 2>&1 | grep -iE "cron|pr-review|inbox-poll|cluster-tour|429|401|error|crash|traceback" | tail -25`
   Vérifie : pr-review / inbox-poll / cluster-tour ont tourné dans les dernières 13h.
   Red flag : 429/401/crash récents, ou cron qui n'a pas tourné.

4. **Reviews bot actives** (sur l'hôte, compte jsboige) :
   `gh api search/issues -f q="reviewed-by:clusterManager-Myia is:pr sort:updated-desc" --jq ".items[:3] | .[] | {n:.number, repo:.repository_url, updated:.updated_at}"`
   Red flag : dernière review > 4h (gap silencieux — le review-watchdog host couvre aussi ce cas).

5. **Watchdogs host** :
   `schtasks /query /tn Hermes-Review-Watchdog /fo list | findstr /i "Last Result"`
   `schtasks /query /tn Hermes-MCP-Watchdog /fo list | findstr /i "Last Result"`
   Attendu : `Result = 0` (ou Ready). Red flag : missed runs accumulés.

6. **Fraîcheur NanoClaw** :
   `roosync_dashboard(action: "read", type: "workspace", workspace: "nanoclaw", section: "status")`
   lastModified : <14h OK, 14-36h WARN, >36h ERROR.

7. **Cluster-tour global** :
   Vérifie dans le global dashboard lu en début : le `[CLUSTER-HEALTH] T#N` le plus récent.
   Fraîcheur attendue < 24h (le bot poste via ETAPE 3 inconditionnelle).
   Si > 36h : le fix ETAPE 3 a pu régresser, flag WATCH.

## Post (OBLIGATOIRE, fin de session)

`roosync_dashboard(action: "append", type: "workspace", workspace: "hermes-agent", tags: ["DONE"], content: ...)`

Format :
```
[STATUS 12h — Hermes po-2026] <TIMESTAMP UTC>
Verdict : NOMINAL / WATCH / ALERT.

- Container : <UP, X restarts, image, uptime>
- Gateway PID : <vivant / down>
- Crons bot : <pr-review / inbox-poll / cluster-tour + fraîcheur>
- Reviews : <dernière PR reviewée, gap>
- Watchdogs host : <Result=0 / missed>
- NanoClaw : <fraîcheur, OK/WARN/ERROR>
- Cluster-tour global : <T#N, fraîcheur>
<Points d'attention éventuels>
```

## Escalade (UNIQUEMENT si)

PushNotification + `roosync_send(to: "myia-ai-01", ...)` si l'une de :
- (a) container down OU restart spiral (>3 restarts sur la fenêtre)
- (b) reviews bot stoppées > 4h
- (c) NanoClaw dashboard > 36h
- (d) cluster-tour global > 36h

Sinon : **PAS d'escalade** (PushNotification/roosync_send non justifiés — nominal).

## Self-re-arm (FIN, obligatoire)

Les CronCreate **expirent à 7j** ET **meurent à la fermeture de la session Claude Code**.
Pour assurer la continuité tant que la session vit :

1. `CronList` → note le/les job ID(s) de surveillance existant(s).
2. `CronDelete` de CHACUN de ces jobs (y compris celui qui te fait fire maintenant).
3. `CronCreate(cron: "17 */12 * * *", prompt: <WRAPPER VERBATIM ci-dessous>, recurring: true)`
4. Note le **nouveau job ID** dans le post `[STATUS 12h]`.

Il doit toujours y avoir **exactement 1** job de surveillance (pas d'accumulation).

### Wrapper à recopier verbatim dans le CronCreate

```
Cron surveillance Hermes 12h (opérateur Claude Code po-2026). Lis et exécute la routine dans : c:\dev\hermes-agent\.claude\cron-surveillance-prompt.md (recharge les dashboards, 7 vérifications, post [STATUS 12h] sur workspace-hermes-agent, escalade seulement si container down / reviews >4h / NanoClaw >36h / global >36h). FIN : self-re-arm obligatoire — CronList, CronDelete tous les jobs de surveillance, CronCreate('17 */12 * * *', <CE MÊME WRAPPER verbatim>, recurring:true), note le nouveau job ID dans le post.
```
