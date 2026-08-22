# ExoClick Zone Prefilter Mission Pattern

Établi lors de CLOE_EXOCLICK_OURDREAM_ZONE_INTELLIGENCE_PREFILTER_V1 (2026-08-01), mission
READ_ONLY_DISCOVERY_ANALYZE_AND_PREPARE. Réutiliser pour toute mission de préfilter
sites/zones d'un réseau de trafic payant (ExoClick) avant tout spend, et pour toute
sélection de sources pré-live d'une offre (ex. OurDream 10138).

## Quand ce pattern s'applique
- Construire des listes ALLOW/BLOCK/CAPPED/REVIEW pour un réseau de trafic (ExoClick)
- Sélection de sources pre-live pour une offre (compatibilité d'intent + historique)
- Récupérer un "prefilter précédent" avant lancement (ou déclarer son absence)

## Autorités de données sur ce VPS (vérifiées 2026-08-01)
| Source | Chemin | Contenu |
|---|---|---|
| Campaign ledger | `openclaw-runtime/lah-openclaw-mvp/data/campaign-ledger.json` | campagnes créées + payloads (zones/sites arrays) |
| Snapshots | `openclaw-runtime/lah-openclaw-mvp/snapshots/` | JSON snapshot campagnes (ex. 8293490 : `zones: []`, `sites.targeted/blocked: []`) |
| Decision log | `openclaw-runtime/lah-openclaw-mvp/data/decision.log` | entrées spend/clicks/conversions (majorité = fixtures de test) |
| Execution receipts | `.../data/execution-receipts.json` | souvent vide — ne pas supposer de contenu |
| ExoClick refs | `openclaw-runtime/exoclick_refs/` | categories.json, countries_clean.json, advertiser-ad-types.json, swagger.json |
| Runtime refs | `lah-openclaw-mvp/data/refs/` | category_name_to_id.json, country_iso2_to_id.json, device_name_to_id.json |
| Gate env | `lah-openclaw-mvp/.env` | `EXOCLICK_LIVE_ENABLED=false` → **aucun appel API**, utiliser les refs locaux |
| Missions antérieures | `~/.hermes/openclaw/opportunity-reports/geo-opportunity-matrix/<mission>/` | rapports microtest, device-subid, crakrevenue audit |

## Faits ExoClick clés (refs locales)
- Catégorie 2 = Adult (parent, `selectable=0` — **non ciblable directement**)
- **Catégorie 526 = Adult - AI (selectable) — cible primaire pour offres AI companion**
- Adjacentes : 492 Dating, 506 Erotic/Sexy, 110 Cartoons/Hentai, 524 Webcams, 515 VR, 517 Manga
- Catégorie 97 = Amateur (défaut historique des anciennes campagnes — faible affinité AI)
- US country id = 840 (countries_clean.json + country_iso2_to_id.json)
- Popunder = advertiser_ad_type 7 ; publisher_ad_types : 12 Mobile Popunder, 3 Popunder, 22 RTB Popunder Supply
- Endpoints sites/zones/stats documentés dans swagger.json mais verrouillés par LIVE=false

## Récupération du prefilter précédent — déclaration d'absence
Si aucun prefilter n'existe, DOCUMENTER l'absence explicitement (PREVIOUS_ZONE_PREFILTER_RECOVERY_REPORT.md) :
- `grep -ri "prefilter"` sur repos + evidence dir ; `grep -ri "whitelist|blacklist|allowlist"`
- `session_search "prefilter zone site whitelist blocklist"` (0 résultat attendu si absent)
- Vérifier les snapshots de campagne : `zones: []` + `sites: {targeted:[], blocked:[]}` = ciblage réseau entier (pas de prefilter appliqué)
Une déclaration d'absence écrite empêche les futurs agents de chercher un prefilter qui n'a jamais existé.

## Crawl borné titres-only pour classification de domaines
Classifier les domaines adultes SANS télécharger de médias :
```bash
curl -sL --max-time 8 --max-filesize 1500000 \
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36" \
  "https://$d" 2>/dev/null | grep -o -i "<title[^>]*>[^<]*</title>" | head -1 | sed 's/<[^>]*>//g'
```
- HTTP 403/406/520 (Cloudflare/challenge) → classer depuis la connaissance de la marque, confiance réduite (0.75-0.85)
- Jamais de média, jamais de screenshot, jamais d'authentification
- Conforme aux contraintes "titles, metadata, publicly observable structure"

## Classes d'intent (taxonomie mission)
VIDEO_TUBE_LINEAR, VIDEO_AGGREGATOR, GALLERY_IMAGE, LIVE_CAM, DATING, CHAT_COMMUNITY,
AI_COMPANION, INTERACTIVE_FANTASY, HENTAI_GAME, STORY_ROLEPLAY, DOWNLOAD_OR_LINK_AGGREGATOR,
GENERAL_ADULT, UNKNOWN_OR_MASKED

## Formule de scoring (5 dimensions, explicable)
```
SCORE = 100 × (wH·H + wI·I + wQ·Q + wC·C + wE·E)   # poids 0.25/0.30/0.15/0.15/0.15
```
- H performance historique, I compatibilité d'intent, Q qualité du trafic, C contexte, E fiabilité des preuves
- Défauts de sécurité : H=0 sans historique (jamais fabriquer) ; I<0.35 pour classes passives ; E<0.5 si inférence seule ; Q<0.4 si risque clic accidentel
- Seuils : >=55 ET (H>0 OU I>=0.85) → ALLOW_INITIAL ; 40-54 → ALLOW_CAPPED_EXPLORATION ; <40 → BLOCK_INITIAL ; E<0.4 → REVIEW_MANUALLY

## Classes de décision
ALLOW_INITIAL, ALLOW_CAPPED_EXPLORATION, BLOCK_INITIAL, REVIEW_MANUALLY, INSUFFICIENT_DATA.
Chaque entrée : site_id, zone_id, domain, decision, decision_scope (SITE/ZONE/CLASS), reason_codes,
intent_class, historical_* (**null quand absent — ne jamais fabriquer**), confidence, evidence_refs,
operator_review_required, proposed_zone_loss_cap, status.

## Structure de sortie (16 artefacts)
1. CLOE_EXOCLICK_OURDREAM_ZONE_PREFILTER_SCHEMA.json
2. HISTORICAL_EXOCLICK_EVIDENCE_INVENTORY.md
3. PREVIOUS_ZONE_PREFILTER_RECOVERY_REPORT.md
4. CURRENT_EXOCLICK_US_MOBILE_POPUNDER_INVENTORY.json
5. DOMAIN_INTENT_CLASSIFICATION_REPORT.md
6. ZONE_COMMERCIAL_INTENT_SCORING_REPORT.md
7-10. OURDREAM_INITIAL_ALLOWLIST.json / _CAPPED_EXPLORATION_LIST.json / _INITIAL_BLOCKLIST.json / _MANUAL_REVIEW_LIST.json
11. OURDREAM_ZONE_PREFILTER_MACHINE_READABLE.json
12. OURDREAM_ZONE_PREFILTER_OPERATOR_REVIEW.md
13. OURDREAM_ZONE_PREFILTER_FINAL_REPORT.md
14. CONTINUITY.json
15. evidence/ (validation, crawl, absence-search)
16. SHA256SUMS

## Pitfalls
- **SHA256SUMS auto-inclusion** : `find . -type f | xargs sha256sum > SHA256SUMS` peut inclure
  SHA256SUMS lui-même (le `>` le crée pendant le find) → entrée périmée, `sha256sum -c` FAILED.
  Correct :
  ```bash
  find . -type f ! -name "SHA256SUMS" -print0 | sort -z | xargs -0 sha256sum > /tmp/sha256.tmp && mv /tmp/sha256.tmp SHA256SUMS
  ```
- **API verrouillée** : ne PAS appeler l'API ExoClick tant que EXOCLICK_LIVE_ENABLED=false ;
  marquer l'inventaire sites/zones UNAVAILABLE et s'appuyer sur les refs locaux. La mission suivante
  (inventory enrichment) exige une autorisation opérateur explicite d'ouverture read-only.
- **Budget/bid jamais inventés** : le prefilter propose des loss caps par zone, jamais le budget final de campagne.
- **Validation 10 points** : parse JSON, conformité schéma, doublons (site+zone+domain+scope), conflits
  ALLOW/BLOCK, scan secrets (tokens, `t.vlmai-1.com`, `liveaccesshub.com/click`, `sub_id=`, `aff_id=`, `clickid=`),
  vérif métriques fabriquées (H doit être null partout), IDs int|null, evidence refs résolubles.
- **Memory lock plein** : si `memory replace` échoue avec "would put memory at X/Y chars", retirer d'abord une
  entrée obsolète (état de tâches daté, ex. liste de crons consolidés — pas durable) puis retenter. Les
  résumés de tâches datés appartiennent aux skills/session_search, pas à la mémoire.
- **Crawl bloqué ≠ classe inconnue** : Cloudflare 403/520 = classe par connaissance de marque avec confiance
  réduite, pas UNKNOWN_OR_MASKED (réservé aux domaines réellement non identifiables).
