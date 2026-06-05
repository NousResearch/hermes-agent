# YouTube Video Extraction Pipeline

Pipeline complet pour télécharger, transcrire, résumer et archiver une vidéo YouTube
dans le knowledge base. Utilise un pattern kanban **two-phase** (comme Mega) pour
isoler le travail CPU (whisper) du travail LLM (résumé).

**Global rules** (background execution, whisper, rate limits, transcription persistence):
`references/video-pipeline-global.md`

## Prérequis

- `yt-dlp` (pip) — **toujours utiliser `--js-runtimes node`** (YouTube bloque les IP datacenter via n-sig challenge, même avec cookies)
- `faster-whisper` (pip) — `large-v3` obligatoire (mandatory for all video content).
- `pyannote.audio>=4.0` (pip) — téléchargement unique du modèle `pyannote/speaker-diarization-3.1`, token HuggingFace requis pour l'accès initial uniquement (gratuit, tout tourne en local ensuite). Version 4.x nécessaire avec torch ≥2.5 (3.x crash sur `AudioMetaData` / `list_audio_backends` manquants). Pin exact: `pip install 'pyannote.audio>=4.0'`.
- `ffmpeg` (système)
- `minio` Python client (pip) — pour l'upload MinIO
- `node` ≥ v20 (système) — requis par yt-dlp pour le n-sig challenge solver
- Cookies YouTube : `/root/.hermes/cookies/yt_cookies.txt` (exportés depuis le navigateur desktop de l'utilisateur)
- HuggingFace token : dans `HF_TOKEN` (env) — généré gratuitement sur huggingface.co/settings/tokens, sert UNIQUEMENT à télécharger le modèle pyannote (pas d'appel API, pas de télémétrie). Sur les profils kanban, source from researcher-videos: `export HF_TOKEN=$(grep -oP 'HF_TOKEN=\K[^#\n]+' /root/.hermes/profiles/researcher-videos/.env | head -1)`

## Kanban two-phase pattern

Chaque vidéo = 2 tickets chaînés avec `--parent` :

```
Ticket 1: KB: Series Name — Ep.X [DOWNLOAD+TRANSCRIBE]  → Phase A (mécanique)
Ticket 2: KB: Series Name — Ep.X [RESUME+NOTE+ARCHIVE]  → Phase B (LLM)
```

Tous chaînés en série avec `--parent` : `1A → 1B → 2A → 2B → ...`
Assignee : `researcher-videos`. Max 2 vidéos par worker session.

Slug format : `chaine_titre-simplifie`. Exemple : `matt-pocock_ai-coding-workflow`.

### Phase A : Download + Diarize + Transcribe (mécanique, pas de LLM)

```
Ticket: KB: Series Name — Ep.X [DOWNLOAD+TRANSCRIBE]
Assignee: researcher-videos
Body:
  1. Extraire métadonnées YouTube (titre, chaîne, durée, vues, date)
  2. Lister les chapitres YouTube natifs (si dispos)
  3. Télécharger vidéo : WebM VP9+Opus 720p → /tmp/yt_SLUG.webm
  4. Extraire audio : deux qualités
     a) WAV 16kHz mono → /tmp/yt_SLUG_16k.wav (pour whisper)
     b) WAV 8kHz mono → /tmp/yt_SLUG_8k.wav (pour pyannote, ⚠️ doit être WAV pas MP3 — pyannote rejette le MP3 avec un mismatch de sample count)
  5. Diarization : pyannote sur audio 8kHz → segments avec étiquettes locuteur
  6. Transcription : faster-whisper `large-v3` int8 sur audio 16kHz, aligné sur segments pyannote → /tmp/yt_SLUG_transcript.json
  7. Chapitrage : chapitres natifs YouTube OU fallback NLP (gap > 3s)
  8. Cleanup : rm /tmp/yt_SLUG_8k.wav /tmp/yt_SLUG_16k.wav /tmp/yt_SLUG_diarization.json
     (GARDER /tmp/yt_SLUG.webm + /tmp/yt_SLUG_transcript.json)
  DO NOT: summarize, create note, upload MinIO, or push git.
```

### Phase B : Resume + Note + Archive (LLM, contexte propre)

```
Ticket: KB: Series Name — Ep.X [RESUME+NOTE+ARCHIVE]
Assignee: researcher-videos
Parent: <phase A ticket ID>
Body:
  1. Charger /tmp/yt_SLUG_transcript.json
  2. Charger la skill knowledge-base et suivre le prompt dans
     references/resume-prompt.md (deux passes)
  3. Note : Knowledge base/SLUG.md
  4. Upload MinIO : .webm, .mp3 (extrait), _transcript.json
  5. Cleanup ALL yt_SLUG.* from /tmp/
  6. Git push vault
```

## Commandes techniques (Phase A)

Ces commandes sont utilisées par le worker dans le ticket DOWNLOAD+TRANSCRIBE.

### 1. Lister les chapitres YouTube natifs

```bash
yt-dlp --cookies /root/.hermes/cookies/yt_cookies.txt --js-runtimes node \
  --print "%(chapters)s" \
  --sleep-requests 1 --sleep-interval 3 --max-sleep-interval 10 --limit-rate 4M \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

Sortie formatée en JSON avec `start_time`, `end_time`, `title`. Si vide → pas de chapitres natifs, fallback NLP.

### 2. Extraire les métadonnées

```bash
yt-dlp --cookies /root/.hermes/cookies/yt_cookies.txt --js-runtimes node \
  --print "%(title)s||%(uploader)s||%(duration)s||%(view_count)s||%(upload_date)s" \
  --sleep-requests 1 --sleep-interval 3 --max-sleep-interval 10 --limit-rate 4M \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

Sortie : `Titre||Chaîne||Durée (secondes)||Vues||YYYYMMDD`

### 3. Télécharger la vidéo (WebM VP9, max 720p)

```bash
yt-dlp --cookies /root/.hermes/cookies/yt_cookies.txt --js-runtimes node \
  -f "bestvideo[height<=720][vcodec^=vp9]+bestaudio[acodec^=opus]/bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --merge-output-format webm \
  -o "/tmp/yt_%(id)s.webm" \
  --sleep-requests 1 --sleep-interval 3 --max-sleep-interval 10 --limit-rate 4M \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Pitfall:** YouTube sert vidéo + audio en flux DASH séparés. `--merge-output-format webm` les fusionne. Si le merge échoue (codec incompatible), fallback : `-f "best[height<=720]"` qui prend le meilleur flux combiné natif.

### 4. Extraire l'audio (deux qualités)

Pyannote a besoin de 8kHz pour économiser la RAM, whisper a besoin de 16kHz pour la qualité.

```bash
# Audio 16kHz mono → whisper (qualité)
ffmpeg -y -i /tmp/yt_VIDEO_ID.webm -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/yt_VIDEO_ID_16k.wav

# Audio 8kHz mono → pyannote (dégradé intentionnel pour économiser la RAM, ⚠️ WAV obligatoire — MP3 rejeté par pyannote)
ffmpeg -y -i /tmp/yt_VIDEO_ID.webm -vn -acodec pcm_s16le -ar 8000 -ac 1 /tmp/yt_VIDEO_ID_8k.wav
```

**Pourquoi 8kHz pour pyannote ?** La diarization identifie les locuteurs par le timbre de la voix (fréquences sous 4 kHz). Les sifflantes (>6 kHz) ne contribuent pas à l'identité vocale. 8kHz vs 16kHz = 2-5% de différence de DER (imperceptible). RAM divisée par ~2 : 3-4 Go pour 9h au lieu de 6-10 Go.

### 5. Diarization (pyannote, étape 1/2 — MANDATORY: background+wait)

**Diarization is mandatory for ALL video content.** Always run pyannote — even for apparent monologues (guest introductions, Q&A segments, off-camera remarks are common in "solo" videos).

Pipeline manuel (pas whisperx) pour contrôle total : pyannote seul d'abord, whisper seul ensuite.
Les deux étapes sont **séquentielles** — jamais en même temps (sécurité RAM).

**Use the CANONICAL script** `scripts/diarize.py` from this skill. Do NOT generate ad-hoc scripts.

```bash
# Copy canonical script to /tmp/ and run
cp "$SKILL_DIR/scripts/diarize.py" /tmp/diarize.py

terminal(
    "python3 /tmp/diarize.py /tmp/yt_VIDEO_ID_8k.wav /tmp/yt_VIDEO_ID_diarization.json",
    background=True, notify_on_complete=True
)
process(action="wait", timeout=28800)  # 8h for long videos
read_file("/tmp/yt_VIDEO_ID_diarization.json", limit=10)  # verify
```

**Sortie :** `/tmp/yt_VIDEO_ID_diarization.json` — segments bruts avec étiquettes.

### 5b. Transcription (faster-whisper, étape 2/2 — MANDATORY: background+wait)

**Use the CANONICAL script** `scripts/transcribe.py` from this skill. Do NOT generate ad-hoc scripts.

```bash
# Copy canonical script to /tmp/ and run
cp "$SKILL_DIR/scripts/transcribe.py" /tmp/transcribe.py

terminal(
    "python3 /tmp/transcribe.py /tmp/yt_VIDEO_ID_16k.wav /tmp/yt_SLUG_transcript.json DURATION_SECS",
    background=True, notify_on_complete=True
)
process(action="wait", timeout=28800)  # 8h for long videos
read_file("/tmp/yt_SLUG_transcript.json", limit=10)  # verify
```

The script uses `large-v3` int8 on CPU with progress every 5 minutes.

### 5b-bis. Fusion diarization + transcription

Merge whisper segments with pyannote speaker labels:

```python
import json

with open("/tmp/yt_VIDEO_ID_diarization.json") as f:
    diar = json.load(f)
with open("/tmp/yt_SLUG_transcript.json") as f:
    trans = json.load(f)

for seg in trans["segments"]:
    seg_mid = (seg["start"] + seg["end"]) / 2
    for dia in diar["segments"]:
        if dia["start"] <= seg_mid <= dia["end"]:
            seg["speaker"] = dia["speaker"]
            break
    if "speaker" not in seg:
        seg["speaker"] = "Unknown"

with open("/tmp/yt_SLUG_transcript.json", "w") as f:
    json.dump(trans, f, indent=2, ensure_ascii=False)

print(f"Merged: {len(trans['segments'])} segments with speaker labels")
```

**Note RAM :** pyannote et whisper tournent séquentiellement (pas en même temps).
- Vidéo <1h : ~1.5 Go RAM
- Vidéo <3h : ~2 Go RAM
- Vidéo 9h : ~3-4 Go RAM (grâce au 8kHz pour pyannote)
Pas de découpage en chunks — étiquettes locuteur continues sur toute la durée.

### 5c. Overlap — gestion des chevauchements

Pyannote détecte l'overlap et produit des labels composites (`SPEAKER_00 | SPEAKER_01`).
Ces segments sont conservés avec une annotation.

Dans le JSON de sortie, ajouter un flag `overlap` :

```json
{
  "start": 45.2,
  "end": 48.7,
  "speaker": "SPEAKER_00 | SPEAKER_01",
  "text": "je pense que c'est— non mais tu comprends pas le problème...",
  "overlap": true
}
```

La note Obsidian marque ces segments avec `⚠️ Chevauchement` pour indiquer
une transcription moins fiable.

### 5d. Identification des locuteurs (depuis les métadonnées)

Tenter de mapper SPEAKER_00, SPEAKER_01 → vrais noms à partir des métadonnées YouTube.
Heuristiques simples :

1. **Description de la vidéo** : chercher les noms propres (personnes physiques)
2. **Titre de la chaîne** : si chaîne personnelle, le nom du créateur
3. **Titre de la vidéo** : patterns type "X interview Y", "X & Y discutent de..."

Si un mapping est trouvé, l'appliquer au JSON et à la note. Sinon → `"Unknown"`.

```python
# Exemple heuristique simple
import re

# Récupérer description depuis yt-dlp
desc = metadata.get("description", "")
# Chercher des noms propres (majuscule, 2+ mots)
names = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', desc))
# Si 2 noms trouvés et 2 speakers → mapping probable
if len(names) == 2 and len(speakers) == 2:
    mapping = dict(zip(sorted(speakers), sorted(names)))
```

### 6. Chapitrage (fallback NLP si pas de chapitres natifs)

Si l'étape 1 n'a pas retourné de chapitres :

```python
# Chapitrage NLP basé sur les gaps de pause et les shifts thématiques
import json, re

with open('/tmp/yt_VIDEO_ID_transcript.json') as f:
    data = json.load(f)

segments = data['segments']
full_text = ' '.join(s['text'] for s in segments)

# Détection simple : sauts > 3 secondes entre segments = limite potentielle de chapitre
chapters = []
current_start = segments[0]['start']
current_text = []

for i, seg in enumerate(segments):
    if i > 0:
        gap = seg['start'] - segments[i-1]['end']
        if gap > 3.0 and len(current_text) > 0:
            chapters.append({
                'start': current_start,
                'title': f'Section {len(chapters)+1}',
                'summary': ' '.join(current_text)[:500]
            })
            current_start = seg['start']
            current_text = []
    current_text.append(seg['text'])

# Dernier chapitre
if current_text:
    chapters.append({
        'start': current_start,
        'title': f'Section {len(chapters)+1}',
        'summary': ' '.join(current_text)[:500]
    })

print(json.dumps(chapters, indent=2))
```

## Commandes techniques (Phase B)

Ces commandes sont utilisées par le worker dans le ticket RESUME+NOTE+ARCHIVE.

### 7. Résumé approfondi (LLM)

Le worker charge la skill `knowledge-base` et suit le prompt dans
`references/resume-prompt.md` (deux passes). Template de note :
`references/youtube-note-template.md`. Note → `Knowledge base/<slug>.md`.

### 8. Uploader vers MinIO

```bash
# Extraire MP3 depuis le WebM
ffmpeg -y -i /tmp/yt_SLUG.webm -vn -acodec libmp3lame -q:a 2 /tmp/yt_SLUG.mp3

# Uploader les 3 fichiers
mc cp /tmp/yt_SLUG.webm minio/knowledge-base/videos/<slug>.webm
mc cp /tmp/yt_SLUG.mp3 minio/knowledge-base/videos/<slug>.mp3
mc cp /tmp/yt_SLUG_transcript.json minio/knowledge-base/videos/<slug>.json
```

### 9. Créer la note dans le vault

Template (voir `references/youtube-note-template.md`). Sauvegarder dans
`Knowledge base/<slug>.md`, puis push Git :

```bash
cd "$OBSIDIAN_VAULT_PATH" && git add -A && git commit -m "add: <slug>" && git push
```

### 10. Nettoyage

```bash
rm /tmp/yt_SLUG.webm /tmp/yt_SLUG.mp3 /tmp/yt_SLUG_transcript.json
```

## Rate limiting

- **Toujours** utiliser `--sleep-requests 1 --sleep-interval 3 --max-sleep-interval 10 --limit-rate 4M`
- **Toujours** utiliser `--js-runtimes node` (n-sig challenge)
- Max **2 vidéos** par worker session
- Au-delà de 2 URLs, sérialiser avec `--parent`
- Cookies fichier persistant `/root/.hermes/cookies/yt_cookies.txt` — ne pas supprimer

## Pre-flight & edge cases

Pre-flight checks (canonical scripts, single process per file, orphan workers): `video-pipeline-global.md`.
Operational branches: `edge-cases.md`.

## Platform-specific notes
- **Ticket body model:** Workers follow the ticket body. Include explicitly:
  `faster-whisper large-v3 int8 (CPU)` in every DOWNLOAD+TRANSCRIBE ticket.
- **n-sig challenge:** Sur IP datacenter, yt-dlp échoue avec "n challenge solving failed". Le flag `--js-runtimes node` est OBLIGATOIRE (Node ≥ v20 requis). Sans lui, yt-dlp ne voit que les images storyboard.
- **Bot detection:** Si yt-dlp retourne "Sign in to confirm you're not a bot", les cookies sont expirés. L'utilisateur doit les ré-exporter depuis son navigateur.
- **Format non trouvé:** Si VP9 720p non dispo, fallback `-f "best[height<=720]"` sur le meilleur format natif.
- **Audio > 2h:** Le WAV 16kHz fait ~700 MB pour 2h. Vérifier l'espace disque avant.
- **whisper OOM:** `large-v3` int8 utilise ~3 Go RAM. Vérifier ~4 Go libres avant lancement.
- **Diarization — HF token manquant:** Si whisperx échoue avec "gated repo", le `HF_TOKEN` n'est pas exporté. Le générer sur huggingface.co/settings/tokens (gratuit, 30s, token Read-only). **Trois modèles gated doivent être acceptés** (pas un seul) : accepter les conditions sur https://huggingface.co/pyannote/speaker-diarization-3.1 PUIS https://huggingface.co/pyannote/segmentation-3.0 PUIS https://huggingface.co/pyannote/speaker-diarization-community-1. Sans accepter les 3, le pipeline échoue avec 403 même avec un token valide. L'ordre d'acceptation n'a pas d'importance. Une seule fois — les modèles sont ensuite en cache local.
- **Diarization — RAM sur vidéos longues (>3h):** L'audio 8kHz limite pyannote à ~3-4 Go même sur 9h. Si OOM malgré tout, vérifier que l'audio 8kHz est bien utilisé (pas de fallback accidentel sur le 16kHz). Pas de découpage en chunks — on perdrait la continuité des étiquettes locuteur.
- **Diarization — qualité 8kHz:** Perte de 2-5% de DER seulement (imperceptible). Si deux voix sont très similaires (même sexe, même registre), la confusion est possible mais rare — ça arrive aussi en 16kHz.
- **Overlap — segments composites:** Pyannote produit `SPEAKER_00 | SPEAKER_01` quand deux personnes parlent simultanément. Ces segments sont conservés avec `overlap: true` dans le JSON et `⚠️ Chevauchement` dans la note Obsidian. La transcription whisper est moins fiable sur ces segments (mélange de voix).
- **Overlap — traitement dans la note:** Le worker Phase B doit détecter `overlap: true` dans le JSON et préfixer le texte avec `⚠️ Chevauchement` dans la note finale. Pas de suppression — on garde l'info même bruitée.
- **Identification locuteur — limites:** L'heuristique basée sur la description/titre/channel est approximative. Ne pas forcer un mapping si ambigu. `"Unknown"` est préférable à une identification erronée.
- **Chapitres vides:** YouTube peut lister des chapitres sans titre. Les ignorer et fallback NLP.
- **Vidéo privée/non listée:** yt-dlp échoue. Le worker doit catcher l'erreur et notifier.
- **Ticket body updates:** When upgrading model in tickets, patch existing bodies to `large-v3 int8` and reclaim.
- **Diarization — API pyannote v4:** pyannote 4.x returns `DiarizeOutput`. Use `diarization.speaker_diarization.itertracks(yield_label=True)`. pyannote 3.x returned `Diarization` with direct `.itertracks()`. The `DiarizeOutput` object replaces the old `Diarization`. Version 4.x is required with torch >=2.5. The parameter `use_auth_token=` was renamed to `token=`.

### Performance benchmarks

Mesuré sur ce serveur (CPU only, 6 vCPU AMD EPYC, no GPU, 8 GB RAM).
Pipeline manuel (pyannote 4.x diarization + faster-whisper `large-v3` int8), séquentiel.

⚠️ **pyannote 4.x est plus lent que 3.x.** Le passage à pyannote >=4.0 (nécessaire pour compatibilité torch >=2.5) a fait passer la diarization de ~0.1× temps réel (v3.x) à ~2-3× temps réel (v4.x) sur CPU. Les benchmarks ci-dessous reflètent pyannote 4.x.

### Première exécution (cold start — compilation PyTorch + cache HF + large-v3 download)

| Vidéo | Audio réel | Diarization | Transcription | Total | Ratio |
|-------|-----------|-------------|---------------|-------|-------|
| 30s (Reel test, 4 locuteurs) | 30s | ~50s | ~0.6s | ~51s | 1.7× |

**Premier run pyannote :** ~1.5-2× temps réel à 16kHz. Prévoyez ~1× temps réel à 8kHz.
La compilation JIT PyTorch et le téléchargement des poids dominent le premier run.
Le premier téléchargement de `large-v3` (= 3 Go) s'ajoute à ce premier run.

### Exécutions suivantes (warm start, large-v3, cpu_threads=6)

Mesures réelles 2026-05-27 (3 vidéos YouTube Biomécanique, 95-121 min chaque, 6 vCPU AMD EPYC, 11 GB RAM).

| Vidéo | Durée | Diarization | Transcription | Total |
|-------|-------|-------------|---------------|-------|
| GuXBKGfoA8M | 95 min | ~4-6h (400-500% CPU, 3.5 GB) | ~4-5h (94% CPU, 3.6 GB) | ~8-11h |
| St0BgcNS7A0 | 121 min | ~4-6h (416% CPU, 2.2 GB) | ~4-5h | ~8-11h |
| NmURuySSpII | 116 min | ~4-6h | ~4-5h | ~8-11h |

### Profil par étape (identifiable via `ps aux`)

| Étape | CPU% | RAM | Processus |
|-------|------|-----|-----------|
| **Diarization** (pyannote) | 400-500% (4-5 cœurs parallèles) | 2-3.5 Go | `python3 /tmp/transcribe.py ...` (phase diarization) |
| **Transcription** (large-v3) | ~94-300% (cpu_threads=6 mais décodeur single-thread) | 3-4 Go | `python3 /tmp/transcribe.py ...` (phase whisper) |
| **Download** (yt-dlp) | ~50% | ~200 Mo | `yt-dlp ...` |
| **Conversion** (ffmpeg) | ~100% | ~100 Mo | `ffmpeg -i ... .webm → .wav` |

**Comment identifier l'étape en cours :**
- CPU > 300% = **diarization** pyannote (parallélisé sur tous les cœurs)
- CPU ~90-100% = **transcription** whisper (décodeur single-thread, encodeur seul bénéficie de cpu_threads)
- CPU ~50% + réseau = **download** yt-dlp
- CPU ~100% + courte durée = **conversion** ffmpeg

### Règles d'estimation (warm, pyannote 4.x, cpu_threads=6)

- Diarization pyannote 4.x (8kHz) : **~2-3× la durée audio**
- Transcription whisper `large-v3` (16kHz, cpu_threads=6) : **~2-3× la durée audio** (réduit de ~5× sans cpu_threads)
- Total pipeline : **~5-6× la durée audio** (séquentiel : diarization puis transcription)
- La durée audio réelle est souvent plus courte que la durée vidéo (silences, pauses)
- **Premier run toujours plus lent** (×2-5 à 16kHz) → les workers en cron amortissent
- **Les 3 vidéos de cette session (95-121 min) : ~8-11h chacune.** Vidéo 2h → prévoir 10-14h.
- **RAM :** diarization ~2-3.5 Go, transcription ~3-4 Go. Les deux sont séquentielles → pic à ~4 Go. Vérifier ~5 Go libres avant lancement.
