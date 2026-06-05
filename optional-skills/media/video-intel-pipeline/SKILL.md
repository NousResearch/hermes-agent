---
name: video-intel-pipeline
description: "Descarga y transcribe videos (YouTube y miles de sitios) con yt-dlp + ffmpeg + faster-whisper, con búsqueda híbrida keyword + semántica."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [video, transcription, yt-dlp, ffmpeg, faster-whisper, whisper, asr, fastembed, embeddings, semantic-search, fts5, sqlite, youtube, rag, media]
    category: media
    related_skills: [youtube-content]
    requires_toolsets: [terminal]
---

# Video Intel Pipeline

Descarga el audio de un video, lo transcribe localmente con `faster-whisper` y
construye índices de búsqueda (keyword FTS5 + embeddings semánticos) para poder
analizar y **buscar dentro** del contenido con timestamps exactos.

## Cuándo usar

Usa este skill **siempre** que el usuario pida:
- analizar un video,
- aprender de un video,
- buscar algo dentro de un video,
- extraer transcript/citas/capítulos de un video.

Soporta YouTube y muchas otras plataformas mediante `yt-dlp`.

> Para solo el transcript de YouTube en texto (sin descargar audio ni ASR local),
> existe el skill `youtube-content`. Este skill descarga el audio real y transcribe
> on-device, por lo que funciona en cualquier sitio soportado por yt-dlp y no
> depende de que la plataforma exponga subtítulos.

## Tools / dependencias requeridas

Este pipeline **requiere** las siguientes herramientas (de ahí su naturaleza
"optional" / heavyweight):

| Tool | Tipo | Para qué |
|------|------|----------|
| `ffmpeg` | binario de sistema | convertir el audio descargado a WAV mono 16 kHz |
| `yt-dlp` | paquete pip | resolver y descargar el audio (1000+ sitios) |
| `faster-whisper` | paquete pip | transcripción / traducción ASR local en CPU (int8) |
| `fastembed` | paquete pip (opcional) | embeddings para búsqueda semántica (`--semantic`) |
| `numpy` | paquete pip (opcional) | álgebra de los embeddings (`--semantic`) |

`faster-whisper` y `fastembed` descargan modelos en el primer run (cachean luego).
`sqlite3` (stdlib) provee el índice keyword FTS5; no requiere instalación.

## Setup (PEP668-safe)

```bash
python3 -m venv /tmp/video-intel-venv
/tmp/video-intel-venv/bin/pip install -U pip yt-dlp faster-whisper fastembed numpy
# ffmpeg debe estar instalado en el sistema:
#   Debian/Ubuntu: sudo apt install ffmpeg   |   macOS: brew install ffmpeg
```

## Arquitectura

1. URL entra.
2. `yt-dlp` resuelve y descarga solo audio (`bestaudio/best`, `noplaylist`).
3. `ffmpeg` convierte a WAV mono 16 kHz.
4. `faster-whisper` transcribe localmente en CPU (o traduce a inglés con `--task translate`).
5. Se construyen índices: keyword (SQLite FTS5) y, con `--semantic`, embeddings (`fastembed`).
6. Artefactos en el `workdir` del video:
   - `metadata.json`
   - `transcript.txt`
   - `segments.json` (timestamps)
   - `search.db` (índice keyword FTS5)
   - `embeddings.npy` + `semantic_index.json` (si `--semantic`)
   - `result.json` (resumen estructural de ejecución)

## Scripts

Ruta del skill instalado: `~/.hermes/skills/media/video-intel-pipeline/scripts/`

- `video_intel_v2.py` — **recomendado**: ingest + búsqueda híbrida (keyword + semantic con RRF).
- `video_intel_pipeline.py` — v1 simple: solo descarga + transcripción.

`SCRIPT` abajo asume el venv del setup; sustitúyelo por `python3` si yt-dlp,
faster-whisper, etc. ya están en el entorno activo.

### v2 (recomendado)

```bash
SCRIPT="/tmp/video-intel-venv/bin/python ~/.hermes/skills/media/video-intel-pipeline/scripts/video_intel_v2.py"

# 1) Ingesta + transcripción + índice keyword + embeddings semánticos
$SCRIPT ingest \
  --url "https://youtu.be/VIDEO_ID" \
  --outdir /tmp/video-intel \
  --semantic

# 2) Buscar dentro del video (híbrido keyword + semantic)
$SCRIPT search \
  --workdir "/tmp/video-intel/<VIDEO_FOLDER>" \
  --query "pricing y costos por anuncio" \
  --top-k 8 \
  --semantic
```

Opciones de `ingest`: `--model tiny|base|small|medium|large-v3` (default `small`),
`--task transcribe|translate`, `--language es` (forzar idioma origen),
`--semantic` + `--semantic-model` (default `BAAI/bge-small-en-v1.5`).

### v1 (simple)

```bash
PY=/tmp/video-intel-venv/bin/python
SCRIPT=~/.hermes/skills/media/video-intel-pipeline/scripts/video_intel_pipeline.py

# Transcripción por defecto
$PY $SCRIPT --url "https://youtu.be/VIDEO_ID" --outdir /tmp/video-intel

# Forzar idioma de entrada
$PY $SCRIPT --url "https://vimeo.com/..." --language es --outdir /tmp/video-intel

# Traducir a inglés durante ASR
$PY $SCRIPT --url "https://youtu.be/VIDEO_ID" --task translate --outdir /tmp/video-intel
```

## Flujo de uso en Hermes

1. Ejecuta el script y valida `result.json`.
2. Si hay transcript vacío, reporta causa real (DRM, URL privada, audio mudo, etc.).
3. Para "analizar/aprender", resume con: tesis, ideas accionables, riesgos, citas con timestamp.
4. Para "buscar algo en el video", usa `search` (o filtra `segments.json` por keyword) y devuelve timestamps exactos.

## Pitfalls

- Si falta `ffmpeg`: falla la conversión a WAV.
- Si el host bloquea plataformas: `yt-dlp` puede requerir cookies/proxy (`--cookies-from-browser`).
- `faster-whisper` y `fastembed` descargan modelos; el primer run tarda más.
- ASR fuerza **CPU + int8** para portabilidad (evita dependencia de CUDA/libcublas).
- `--semantic` degrada con gracia: si faltan `fastembed`/`numpy`, sigue con keyword y reporta el error en `result.json`.
- Videos muy largos: la transcripción es O(duración); usa un `--model` menor para abaratar.

## Verificación mínima antes de responder

- `result.json.success == true`
- `segments_count > 0`
- `duration_seconds > 0`
- incluir al menos 2 timestamps en la respuesta al usuario cuando pida análisis profundo.
