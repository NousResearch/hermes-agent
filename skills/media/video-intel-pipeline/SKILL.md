---
name: video-intel-pipeline
description: "Descarga y transcribe videos (YouTube y miles de sitios) con yt-dlp + ffmpeg + faster-whisper para análisis confiable."
platforms: [linux, macos]
---

# Video Intel Pipeline

## Cuándo usar

Usa este skill **siempre** que el usuario pida:
- analizar un video,
- aprender de un video,
- buscar algo dentro de un video,
- extraer transcript/citas/capítulos de un video.

Soporta YouTube y muchas otras plataformas mediante `yt-dlp`.

## Arquitectura

1. URL entra.
2. `yt-dlp` resuelve y descarga solo audio.
3. `ffmpeg` convierte a WAV mono 16kHz.
4. `faster-whisper` transcribe localmente (o traduce a inglés si se pide).
5. Se guardan artefactos:
   - `metadata.json`
   - `transcript.txt`
   - `segments.json` (timestamps)
   - `result.json` (resumen estructural de ejecución)

## Setup recomendado (PEP668-safe)

```bash
python3 -m venv /tmp/video-intel-venv
/tmp/video-intel-venv/bin/pip install -U pip yt-dlp faster-whisper fastembed numpy
# ffmpeg debe estar instalado en el sistema
```

## Script

Usar:
- `scripts/video_intel_pipeline.py` (v1 simple)
- `scripts/video_intel_v2.py` (**recomendado**, con búsqueda híbrida)

### Ejemplos v2 (recomendado)

```bash
# 1) Ingesta + transcripción + índice keyword + embeddings semánticos
python SKILL_DIR/scripts/video_intel_v2.py \
  ingest \
  --url "https://youtu.be/VIDEO_ID" \
  --outdir /tmp/video-intel \
  --semantic

# 2) Buscar dentro del video (híbrido keyword + semantic)
python SKILL_DIR/scripts/video_intel_v2.py \
  search \
  --workdir "/tmp/video-intel/<VIDEO_FOLDER>" \
  --query "pricing y costos por anuncio" \
  --top-k 8 \
  --semantic
```

### Ejemplos v1

```bash
# Transcripción por defecto
/tmp/video-intel-venv/bin/python SKILL_DIR/scripts/video_intel_pipeline.py \
  --url "https://youtu.be/VIDEO_ID" \
  --outdir /tmp/video-intel

# Forzar idioma de entrada
/tmp/video-intel-venv/bin/python SKILL_DIR/scripts/video_intel_pipeline.py \
  --url "https://vimeo.com/..." \
  --language es \
  --outdir /tmp/video-intel

# Traducir a inglés durante ASR
/tmp/video-intel-venv/bin/python SKILL_DIR/scripts/video_intel_pipeline.py \
  --url "https://youtu.be/VIDEO_ID" \
  --task translate \
  --outdir /tmp/video-intel
```

## Flujo de uso en Hermes

1. Ejecuta el script y valida `result.json`.
2. Si hay transcript vacío, reporta causa real (DRM, URL privada, audio mudo, etc.).
3. Para "analizar/aprender", resume con:
   - tesis,
   - ideas accionables,
   - riesgos,
   - citas con timestamp.
4. Para "buscar algo en el video", filtra en `segments.json` por keywords y devuelve timestamps exactos.

## Pitfalls

- Si `ffmpeg` falta: falla la conversión.
- Si el host bloquea plataformas: `yt-dlp` puede requerir cookies/proxy.
- `faster-whisper` descarga modelos; el primer run tarda más.
- Videos muy largos: usa `--max-duration` si necesitas recortar costos.

## Verificación mínima antes de responder

- `result.json.success == true`
- `segments_count > 0`
- `duration_seconds > 0`
- incluir al menos 2 timestamps en la respuesta al usuario cuando pida análisis profundo.
