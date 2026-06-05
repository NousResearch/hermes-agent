# MiniMax Fallback — cuando Higgsfield no está disponible

Si `higgsfield account status` responde `Not authenticated` o
`Session expired` y el usuario no puede hacer login interactivo:

## Instalación

```bash
npm install -g mmx-cli
# Verificar:
mmx auth status
```

NOTA: `mmx` es un CLI de Node.js. En este host está instalado globalmente
via npm. Si `mmx` no está en PATH, usar `npx mmx-cli <command>`.

## Autenticación

La API key está en Infisical (secret `MINIMAX_API_KEY`):

```bash
MINIMAX_KEY="$MINIMAX_API_KEY"
npx mmx-cli auth login --api-key "$MINIMAX_KEY"
```

Región: auto-detectada (global para este host).
Cuota: 3 videos/día, 21/semana, 99% general.

## Comandos útiles

```bash
# Estado
mmx auth status
mmx quota

# Imagen
mmx image generate --prompt "..." --aspect-ratio 9:16 --out /tmp/img.png

# Música instrumental (para background de reels)
mmx music generate --prompt "cinematic orchestral" --instrumental --out /tmp/music.mp3

# Música con letra auto-generada
mmx music generate --prompt "uplifting pop" --lyrics-optimizer --out /tmp/song.mp3

# Video (alternativa para animaciones simples)
mmx video generate --prompt "ocean waves at sunset" --out /tmp/vid.mp4
```

## Limitaciones conocidas

- `music-2.6` genera canciones completas (~1:30) incluso con `--instrumental`.
  Recortar con FFmpeg: `ffmpeg -i music.mp3 -t 9 -acodec copy trimmed.mp3`
- No tiene control granular de duración en music gen.
- Para música exacta de 9s, usar FFmpeg synths (ver `references/original-reel-composition.md`
  en `instagram-research-toolkit`).

## Referencia

- MiniMax CLI GitHub: https://github.com/MiniMax-AI/cli
- API docs: https://platform.minimax.io/docs/api-reference
