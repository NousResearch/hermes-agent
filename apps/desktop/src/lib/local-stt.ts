import { atom } from 'nanostores'

// Optional local dictation. When `voice.dictation.stt` is "local" the renderer
// posts recorded audio straight at an OpenAI-compatible Whisper server on
// loopback instead of the backend's /api/audio/transcribe.
//
// This is a client-side shortcut, not a second STT stack: the backend stays
// authoritative (sessions, echo transcripts, provider config are untouched) and
// is always the next rung down. Per AGENTS.md "Cross everything as an
// observable ladder", a failed *read* falls to the next rung — so every local
// failure (no server, non-2xx, junk body, timeout) silently retries through the
// backend and the user never sees a local-server error.

/** Whisper.cpp's `--port` default, and the port the docs suggest for the
 *  sidecar server; overridable via `voice.dictation.local_stt_port`. */
export const LOCAL_STT_DEFAULT_PORT = 8090

/** Loopback either answers or is not there: a refused connection returns
 *  instantly, so this floor only ever elapses for a server that took the socket
 *  and then stalled. Keeps dictation snappy when the sidecar is dead or wedged. */
export const LOCAL_STT_MIN_TIMEOUT_MS = 2_500

/** Past this, waiting on the local rung costs more than the backend roundtrip
 *  it would fall back to. */
export const LOCAL_STT_MAX_TIMEOUT_MS = 60_000

// Budget derived from clip size, mirroring audioTranscribeRequestTimeoutMs():
// the base64 data URL's length tracks clip length, and ~0.1ms/char allows ~2s
// of transcription per 1s of audio. Only the floor and cap differ from the
// backend rung — a flat timeout would bound the transcription itself, so a
// genuinely slow-but-alive server would abort on every long clip and "local"
// would quietly mean "never local" for anything but the shortest takes.
const LOCAL_STT_TIMEOUT_MS_PER_CHAR = 0.1

export function localSttTimeoutMs(dataUrl: string): number {
  const estimated = Math.max(
    LOCAL_STT_MIN_TIMEOUT_MS,
    Math.ceil(String(dataUrl || '').length * LOCAL_STT_TIMEOUT_MS_PER_CHAR)
  )

  return Math.min(LOCAL_STT_MAX_TIMEOUT_MS, estimated)
}

/** OpenAI-compatible servers require a `model` part. whisper.cpp's server and
 *  faster-whisper-server ignore it (the loaded model wins), so this default
 *  works unconfigured — but model-routing servers (speaches, LocalAI) dispatch
 *  on it and need a real id, hence `voice.dictation.local_stt_model`.
 *
 *  Deliberately NOT `stt.local.model`: that key is a faster-whisper size
 *  ("base"), which a routing server would reject as a model id and turn every
 *  dictation into a silent fallback. */
export const LOCAL_STT_DEFAULT_MODEL = 'whisper-1'

export const DICTATION_STT_MODES = ['backend', 'local'] as const

export type DictationSttMode = (typeof DICTATION_STT_MODES)[number]

export interface LocalSttSettings {
  mode: DictationSttMode
  port: number
  /** `model` part sent in the multipart upload. */
  model: string
  /** Global `stt.language` hint, forwarded when set; '' means auto-detect. */
  language: string
}

export const LOCAL_STT_DEFAULTS: LocalSttSettings = {
  mode: 'backend',
  port: LOCAL_STT_DEFAULT_PORT,
  model: LOCAL_STT_DEFAULT_MODEL,
  language: ''
}

/** The slice of config.yaml this reads. Values are `unknown` because config is
 *  whatever the user wrote — a hand-edited port can arrive as a string. */
interface DictationConfigShape {
  stt?: { language?: unknown } | null
  voice?: {
    dictation?: { stt?: unknown; local_stt_port?: unknown; local_stt_model?: unknown } | null
  } | null
}

const trimmed = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

// Both shapes resolve through Number(), not parseInt(): parseInt("1.5") is 1 —
// a privileged port — and parseInt("9000.5") silently truncates, while the
// numeric path rejects both as non-integers. Whether a hand-edited port is
// valid must not depend on whether the user quoted it in config.yaml.
function sttPort(value: unknown): number {
  const port = typeof value === 'number' ? value : Number(trimmed(value))

  return Number.isInteger(port) && port > 0 && port < 65_536 ? port : LOCAL_STT_DEFAULT_PORT
}

/** Single resolver for the dictation policy, so the settings UI, the store, and
 *  transcribeAudio() can never disagree about what the config means. */
export function localSttSettingsFromConfig(config: DictationConfigShape | null | undefined): LocalSttSettings {
  const dictation = config?.voice?.dictation

  return {
    mode: trimmed(dictation?.stt).toLowerCase() === 'local' ? 'local' : 'backend',
    port: sttPort(dictation?.local_stt_port),
    // A blank/garbage model would make servers that ignore the part keep
    // working while routing servers 400 — resolve to the default instead.
    model: trimmed(dictation?.local_stt_model) || LOCAL_STT_DEFAULT_MODEL,
    language: trimmed(config?.stt?.language)
  }
}

// Read synchronously by transcribeAudio() on every dictation, so the backend
// path stays a plain call with no extra await (and no config fetch) in front of
// it. Seeded from config alongside the other voice prefs on mount/refresh.
export const $localStt = atom<LocalSttSettings>(LOCAL_STT_DEFAULTS)

/** Seed the dictation settings from a loaded config payload (mount / refresh). */
export function applyLocalSttFromConfig(config: DictationConfigShape | null | undefined) {
  $localStt.set(localSttSettingsFromConfig(config))
}

export const localSttEndpoint = (port: number) => `http://127.0.0.1:${port}/v1/audio/transcriptions`

// Servers that shell out to ffmpeg key the decoder off the upload's extension,
// so the filename is not cosmetic — a .webm clip named ".bin" fails to decode.
const AUDIO_EXTENSIONS: Record<string, string> = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/flac': 'flac'
}

function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(base64)
  const bytes = new Uint8Array(new ArrayBuffer(binary.length))

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }

  return bytes
}

/** The recorder hands dictation around as a data URL (that's what the backend
 *  endpoint takes); multipart needs real bytes, so decode rather than re-encode. */
export function audioFileFromDataUrl(dataUrl: string, mimeType?: string): File {
  const comma = dataUrl.indexOf(',')

  if (!dataUrl.startsWith('data:') || comma < 0) {
    throw new Error('Local STT: unsupported audio payload')
  }

  const header = dataUrl.slice('data:'.length, comma)
  const payload = dataUrl.slice(comma + 1)
  const type = trimmed(mimeType) || header.split(';')[0] || 'audio/webm'

  const bytes: Uint8Array<ArrayBuffer> = header.includes(';base64')
    ? base64ToBytes(payload)
    : new Uint8Array(new TextEncoder().encode(decodeURIComponent(payload)))

  return new File([bytes], `dictation.${AUDIO_EXTENSIONS[type] ?? 'webm'}`, { type })
}

/**
 * POST the clip to the local OpenAI-compatible transcription endpoint and
 * return its text. Throws on anything that isn't a usable transcript — callers
 * treat every throw as "fall back to the backend".
 */
export async function transcribeWithLocalStt(
  dataUrl: string,
  mimeType: string | undefined,
  settings: LocalSttSettings
): Promise<string> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), localSttTimeoutMs(dataUrl))

  try {
    const form = new FormData()

    form.append('file', audioFileFromDataUrl(dataUrl, mimeType))
    form.append('model', settings.model)

    if (settings.language) {
      form.append('language', settings.language)
    }

    const response = await fetch(localSttEndpoint(settings.port), {
      method: 'POST',
      body: form,
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error(`Local STT responded ${response.status}`)
    }

    const payload: unknown = await response.json()
    const text = (payload as { text?: unknown } | null)?.text

    if (typeof text !== 'string') {
      throw new Error('Local STT returned no transcript')
    }

    return text
  } finally {
    clearTimeout(timer)
  }
}
