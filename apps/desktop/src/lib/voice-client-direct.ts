import { profileScoped } from '@/api/client'
import { getApiRequestConnection, getApiRequestProfile, hermesApi } from '@/hermes'

/**
 * Client-direct voice: call the active profile's STT/TTS providers straight
 * from the desktop, cutting the audio relay hop through the gateway.
 *
 * The gateway stays the single source of truth for WHICH provider and WHICH
 * credentials to use — `GET /api/audio/voice-config` returns the same
 * resolution the gateway's own relay endpoints would apply (see
 * `tools/voice_client_config.py`). This module only executes the provider
 * call locally. Direction of travel:
 *
 *   mic → provider (audio up, once) → text → gateway   (STT)
 *   gateway → text (already streaming) → provider → speaker   (TTS)
 *
 * Keys live in renderer MEMORY only — never persisted, never logged. A
 * provider that can only run on the gateway host resolves to
 * `{mode:'relay'}` and callers fall back to the existing relay endpoints.
 */

/** Mono 16-bit PCM at this rate: what WAV-only endpoints (Meta) accept, and the smaller of the two. */
const STT_WAV_SAMPLE_RATE = 16000

export interface DirectSttConfig {
  mode: 'direct'
  wire: 'elevenlabs-stt' | 'openai-multipart' | 'xai-stt'
  provider: string
  base_url: string
  api_key: string
  model: null | string
  language: null | string
  /**
   * Transcription body format the provider accepts, resolved server-side (`json` for OpenRouter,
   * which 400s on `text`; `text` for OpenAI/Groq). Optional: absent means the pre-existing `text`.
   */
  response_format?: null | string
  /**
   * Container the model requires for the upload. `wav` = the recorder's WebM/Opus is refused
   * (Meta's transcription endpoint: "requires WAV audio", mono at 16/24 kHz), so the blob is
   * re-encoded here. Absent = post the recording as-is.
   */
  audio_format?: 'wav' | null
}

export interface DirectTtsConfig {
  mode: 'direct'
  wire: 'elevenlabs-tts' | 'openai-speech'
  provider: string
  base_url: string
  api_key: string
  model: null | string
  voice: null | string
  speed: null | number
}

interface RelayConfig {
  mode: 'relay'
  reason?: string
}

export interface VoiceClientConfig {
  stt: DirectSttConfig | RelayConfig
  tts: DirectTtsConfig | RelayConfig
}

// ---------------------------------------------------------------------------
// Config fetch + cache. Keyed by (connection, profile) so a profile/backend
// switch never reuses another scope's credentials; TTL'd so a config change
// on the gateway propagates within a minute without a per-utterance fetch.
// ---------------------------------------------------------------------------

const CONFIG_TTL_MS = 60_000

let cached: { key: string; at: number; config: VoiceClientConfig } | null = null
let inflight: { key: string; promise: Promise<null | VoiceClientConfig> } | null = null

function scopeKey(): string {
  return `${getApiRequestConnection() ?? 'local'}::${getApiRequestProfile() ?? 'default'}`
}

/** Drop cached credentials (used by tests; scope changes rotate the key). */
export function clearVoiceClientConfigCache(): void {
  cached = null
  inflight = null
}

export async function fetchVoiceClientConfig(): Promise<null | VoiceClientConfig> {
  const key = scopeKey()

  if (cached && cached.key === key && Date.now() - cached.at < CONFIG_TTL_MS) {
    return cached.config
  }

  if (inflight && inflight.key === key) {
    return inflight.promise
  }

  const promise = (async () => {
    try {
      // hermesApi carries connectionScoped(); profileScoped() adds the
      // profile — the same routing every relay audio call uses, so the
      // config comes from the backend the user is actually talking to.
      const response = await hermesApi<{ ok: boolean } & VoiceClientConfig>({
        ...profileScoped(),
        path: '/api/audio/voice-config'
      })

      if (!response?.ok || !response.stt || !response.tts) {
        return null
      }

      const config: VoiceClientConfig = { stt: response.stt, tts: response.tts }
      cached = { key, at: Date.now(), config }

      return config
    } catch {
      // Older backend without the endpoint / transient failure → relay.
      return null
    } finally {
      inflight = null
    }
  })()

  inflight = { key, promise }

  return promise
}

// ---------------------------------------------------------------------------
// STT — audio blob → transcript, provider-direct.
// ---------------------------------------------------------------------------

/** Mono 16-bit PCM WAV from float samples: RIFF header + PCM data chunk. */
export function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const dataBytes = samples.length * 2
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)

  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i))
    }
  }

  ascii(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM fmt chunk size
  view.setUint16(20, 1, true) // format = PCM
  view.setUint16(22, 1, true) // channels = mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  ascii(36, 'data')
  view.setUint32(40, dataBytes, true)

  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
  }

  return new Blob([view], { type: 'audio/wav' })
}

/**
 * Re-encode a recording as mono WAV at `sampleRate`, for models that refuse WebM/Opus.
 *
 * Decoding happens with WebAudio (the browser always decodes what it just recorded), and the
 * resample happens in an OfflineAudioContext at the required rate — no ffmpeg, no server hop.
 */
export async function toWavMono(audio: Blob, sampleRate: number): Promise<Blob> {
  const AudioCtor = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext
  const OfflineCtor = (globalThis as { OfflineAudioContext?: typeof OfflineAudioContext })
    .OfflineAudioContext

  if (!AudioCtor || !OfflineCtor) {
    throw new Error('This build cannot re-encode audio to WAV (no WebAudio available)')
  }

  const decodeContext = new AudioCtor()

  try {
    const decoded = await decodeContext.decodeAudioData(await audio.arrayBuffer())
    const frames = Math.max(1, Math.ceil(decoded.duration * sampleRate))
    const renderer = new OfflineCtor(1, frames, sampleRate)
    const source = renderer.createBufferSource()

    source.buffer = decoded
    source.connect(renderer.destination)
    source.start()

    const rendered = await renderer.startRendering()

    return encodeWavPcm16(rendered.getChannelData(0), sampleRate)
  } finally {
    void decodeContext.close?.()
  }
}

function sttFileName(audio: Blob): string {
  const subtype = (audio.type.split(';')[0].split('/')[1] || 'webm').toLowerCase()

  return `recording.${subtype === 'mpeg' ? 'mp3' : subtype}`
}

async function providerErrorText(response: Response): Promise<string> {
  const body = await response.text().catch(() => '')

  try {
    const parsed = JSON.parse(body)
    const detail = parsed?.error?.message ?? parsed?.detail ?? parsed?.error

    if (typeof detail === 'string' && detail) {
      return detail
    }
  } catch {
    // fall through to raw body
  }

  return body.slice(0, 300)
}

/**
 * Normalize an OpenAI-compatible transcription HTTP body to spoken text.
 *
 * Groq and OpenAI honor `response_format=text` and return a bare string.
 * Mistral Voxtral ignores that flag and returns JSON
 * `{ text, model, usage, ... }` — dumping that object into the Desktop
 * composer is the dictation regression (plain speech becomes raw JSON).
 */
export function transcriptFromOpenAiMultipartBody(body: string): string {
  const trimmed = String(body || '').trim()

  if (!trimmed) {
    return ''
  }

  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed) as { text?: unknown }

      if (typeof parsed?.text === 'string') {
        return parsed.text.trim()
      }
    } catch {
      // Not JSON — treat the body as the transcript.
    }
  }

  return trimmed
}

/**
 * Transcribe provider-direct. Returns the transcript ('' = silence), or null
 * when the profile's provider isn't client-callable — the caller relays.
 * Provider REJECTIONS throw: the configured provider said no, and silently
 * re-running the same request through the gateway would just fail again
 * slower and hide the real error.
 */
export async function transcribeAudioClientDirect(audio: Blob): Promise<null | string> {
  const config = await fetchVoiceClientConfig()
  const stt = config?.stt

  if (!stt || stt.mode !== 'direct') {
    return null
  }

  if (stt.wire === 'openai-multipart') {
    // A model that only accepts WAV rejects the recorder's WebM/Opus outright (HTTP 400), so
    // convert first — the required rate comes from the same server-resolved capability.
    const wantsWav = stt.audio_format === 'wav' && !audio.type.includes('wav')
    const payload = wantsWav ? await toWavMono(audio, STT_WAV_SAMPLE_RATE) : audio
    const form = new FormData()
    form.set('file', payload, (stt.audio_format === 'wav' ? 'recording.wav' : sttFileName(audio)))

    if (stt.model) {
      form.set('model', stt.model)
    }

    // Server-resolved: OpenRouter accepts only json/verbose_json, OpenAI and Groq return a bare
    // string for `text`. Never hardcode — a wrong value here is an HTTP 400 on every dictation.
    form.set('response_format', stt.response_format ?? 'text')

    if (stt.language) {
      form.set('language', stt.language)
    }

    const response = await fetch(`${stt.base_url.replace(/\/+$/, '')}/audio/transcriptions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${stt.api_key}` },
      body: form
    })

    if (!response.ok) {
      throw new Error(`${stt.provider} STT error (HTTP ${response.status}): ${await providerErrorText(response)}`)
    }

    return transcriptFromOpenAiMultipartBody(await response.text())
  }

  if (stt.wire === 'xai-stt') {
    const form = new FormData()
    form.set('file', audio, sttFileName(audio))
    form.set('format', 'true')

    if (stt.language) {
      form.set('language', stt.language)
    }

    const response = await fetch(`${stt.base_url.replace(/\/+$/, '')}/stt`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${stt.api_key}` },
      body: form
    })

    if (!response.ok) {
      throw new Error(`xAI STT error (HTTP ${response.status}): ${await providerErrorText(response)}`)
    }

    const result = (await response.json()) as { text?: string }

    return (result.text || '').trim()
  }

  if (stt.wire === 'elevenlabs-stt') {
    const form = new FormData()
    form.set('file', audio, sttFileName(audio))

    if (stt.model) {
      form.set('model_id', stt.model)
    }

    if (stt.language) {
      form.set('language_code', stt.language)
    }

    const response = await fetch(`${stt.base_url.replace(/\/+$/, '')}/speech-to-text`, {
      method: 'POST',
      headers: { 'xi-api-key': stt.api_key },
      body: form
    })

    if (!response.ok) {
      throw new Error(`ElevenLabs STT error (HTTP ${response.status}): ${await providerErrorText(response)}`)
    }

    const result = (await response.json()) as { text?: string }

    return (result.text || '').trim()
  }

  return null
}

// ---------------------------------------------------------------------------
// TTS — text → audio bytes, provider-direct. One call per sentence/segment;
// the playback queue in voice-playback.ts owns ordering and barge-in.
// ---------------------------------------------------------------------------

/** Resolve the profile's TTS config when it is client-callable, else null. */
export async function directTtsConfig(): Promise<DirectTtsConfig | null> {
  const config = await fetchVoiceClientConfig()

  return config?.tts && config.tts.mode === 'direct' ? config.tts : null
}

/** Synthesize one text segment to audio bytes (mp3). Throws on provider rejection. */
export async function synthesizeSpeechClientDirect(tts: DirectTtsConfig, text: string): Promise<ArrayBuffer> {
  if (tts.wire === 'openai-speech') {
    const body: Record<string, unknown> = {
      model: tts.model,
      voice: tts.voice,
      input: text,
      response_format: 'mp3'
    }

    if (tts.speed && tts.speed !== 1) {
      body.speed = tts.speed
    }

    const response = await fetch(`${tts.base_url.replace(/\/+$/, '')}/audio/speech`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${tts.api_key}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (!response.ok) {
      throw new Error(`${tts.provider} TTS error (HTTP ${response.status}): ${await providerErrorText(response)}`)
    }

    return response.arrayBuffer()
  }

  if (tts.wire === 'elevenlabs-tts') {
    const response = await fetch(
      `${tts.base_url.replace(/\/+$/, '')}/text-to-speech/${encodeURIComponent(tts.voice || '')}`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': tts.api_key,
          'Content-Type': 'application/json',
          Accept: 'audio/mpeg'
        },
        body: JSON.stringify({ text, model_id: tts.model })
      }
    )

    if (!response.ok) {
      throw new Error(`ElevenLabs TTS error (HTTP ${response.status}): ${await providerErrorText(response)}`)
    }

    return response.arrayBuffer()
  }

  throw new Error(`Unknown TTS wire: ${(tts as { wire?: string }).wire}`)
}

// ---------------------------------------------------------------------------
// Sentence cutter for the streaming TTS session — mirrors the server-side
// SentenceChunker's contract: emit complete sentences as they form, hold
// the incomplete tail, flush everything on finish.
// ---------------------------------------------------------------------------

const SENTENCE_BOUNDARY_RE = /[.!?…。！？]+["'”’)\]]*\s+/g
const MIN_SENTENCE_CHARS = 24

export function cutSentences(buffer: string, flush: boolean): { sentences: string[]; rest: string } {
  const sentences: string[] = []
  let rest = buffer
  let start = 0

  SENTENCE_BOUNDARY_RE.lastIndex = 0

  let match = SENTENCE_BOUNDARY_RE.exec(buffer)

  while (match) {
    const end = match.index + match[0].length
    const candidate = buffer.slice(start, end).trim()

    // Too-short fragments ("e.g. ", "1. ") stay buffered so we don't fire a
    // provider call per abbreviation — unless a later boundary extends them.
    if (candidate.length >= MIN_SENTENCE_CHARS) {
      sentences.push(candidate)
      start = end
    }

    match = SENTENCE_BOUNDARY_RE.exec(buffer)
  }

  rest = buffer.slice(start)

  if (flush) {
    const tail = rest.trim()

    if (tail) {
      sentences.push(tail)
    }

    rest = ''
  }

  return { sentences, rest }
}
