import { deleteEnvVar, revealEnvVar, setEnvVar } from '@/hermes'

export const GEMINI_LIVE_VOICE_STORAGE = 'hermes_gemini_live_voice'
export const GEMINI_LIVE_MODEL_STORAGE = 'hermes_gemini_live_model'

export const DEFAULT_GEMINI_LIVE_VOICE = 'Puck'
export const DEFAULT_GEMINI_LIVE_MODEL = 'gemini-3.8-live'

/**
 * Live models differ in how they treat `thinkingConfig`, so it must be sent per
 * model rather than unconditionally (see `thinkingConfigForModel`).
 */
export const THINKING_LEVEL_MODELS = ['gemini-3.8-live-extended-thinking'] as const
export const DEFAULT_THINKING_LEVEL = 'LOW'

export type GeminiLiveVoice = {
  id: string
  label: string
  /** Set only where the character is published; acceptance is not character. */
  tone?: string
}

export type GeminiLiveModel = { id: string; label: string }

// Every voice below was accepted by the Live API and returned audio. The first
// five carry the tone descriptions from Google's published voice list; the rest
// are verified to connect but deliberately left without a tone claim.
export const GEMINI_VOICES: readonly GeminiLiveVoice[] = [
  { id: 'Puck', label: 'Puck', tone: 'Playful, energetic, high dynamic range' },
  { id: 'Charon', label: 'Charon', tone: 'Deep, calm, authoritative' },
  { id: 'Aoede', label: 'Aoede', tone: 'Warm, natural, expressive' },
  { id: 'Kore', label: 'Kore', tone: 'Clear, balanced, pleasant' },
  { id: 'Fenrir', label: 'Fenrir', tone: 'Bold, direct, resonant' },
  { id: 'Zephyr', label: 'Zephyr' },
  { id: 'Leda', label: 'Leda' },
  { id: 'Orus', label: 'Orus' },
  { id: 'Callirrhoe', label: 'Callirrhoe' },
  { id: 'Autonoe', label: 'Autonoe' },
  { id: 'Enceladus', label: 'Enceladus' },
  { id: 'Iapetus', label: 'Iapetus' },
  { id: 'Umbriel', label: 'Umbriel' },
  { id: 'Algieba', label: 'Algieba' },
  { id: 'Despina', label: 'Despina' },
  { id: 'Erinome', label: 'Erinome' },
  { id: 'Algenib', label: 'Algenib' },
  { id: 'Rasalgethi', label: 'Rasalgethi' },
  { id: 'Laomedeia', label: 'Laomedeia' },
  { id: 'Achernar', label: 'Achernar' },
  { id: 'Alnilam', label: 'Alnilam' },
  { id: 'Schedar', label: 'Schedar' },
  { id: 'Gacrux', label: 'Gacrux' },
  { id: 'Pulcherrima', label: 'Pulcherrima' },
  { id: 'Achird', label: 'Achird' },
  { id: 'Zubenelgenubi', label: 'Zubenelgenubi' },
  { id: 'Vindemiatrix', label: 'Vindemiatrix' },
  { id: 'Sadachbia', label: 'Sadachbia' },
  { id: 'Sadaltager', label: 'Sadaltager' },
  { id: 'Sulafat', label: 'Sulafat' }
]

export const GEMINI_LIVE_MODELS: readonly GeminiLiveModel[] = [
  { id: 'gemini-3.1-flash-live-preview', label: 'Gemini 3.1 Flash Live (Fastest to first audio)' },
  { id: 'gemini-3.8-live', label: 'Gemini 3.8 Live (Recommended)' },
  { id: 'gemini-3.8-live-extended-thinking', label: 'Gemini 3.8 Live Extended Thinking' },
  { id: 'gemini-2.5-flash-native-audio-latest', label: 'Gemini 2.5 Flash Native Audio' }
]

export function getStoredGeminiLiveVoice(): string {
  try {
    return localStorage.getItem(GEMINI_LIVE_VOICE_STORAGE)?.trim() || DEFAULT_GEMINI_LIVE_VOICE
  } catch {
    return DEFAULT_GEMINI_LIVE_VOICE
  }
}

export function setStoredGeminiLiveVoice(voice: string): void {
  try {
    localStorage.setItem(GEMINI_LIVE_VOICE_STORAGE, voice)
  } catch {}
}

// Retired ids that no longer connect: a stored value pointing at one of these
// would leave the dialog showing a model that fails at setup.
export const RETIRED_GEMINI_LIVE_MODELS: readonly string[] = [
  'gemini-2.0-flash-exp',
  'gemini-2.5-flash'
]

export function getStoredGeminiLiveModel(): string {
  try {
    const stored = localStorage.getItem(GEMINI_LIVE_MODEL_STORAGE)?.trim()
    if (stored && !RETIRED_GEMINI_LIVE_MODELS.includes(stored)) {
      return stored
    }
    return DEFAULT_GEMINI_LIVE_MODEL
  } catch {
    return DEFAULT_GEMINI_LIVE_MODEL
  }
}

export function setStoredGeminiLiveModel(model: string): void {
  try {
    localStorage.setItem(GEMINI_LIVE_MODEL_STORAGE, model)
  } catch {}
}

/**
 * `thinkingConfig` cannot be sent unconditionally — the models disagree:
 *
 *   - `gemini-3.8-live-extended-thinking` rejects setup without it:
 *     "Thinking level must be specified for this model" (code 1007).
 *   - `gemini-3.8-live` rejects setup when it IS present:
 *     "Thinking level is not supported for this model" (code 1007).
 *
 * `MINIMAL` is rejected by the extended-thinking model; `LOW` and `HIGH` connect.
 */
export function thinkingConfigForModel(model: string): { thinkingLevel: string } | null {
  const bare = model.startsWith('models/') ? model.slice('models/'.length) : model
  return (THINKING_LEVEL_MODELS as readonly string[]).includes(bare)
    ? { thinkingLevel: DEFAULT_THINKING_LEVEL }
    : null
}

/**
 * Resolves the Google / Gemini API key from the gateway environment credentials (.env).
 * Matches the canonical GEMINI_API_KEY (and GOOGLE_API_KEY) configured in Settings -> Keys.
 * Automatically purges any legacy plaintext localStorage keys.
 */
export async function resolveGeminiLiveApiKey(): Promise<string | null> {
  // Purge any legacy unencrypted plaintext localStorage key
  try {
    localStorage.removeItem('hermes_gemini_live_api_key')
  } catch {}

  // 1. Probe GEMINI_API_KEY in gateway environment (.env)
  try {
    const res = await revealEnvVar('GEMINI_API_KEY')
    if (res?.value?.trim()) {
      return res.value.trim()
    }
  } catch {}

  // 2. Probe GOOGLE_API_KEY in gateway environment (.env)
  try {
    const res = await revealEnvVar('GOOGLE_API_KEY')
    if (res?.value?.trim()) {
      return res.value.trim()
    }
  } catch {}

  return null
}

/**
 * Persists or updates the Gemini API key into the gateway environment credentials (.env).
 * Matches how Settings -> Keys and Settings -> Providers store API keys securely.
 */
export async function saveGeminiApiKey(key: string): Promise<{ ok: boolean }> {
  // Purge any legacy plaintext localStorage key
  try {
    localStorage.removeItem('hermes_gemini_live_api_key')
  } catch {}

  const trimmed = key.trim()
  if (trimmed) {
    return setEnvVar('GEMINI_API_KEY', trimmed)
  } else {
    return deleteEnvVar('GEMINI_API_KEY')
  }
}

export interface LiveTranscriptFragment {
  speaker: 'assistant' | 'user'
  text: string
  interim?: boolean
  startMs?: number
  endMs?: number
}

export interface GeminiLiveHandlers {
  onDelegation: (delegationId: string, request: string) => void
  onError: (message: string, fatal: boolean) => void
  onClosed: (reason: string) => void
  onTranscript?: (fragment: LiveTranscriptFragment) => void
  onSpeakingChange?: (speaking: boolean) => void
  onLevel?: (level: number) => void
  onTurnComplete?: () => void
  onInterrupted?: () => void
}

/**
 * GeminiLiveSession runs 100% inside Hermes Desktop on the client's laptop.
 * Connects directly to Google's WebSocket, handles audio capture, downsamples
 * to 16kHz PCM, receives 24kHz PCM playback, and delegates real tasks to Hermes.
 */
export class GeminiLiveSession {
  private ws: WebSocket | null = null
  private micStream: MediaStream | null = null
  private inputAudioCtx: AudioContext | null = null
  private micSource: MediaStreamAudioSourceNode | null = null
  private muteGain: GainNode | null = null
  private processorNode: ScriptProcessorNode | null = null
  private playbackAudioCtx: AudioContext | null = null
  private activeSources: AudioBufferSourceNode[] = []
  private nextPlaybackTime = 0
  private started = false
  private finalized = false
  private isSpeaking = false
  private sessionStartTime = 0
  private muted = false

  constructor(
    private readonly handlers: GeminiLiveHandlers,
    private readonly options: {
      apiKey: string
      model?: string
      voice?: string
      systemInstruction?: string
    }
  ) {}

  get connected(): boolean {
    return this.started && this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  public setMuted(muted: boolean): void {
    this.muted = muted
    if (muted) {
      this.handlers.onLevel?.(0)
    }
  }

  public isMuted(): boolean {
    return this.muted
  }

  async start(): Promise<void> {
    if (this.started) {
      throw new Error('Gemini Live session already started')
    }

    this.sessionStartTime = Date.now()
    const apiKey = this.options.apiKey.trim()
    if (!apiKey) {
      throw new Error('Google Gemini API Key is required for Gemini Live mode.')
    }

    const modelName = this.options.model || getStoredGeminiLiveModel()
    const voiceName = this.options.voice || getStoredGeminiLiveVoice()

    const wsUrl = new URL(
      'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
    )
    wsUrl.searchParams.set('key', apiKey)

    const ws = new WebSocket(wsUrl.toString())
    this.ws = ws

    // Await setupComplete from Gemini Live before opening media capture
    await new Promise<void>((resolve, reject) => {
      const openTimeout = window.setTimeout(() => {
        reject(new Error('Connection to Gemini Live WebSocket timed out.'))
      }, 15_000)

      let resolved = false

      const setupListener = async (event: MessageEvent) => {
        try {
          let text = ''
          if (typeof event.data === 'string') {
            text = event.data
          } else if (event.data instanceof Blob) {
            text = await event.data.text()
          } else if (event.data instanceof ArrayBuffer) {
            text = new TextDecoder().decode(event.data)
          }

          if (text) {
            const parsed = JSON.parse(text)
            if (parsed.setupComplete) {
              resolved = true
              window.clearTimeout(openTimeout)
              ws.removeEventListener('message', setupListener)
              resolve()
            } else if (parsed.error) {
              resolved = true
              window.clearTimeout(openTimeout)
              ws.removeEventListener('message', setupListener)
              const msg = parsed.error.message || JSON.stringify(parsed.error)
              reject(new Error(`Gemini Live setup failed: ${msg}`))
            }
          }
        } catch {}
      }

      ws.addEventListener('message', setupListener)

      ws.onopen = () => {
        // Sent per model, never unconditionally: see thinkingConfigForModel().
        const thinkingConfig = thinkingConfigForModel(modelName)

        const setupMessage = {
          setup: {
            model: modelName.startsWith('models/') ? modelName : `models/${modelName}`,
            generationConfig: {
              responseModalities: ['AUDIO'],
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: {
                    voiceName
                  }
                }
              },
              ...(thinkingConfig ? { thinkingConfig } : {})
            },
            inputAudioTranscription: {},
            outputAudioTranscription: {},
            systemInstruction: {
              parts: [
                {
                  text:
                    this.options.systemInstruction ||
                    'You are the Gemini Live voice interface for Hermes Agent on desktop. ' +
                      'You speak conversationally, concisely, and naturally. ' +
                      'When the user asks you to perform an agent task, write code, run commands, inspect files, or search the web, ' +
                      "call the tool 'ask_hermes' with the user's request. When Hermes returns the result, summarize it clearly aloud."
                }
              ]
            },
            tools: [
              {
                functionDeclarations: [
                  {
                    name: 'ask_hermes',
                    description:
                      'Delegate a question, task, code execution, terminal command, file search, or agent operation to Hermes Agent.',
                    parameters: {
                      type: 'OBJECT',
                      properties: {
                        request: {
                          type: 'STRING',
                          description: "The exact user request or instruction to delegate to Hermes Agent."
                        }
                      },
                      required: ['request']
                    }
                  }
                ]
              }
            ]
          }
        }

        try {
          ws.send(JSON.stringify(setupMessage))
        } catch (err: any) {
          window.clearTimeout(openTimeout)
          ws.removeEventListener('message', setupListener)
          reject(err)
        }
      }

      ws.onerror = () => {
        if (!resolved) {
          window.clearTimeout(openTimeout)
          ws.removeEventListener('message', setupListener)
          reject(new Error('Failed to connect to Google Gemini Live WebSocket.'))
        }
      }

      ws.onclose = ev => {
        if (!resolved) {
          window.clearTimeout(openTimeout)
          ws.removeEventListener('message', setupListener)
          reject(new Error(ev.reason || `WebSocket closed before setup completed (code: ${ev.code})`))
        }
      }
    })

    ws.onmessage = async (event: MessageEvent) => {
      try {
        let text = ''
        if (typeof event.data === 'string') {
          text = event.data
        } else if (event.data instanceof Blob) {
          text = await event.data.text()
        } else if (event.data instanceof ArrayBuffer) {
          text = new TextDecoder().decode(event.data)
        }

        if (text) {
          const parsed = JSON.parse(text)
          if (parsed.error) {
            const errMsg = parsed.error.message || JSON.stringify(parsed.error)
            console.error('[Gemini Live] Server error:', parsed.error)
            this.handlers.onError(errMsg, true)
            this.finish(`Gemini Live Error: ${errMsg}`)
            return
          }
          this.handleServerMessage(parsed)
        }
      } catch (err: any) {
        this.handlers.onError(err?.message || 'Error parsing Gemini Live message', false)
      }
    }

    ws.onclose = ev => {
      if (!this.finalized) {
        this.finish(ev.reason || `WebSocket closed (code: ${ev.code})`)
      }
    }

    if (this.finalized) {
      return
    }

    // Initialize Local Playback AudioContext (24kHz) and resume if suspended
    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext
    if (AudioContextCtor) {
      const playbackCtx = new AudioContextCtor({ sampleRate: 24000 })
      if (playbackCtx.state === 'suspended') {
        await playbackCtx.resume()
      }
      if (this.finalized) {
        void playbackCtx.close().catch(() => undefined)
        return
      }
      this.playbackAudioCtx = playbackCtx
    }

    if (this.finalized) {
      return
    }

    // Initialize Local Microphone Capture
    await this.startMicrophoneCapture()
    if (this.finalized) {
      return
    }
    this.started = true
  }

  private async startMicrophoneCapture(): Promise<void> {
    if (this.finalized) {
      return
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    })

    if (this.finalized) {
      for (const track of stream.getTracks()) {
        try {
          track.stop()
        } catch {}
      }
      return
    }
    this.micStream = stream

    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext
    let inputCtx: AudioContext
    try {
      // Attempt native 16kHz context so Chromium's native resampler handles downsampling
      inputCtx = new AudioContextCtor({ sampleRate: 16000 })
    } catch {
      inputCtx = new AudioContextCtor()
    }

    if (this.finalized) {
      void inputCtx.close().catch(() => undefined)
      for (const track of stream.getTracks()) {
        try {
          track.stop()
        } catch {}
      }
      this.micStream = null
      return
    }

    if (inputCtx.state === 'suspended') {
      await inputCtx.resume()
      if (this.finalized) {
        void inputCtx.close().catch(() => undefined)
        for (const track of stream.getTracks()) {
          try {
            track.stop()
          } catch {}
        }
        this.micStream = null
        return
      }
    }
    this.inputAudioCtx = inputCtx

    const source = inputCtx.createMediaStreamSource(stream)
    this.micSource = source

    // 2048 samples per buffer provides ~43ms latency at 48kHz, or 128ms at 16kHz
    const processor = inputCtx.createScriptProcessor(2048, 1, 1)
    this.processorNode = processor

    // Mute gain node to prevent speaker feedback while keeping ScriptProcessor active
    const muteGain = inputCtx.createGain()
    muteGain.gain.value = 0
    this.muteGain = muteGain

    source.connect(processor)
    processor.connect(muteGain)
    muteGain.connect(inputCtx.destination)

    const inputSampleRate = inputCtx.sampleRate
    const targetSampleRate = 16000
    const ratio = inputSampleRate / targetSampleRate

    processor.onaudioprocess = e => {
      if (!this.connected || this.muted) {
        return
      }

      const inputBuffer = e.inputBuffer.getChannelData(0)
      if (!inputBuffer || inputBuffer.length === 0) {
        return
      }

      // Compute volume level for RMS meter
      let sum = 0
      for (let i = 0; i < inputBuffer.length; i++) {
        sum += inputBuffer[i] * inputBuffer[i]
      }
      const rms = Math.sqrt(sum / inputBuffer.length)
      const normalizedLevel = Math.min(1, rms / 0.3)
      this.handlers.onLevel?.(normalizedLevel)

      // Downsample to 16kHz with anti-aliasing box filter if needed
      const outputLength = Math.floor(inputBuffer.length / ratio)
      const pcm16 = new Int16Array(outputLength)

      if (Math.abs(ratio - 1) < 0.01) {
        for (let i = 0; i < outputLength; i++) {
          const s = Math.max(-1, Math.min(1, inputBuffer[i] || 0))
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
        }
      } else {
        for (let i = 0; i < outputLength; i++) {
          const start = Math.floor(i * ratio)
          const end = Math.min(inputBuffer.length, Math.floor((i + 1) * ratio))
          let wSum = 0
          let wCount = 0
          for (let j = start; j < end; j++) {
            wSum += inputBuffer[j]
            wCount++
          }
          const s = wCount > 0 ? wSum / wCount : (inputBuffer[start] || 0)
          const clamped = Math.max(-1, Math.min(1, s))
          pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
        }
      }

      // Encode Int16Array to base64
      const bytes = new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength)
      let binary = ''
      const len = bytes.byteLength
      const CHUNK_SZ = 0x8000
      for (let i = 0; i < len; i += CHUNK_SZ) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + CHUNK_SZ, len)) as any)
      }
      const base64Audio = btoa(binary)

      // Send to Gemini Live WebSocket using the official `audio` field
      const audioFrame = {
        realtimeInput: {
          audio: {
            mimeType: 'audio/pcm;rate=16000',
            data: base64Audio
          }
        }
      }

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(audioFrame))
      }
    }
  }

  private handleServerMessage(msg: Record<string, any>): void {
    const serverContent = msg.serverContent || msg.server_content
    // 1. Check for Server Content (Audio / Text / Interruption)
    if (serverContent) {
      if (serverContent.interrupted) {
        // Instantaneous Barge-In!
        this.interrupt()
        this.handlers.onInterrupted?.()
        return
      }

      // 1a. User speech transcription from Google
      const userFinal = serverContent.inputTranscription?.text || serverContent.input_transcription?.text
      const userInterim =
        serverContent.interimInputTranscription?.text || serverContent.interim_input_transcription?.text
      if (userFinal || userInterim) {
        const now = Date.now() - this.sessionStartTime
        this.handlers.onTranscript?.({
          speaker: 'user',
          text: userFinal || userInterim,
          interim: !userFinal && Boolean(userInterim),
          startMs: now,
          endMs: now + 500
        })
      }

      // 1b. Model speech transcription and audio playback
      let assistantText =
        serverContent.outputTranscription?.text || serverContent.output_transcription?.text || ''

      const modelTurn = serverContent.modelTurn || serverContent.model_turn
      if (modelTurn?.parts) {
        for (const part of modelTurn.parts) {
          if (part.text && !assistantText) {
            assistantText = part.text
          }

          const inlineData = part.inlineData || part.inline_data
          const mime = inlineData?.mimeType || inlineData?.mime_type
          if (inlineData?.data && mime?.startsWith('audio/pcm')) {
            this.queueAudioPlayback(inlineData.data)
          }
        }
      }

      if (assistantText) {
        const now = Date.now() - this.sessionStartTime
        this.handlers.onTranscript?.({
          speaker: 'assistant',
          text: assistantText,
          interim: false,
          startMs: now,
          endMs: now + 500
        })
      }

      const turnComplete = serverContent.turnComplete ?? serverContent.turn_complete
      if (turnComplete) {
        this.handlers.onTurnComplete?.()
      }
    }

    // 2. Check for Tool Calls (Delegation to Hermes)
    const toolCall = msg.toolCall || msg.tool_call
    const functionCalls = toolCall?.functionCalls || toolCall?.function_calls
    if (functionCalls) {
      for (const call of functionCalls) {
        if (call.name === 'ask_hermes') {
          const requestId = call.id || `call_${Date.now()}`
          const userRequest = call.args?.request || ''
          this.handlers.onDelegation(requestId, userRequest)
        }
      }
    }
  }

  private queueAudioPlayback(base64Data: string): void {
    if (!this.playbackAudioCtx) {
      return
    }

    try {
      const binary = atob(base64Data)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i)
      }
      const pcm16 = new Int16Array(bytes.buffer)

      // Convert 16-bit PCM to Float32
      const float32 = new Float32Array(pcm16.length)
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768.0
      }

      const audioBuffer = this.playbackAudioCtx.createBuffer(1, float32.length, 24000)
      audioBuffer.copyToChannel(float32, 0)

      const source = this.playbackAudioCtx.createBufferSource()
      source.buffer = audioBuffer
      source.connect(this.playbackAudioCtx.destination)

      const now = this.playbackAudioCtx.currentTime
      const startTime = Math.max(now, this.nextPlaybackTime)
      source.start(startTime)
      this.nextPlaybackTime = startTime + audioBuffer.duration

      this.activeSources.push(source)
      this.setSpeaking(true)

      source.onended = () => {
        const index = this.activeSources.indexOf(source)
        if (index !== -1) {
          this.activeSources.splice(index, 1)
        }
        if (this.activeSources.length === 0) {
          this.setSpeaking(false)
        }
      }
    } catch (err: any) {
      this.handlers.onError(`Audio playback error: ${err?.message || err}`, false)
    }
  }

  private setSpeaking(speaking: boolean): void {
    if (this.isSpeaking !== speaking) {
      this.isSpeaking = speaking
      this.handlers.onSpeakingChange?.(speaking)
    }
  }

  /**
   * Barge-in cancellation: immediately halts all playing audio sources and resets queue.
   */
  interrupt(): void {
    for (const source of this.activeSources) {
      try {
        source.stop()
        source.disconnect()
      } catch {}
    }
    this.activeSources = []
    if (this.playbackAudioCtx) {
      this.nextPlaybackTime = this.playbackAudioCtx.currentTime
    }
    this.setSpeaking(false)
  }

  /**
   * Sends the result of an 'ask_hermes' tool delegation back to Gemini Live.
   */
  sendToolResponse(callId: string, result: string, name = 'ask_hermes'): boolean {
    if (!this.connected || !this.ws) {
      return false
    }

    const payload = {
      toolResponse: {
        functionResponses: [
          {
            id: callId,
            name,
            response: {
              output: {
                result
              }
            }
          }
        ]
      }
    }

    this.ws.send(JSON.stringify(payload))
    return true
  }

  /**
   * Sends commentary or context updates from Hermes back to Gemini Live.
   */
  sendCommentary(text: string): boolean {
    if (!this.connected || !this.ws) {
      return false
    }

    const payload = {
      clientContent: {
        turns: [
          {
            role: 'user',
            parts: [{ text: `[Hermes update to explain to user]: ${text}` }]
          }
        ],
        turnComplete: true
      }
    }

    this.ws.send(JSON.stringify(payload))
    return true
  }

  /**
   * Sends a quiet update of what tool Hermes is running right now.
   */
  sendToolActivity(label: string): boolean {
    if (!this.connected || !this.ws || !label) {
      return false
    }

    const payload = {
      clientContent: {
        turns: [
          {
            role: 'user',
            parts: [{ text: `[Hermes status: ${label}]` }]
          }
        ],
        turnComplete: true
      }
    }

    try {
      this.ws.send(JSON.stringify(payload))
      return true
    } catch {
      return false
    }
  }

  /**
   * Signals the end of the user audio stream without sending synthetic prompt text.
   * Tells Google's server to flush any buffered audio and complete the turn.
   */
  stopAudioStream(): boolean {
    if (!this.connected || !this.ws) {
      return false
    }

    try {
      this.ws.send(
        JSON.stringify({
          realtimeInput: {
            audioStreamEnd: true
          }
        })
      )
      return true
    } catch {
      return false
    }
  }

  speak(delegationId: string, text: string): void {
    if (!text || !text.trim()) {
      return
    }
    if (delegationId) {
      this.sendToolResponse(delegationId, text)
    } else {
      this.sendCommentary(text)
    }
  }

  think(delegationId: string, note: string): void {
    this.sendToolActivity(note)
  }

  close(): void {
    this.finish('close_requested')
  }

  finish(reason = 'closed'): void {
    if (this.finalized) {
      return
    }

    this.finalized = true
    this.interrupt()

    // Stop mic stream
    if (this.micStream) {
      for (const track of this.micStream.getTracks()) {
        track.stop()
      }
      this.micStream = null
    }

    if (this.micSource) {
      this.micSource.disconnect()
      this.micSource = null
    }

    if (this.muteGain) {
      this.muteGain.disconnect()
      this.muteGain = null
    }

    if (this.processorNode) {
      this.processorNode.disconnect()
      this.processorNode = null
    }

    if (this.inputAudioCtx) {
      void this.inputAudioCtx.close().catch(() => undefined)
      this.inputAudioCtx = null
    }

    if (this.playbackAudioCtx) {
      void this.playbackAudioCtx.close().catch(() => undefined)
      this.playbackAudioCtx = null
    }

    if (this.ws) {
      try {
        this.ws.close()
      } catch {}
      this.ws = null
    }

    this.started = false
    this.handlers.onClosed(reason)
  }
}
