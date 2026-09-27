import { hermesApi, profileScoped } from './client'

/** The two Live voice providers: OpenAI Realtime (WebRTC) and Gemini Live (WebSocket). */
export type LiveVoiceProviderId = 'gemini' | 'openai'

/** `GET /api/voice/realtime/status` — whether Live voice can start (the provider's key is set). */
export interface RealtimeVoiceStatusResponse {
  available: boolean
  language: string
  model: string
  provider: LiveVoiceProviderId
  voice: string
}

/**
 * `POST /api/voice/realtime/session`, OpenAI: a short-lived client secret for
 * a session the backend has already configured. The OpenAI key itself never
 * reaches the renderer.
 */
export interface OpenAiRealtimeVoiceSessionResponse {
  provider: 'openai'
  calls_url: string
  client_secret: string
  expires_at: null | number
  model: string
  voice: string
}

/**
 * `POST /api/voice/realtime/session`, Gemini: a one-use ephemeral token with
 * `setup` locked into it, and the WebSocket to open with it. The Google key
 * never reaches the renderer.
 */
export interface GeminiLiveVoiceSessionResponse {
  provider: 'gemini'
  language: string
  model: string
  setup: Record<string, unknown>
  token: string
  voice: string
  ws_url: string
}

export type RealtimeVoiceSessionResponse = GeminiLiveVoiceSessionResponse | OpenAiRealtimeVoiceSessionResponse

/** Pass `provider` to ask about one that is not selected yet (onboarding, settings). */
export function getRealtimeVoiceStatus(provider?: LiveVoiceProviderId): Promise<RealtimeVoiceStatusResponse> {
  return hermesApi<RealtimeVoiceStatusResponse>({
    ...profileScoped(),
    path: provider ? `/api/voice/realtime/status?provider=${provider}` : '/api/voice/realtime/status'
  })
}

export async function createRealtimeVoiceSession(): Promise<RealtimeVoiceSessionResponse> {
  try {
    return await hermesApi<RealtimeVoiceSessionResponse>({
      ...profileScoped(),
      method: 'POST',
      path: '/api/voice/realtime/session'
    })
  } catch (error) {
    if (error instanceof Error && /\b(404|405)\b/.test(error.message)) {
      throw new Error(
        'Wybrany Hermes nie obsługuje rozmowy Live. Uruchom najnowszą wersję Agenta Cześka z silnikiem dołączonym do instalatora. Twoje klucze i rozmowy pozostają w profilu.'
      )
    }

    throw error
  }
}
