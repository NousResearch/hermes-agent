import { useMemo } from 'react'

import { type ExtensionVoiceChatMode, isExtensionVoiceChatMode } from '@/lib/voice-live'

import { useContributions } from './react/use-contributions'
import { registry } from './registry'
import type { Contribution } from './types'

/**
 * Live voice engines — a full-duplex voice frontend an extension provides,
 * listed next to the built-in chained and GPT-Live engines.
 *
 *   data kind (`data`):  voice.liveEngines  (LiveVoiceEngine)
 *
 * The plugin owns the vendor transport, the audio and the short-lived session
 * secret (minted by its own backend; it never holds a Hermes OAuth token).
 * Core owns everything around it: the engine picker rows, Settings ▸ Voice,
 * the provider sign-in UI, one call at a time, teardown, and delegation — every
 * request the voice hands over becomes an ordinary Hermes turn on the open chat
 * (`prompt.submit`, surface `voice-live`), with tools and approvals intact.
 *
 * The namespaced contribution id (`<plugin>:<engine>`) is the value persisted
 * in `voice.voice_chat_mode`.
 */
export const VOICE_ENGINE_AREA = 'voice.liveEngines'

export interface LiveVoiceEngineStatus {
  available: boolean
  /** Plain-text reason shown under the engine row when it cannot start. */
  reason?: null | string
}

export interface LiveVoiceEngineHistoryMessage {
  role: 'assistant' | 'user'
  text: string
}

/** What core hands an engine when a call starts. */
export interface LiveVoiceEngineIO {
  /** The chat this call belongs to (null on a fresh draft). */
  sessionId: null | string
  /** Recent text turns of that chat, oldest first, to seed the voice model. */
  history: LiveVoiceEngineHistoryMessage[]
  /** Hand a request to Hermes. Core runs it as a normal turn on the chat
   *  (`prompt.submit`, surface `voice-live`, `voiceContext` rides the model
   *  input only) and resolves with the turn's final assistant text, prepared
   *  for speech (markdown stripped). Resolves empty when the turn produced no
   *  reply, a newer request superseded it, or the call ended; rejects only
   *  when the turn could not be submitted. */
  delegate: (request: string, voiceContext: string) => Promise<string>
  /** Caption text, for the spoken stop phrase and future live captions. */
  onTranscript?: (speaker: 'assistant' | 'user', text: string) => void
  /** The engine's audio output started or stopped. */
  onSpeakingChange?: (speaking: boolean) => void
  /** A vendor/transport error; `fatal` ends the call. */
  onError: (message: string, fatal: boolean) => void
  /** The engine's session ended on its own (not through `stop()`). Pass
   *  `'close_requested'` when the user hung up through the voice (a spoken
   *  goodbye); any other reason is reported to the user as a dropped call. */
  onClosed: (reason: string) => void
}

export interface LiveVoiceEngineSession {
  /** Release the microphone, close the transport. Idempotent. */
  stop: () => Promise<void>
  setMuted?: (muted: boolean) => void
}

/** Payload of a `voice.liveEngines` data contribution. */
export interface LiveVoiceEngine {
  /** Row label, e.g. "Grok Voice". */
  label: string
  /** One line under the label, e.g. "full-duplex, delegates to Hermes". */
  description?: string
  /** A Hermes OAuth provider id from `GET /api/providers/oauth` (for example
   *  `xai-oauth` or `openai-codex`). Core shows its sign-in state and runs the
   *  existing sign-in flow; the plugin's backend reaches the provider through
   *  `credentialed_provider_request`, so no token ever reaches the renderer. */
  signIn?: { provider: string }
  /** Keys of THIS plugin's manifest `config_schema` to render inline under
   *  Settings ▸ Voice. Values stay in `plugins.entries.<id>.settings`. */
  settingsKeys?: readonly string[]
  /** Whether the engine can start right now. Called when a picker opens. */
  status: () => Promise<LiveVoiceEngineStatus>
  /** Start a call. Resolves once the session is up; rejects to abort. */
  start: (io: LiveVoiceEngineIO) => Promise<LiveVoiceEngineSession>
}

export interface ResolvedLiveVoiceEngine {
  /** Namespaced id (`<plugin>:<engine>`), the `voice.voice_chat_mode` value. */
  id: ExtensionVoiceChatMode
  /** The plugin id (the part before the colon). */
  pluginId: string
  label: string
  description: null | string
  signIn: null | { provider: string }
  settingsKeys: readonly string[]
  engine: LiveVoiceEngine
}

const warned = new Set<string>()

function warnOnce(id: string, message: string) {
  if (!warned.has(id)) {
    warned.add(id)
    console.warn(`[${VOICE_ENGINE_AREA}] ${id}: ${message}`)
  }
}

const text = (value: unknown): null | string =>
  typeof value === 'string' && value.trim() ? value.replace(/\s+/g, ' ').trim() : null

function resolveOne(contribution: Contribution): null | ResolvedLiveVoiceEngine {
  if (!isExtensionVoiceChatMode(contribution.id)) {
    warnOnce(contribution.id, 'id must be <plugin>:<engine> in lowercase letters, digits, - and _; skipped')

    return null
  }

  const data = contribution.data as Partial<LiveVoiceEngine> | undefined
  const label = text(data?.label)

  if (!data || !label || typeof data.status !== 'function' || typeof data.start !== 'function') {
    warnOnce(contribution.id, 'data needs label, status() and start(); skipped')

    return null
  }

  const provider = text(data.signIn?.provider)

  return {
    description: text(data.description),
    engine: data as LiveVoiceEngine,
    id: contribution.id,
    label,
    pluginId: contribution.id.slice(0, contribution.id.indexOf(':')),
    settingsKeys: Array.isArray(data.settingsKeys)
      ? data.settingsKeys.filter((key): key is string => typeof key === 'string' && key.length > 0)
      : [],
    signIn: provider ? { provider } : null
  }
}

// Code-unit order, not localeCompare: the picker must read the same in every locale.
const byId = (a: ResolvedLiveVoiceEngine, b: ResolvedLiveVoiceEngine) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)

/** The valid engines among an area's contributions, ordered by id. */
export function resolveLiveVoiceEngines(contributions: readonly Contribution[]): ResolvedLiveVoiceEngine[] {
  return contributions
    .map(resolveOne)
    .filter((engine): engine is ResolvedLiveVoiceEngine => engine !== null)
    .sort(byId)
}

/** Every registered engine, live (re-renders when a plugin loads or unloads). */
export function useLiveVoiceEngines(): ResolvedLiveVoiceEngine[] {
  const contributions = useContributions(VOICE_ENGINE_AREA)

  return useMemo(() => resolveLiveVoiceEngines(contributions), [contributions])
}

/** The registered engine for a `voice.voice_chat_mode` value, or null (a
 *  built-in mode, or a plugin that is disabled or uninstalled). */
export function findLiveVoiceEngine(mode: unknown): null | ResolvedLiveVoiceEngine {
  if (!isExtensionVoiceChatMode(mode)) {
    return null
  }

  return resolveLiveVoiceEngines(registry.getArea(VOICE_ENGINE_AREA)).find(engine => engine.id === mode) ?? null
}

/** An engine's `status()`, never throwing: a failing or hanging engine reads as
 *  unavailable with its error text. */
export async function liveVoiceEngineStatus(
  engine: ResolvedLiveVoiceEngine,
  timeoutMs = 5_000
): Promise<{ available: boolean; reason: null | string }> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    const answer = await Promise.race([
      engine.engine.status(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('status check timed out')), timeoutMs)
      })
    ])

    return { available: Boolean(answer?.available), reason: text(answer?.reason) }
  } catch (error) {
    return { available: false, reason: text(error instanceof Error ? error.message : String(error)) }
  } finally {
    clearTimeout(timer)
  }
}
