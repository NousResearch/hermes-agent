import { atom } from 'nanostores'

import type { ContextFull } from '@/types/hermes'

export type ContextInspectorBucket = 'all' | 'assistant' | 'system' | 'tools' | 'user'
export type ContextInspectorTab = 'raw' | 'transcript'
export type ContextInspectorRequest = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export type ContextInspectorSource =
  | { status: 'idle' }
  | { runtimeSessionId?: string; sessionId?: string; status: 'loading' }
  | { runtimeSessionId: string; sessionId: string; status: 'ready' }
  | { runtimeSessionId?: string; sessionId?: string; status: 'empty' }
  | { error: string; runtimeSessionId?: string; sessionId?: string; status: 'error' }

export interface ContextInspectorOptions {
  runtimeIdByStoredSessionId?: ReadonlyMap<string, string>
}

export const $contextInspectorOpen = atom(false)
export const $contextSource = atom<ContextInspectorSource>({ status: 'idle' })
export const $contextData = atom<ContextFull | null>(null)
export const $activeBucket = atom<ContextInspectorBucket>('all')
export const $activeTab = atom<ContextInspectorTab>('transcript')

// Monotonic open sequence. Each openContextInspector() call claims the next
// number; a resolving fetch only commits its result if it is still the latest
// open. Prevents a slower earlier request (session A) from overwriting a newer
// open (session B) when the user re-opens the inspector mid-flight.
let contextRequestSeq = 0

function emptyContextFull(): ContextFull {
  return {
    available: false,
    context_max: 0,
    context_used: 0,
    exact_capture_available: false,
    messages: [],
    model: '',
    raw_unredacted: true,
    slices: [],
    source: 'reconstructed_base',
    source_label: 'Context available after the agent initializes',
    state: 'agent_not_built'
  }
}

function messageForError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function resolveContextRuntimeSessionId(
  sessionId: null | string,
  runtimeIdByStoredSessionId?: ReadonlyMap<string, string>
): null | string {
  if (!sessionId) {
    return null
  }

  if (!runtimeIdByStoredSessionId) {
    return sessionId
  }

  const mappedRuntimeId = runtimeIdByStoredSessionId.get(sessionId)

  if (mappedRuntimeId) {
    return mappedRuntimeId
  }

  for (const runtimeId of runtimeIdByStoredSessionId.values()) {
    if (runtimeId === sessionId) {
      return sessionId
    }
  }

  return null
}

export async function openContextInspector(
  sessionId: null | string,
  requestGateway: ContextInspectorRequest,
  options: ContextInspectorOptions = {}
): Promise<void> {
  const seq = ++contextRequestSeq
  $contextInspectorOpen.set(true)
  $activeBucket.set('all')
  $activeTab.set('transcript')

  const runtimeSessionId = resolveContextRuntimeSessionId(sessionId, options.runtimeIdByStoredSessionId)

  if (!runtimeSessionId) {
    $contextData.set(emptyContextFull())
    $contextSource.set({ sessionId: sessionId ?? undefined, status: 'empty' })

    return
  }

  $contextData.set(null)
  $contextSource.set({ runtimeSessionId, sessionId: sessionId ?? runtimeSessionId, status: 'loading' })

  try {
    const data = await requestGateway<ContextFull>('session.context_full', { session_id: runtimeSessionId })

    // A newer open superseded this request while it was in flight — drop it.
    if (seq !== contextRequestSeq) {
      return
    }

    $contextData.set(data)
    $contextSource.set({
      runtimeSessionId,
      sessionId: sessionId ?? runtimeSessionId,
      status: data.available && data.state === 'ready' ? 'ready' : 'empty'
    })
  } catch (error) {
    if (seq !== contextRequestSeq) {
      return
    }

    $contextData.set(null)
    $contextSource.set({
      error: messageForError(error),
      runtimeSessionId,
      sessionId: sessionId ?? runtimeSessionId,
      status: 'error'
    })
  }
}

export function closeContextInspector(): void {
  $contextInspectorOpen.set(false)
}
