import { atom } from 'nanostores'

import type { ChatMessage } from '@/lib/chat-messages'
import { persistStringArray, storedStringArray } from '@/lib/storage'

export type MultitaskLayout = 'grid-2' | 'grid-3' | 'horizontal' | 'vertical'

export interface MultitaskTileState {
  storedSessionId: string
  runtimeSessionId: string | null
  title: string | null
  model: string | null
  messages: ChatMessage[]
  busy: boolean
  awaitingResponse: boolean
  cwd: string | null
  streamId: string | null
  error: string | null
}

const MULTITASK_STORAGE_KEY = 'hermes:multitask:sessionIds'
const MULTITASK_LAYOUT_KEY = 'hermes:multitask:layout'

// ─── Atoms ───────────────────────────────────────────────────────────

export const $multitaskSessionIds = atom<string[]>(storedStringArray(MULTITASK_STORAGE_KEY))

export const $multitaskTileStates = atom<Map<string, MultitaskTileState>>(new Map())

export const $multitaskLayout = atom<MultitaskLayout>(
  (typeof localStorage !== 'undefined'
    ? (localStorage.getItem(MULTITASK_LAYOUT_KEY) as MultitaskLayout | null)
    : null) ?? 'grid-2'
)

// ─── Actions ─────────────────────────────────────────────────────────

export function persistMultitaskSessionIds(ids: string[]): void {
  persistStringArray(MULTITASK_STORAGE_KEY, ids)
}

export function addMultitaskSession(storedSessionId: string): void {
  const ids = $multitaskSessionIds.get()
  if (ids.includes(storedSessionId)) return
  const next = [...ids, storedSessionId]
  $multitaskSessionIds.set(next)
  persistMultitaskSessionIds(next)

  const states = $multitaskTileStates.get()
  if (!states.has(storedSessionId)) {
    states.set(storedSessionId, {
      storedSessionId,
      runtimeSessionId: null,
      title: null,
      model: null,
      messages: [],
      busy: false,
      awaitingResponse: false,
      cwd: null,
      streamId: null,
      error: null
    })
    $multitaskTileStates.set(new Map(states))
  }
}

export function removeMultitaskSession(storedSessionId: string): void {
  const ids = $multitaskSessionIds.get().filter(id => id !== storedSessionId)
  $multitaskSessionIds.set(ids)
  persistMultitaskSessionIds(ids)

  const states = $multitaskTileStates.get()
  states.delete(storedSessionId)
  $multitaskTileStates.set(new Map(states))
}

export function clearAllMultitaskSessions(): void {
  $multitaskSessionIds.set([])
  persistMultitaskSessionIds([])
  $multitaskTileStates.set(new Map())
}

export function updateTileState(
  storedSessionId: string,
  updater: (state: MultitaskTileState) => MultitaskTileState
): void {
  const states = $multitaskTileStates.get()
  const current = states.get(storedSessionId)
  if (!current) return
  states.set(storedSessionId, updater({ ...current }))
  $multitaskTileStates.set(new Map(states))
}

export function setMultitaskLayout(layout: MultitaskLayout): void {
  $multitaskLayout.set(layout)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(MULTITASK_LAYOUT_KEY, layout)
  }
}
