import { atom, computed, type ReadableAtom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

export type ToolViewMode = 'product' | 'technical'

type ToolDisclosureStates = Record<string, boolean>

const TOOL_VIEW_TECHNICAL_STORAGE_KEY = 'hermes.desktop.toolView.technical'
const TOOL_DISCLOSURE_STORAGE_KEY = 'hermes.desktop.toolDisclosure.v1'
const KEEP_TOOL_CALLS_EXPANDED_KEY = 'hermes.desktop.keepToolCallsExpanded'
const MAX_DISCLOSURE_STATES = 240

export const $toolViewMode = atom<ToolViewMode>(
  storedBoolean(TOOL_VIEW_TECHNICAL_STORAGE_KEY, false) ? 'technical' : 'product'
)
export const $toolDisclosureStates = atom<ToolDisclosureStates>(loadToolDisclosureStates())

// When true, all tool-call accordions start expanded and stay expanded.
// The user can still collapse individual ones; re-render restores expanded.
export const $keepToolCallsExpanded = atom<boolean>(
  storedBoolean(KEEP_TOOL_CALLS_EXPANDED_KEY, false)
)
const disclosureOpenCache = new Map<string, ReadableAtom<boolean | undefined>>()

$toolViewMode.subscribe(mode => persistBoolean(TOOL_VIEW_TECHNICAL_STORAGE_KEY, mode === 'technical'))
$toolDisclosureStates.subscribe(persistToolDisclosureStates)
$keepToolCallsExpanded.subscribe(v => persistBoolean(KEEP_TOOL_CALLS_EXPANDED_KEY, v))

export function setToolViewMode(mode: ToolViewMode) {
  $toolViewMode.set(mode)
}

export function setKeepToolCallsExpanded(v: boolean) {
  $keepToolCallsExpanded.set(v)
}

export function $toolDisclosureOpen(id: string): ReadableAtom<boolean | undefined> {
  let cached = disclosureOpenCache.get(id)

  if (!cached) {
    cached = computed($toolDisclosureStates, states => states[id])
    disclosureOpenCache.set(id, cached)
  }

  return cached
}

function loadToolDisclosureStates(): ToolDisclosureStates {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(TOOL_DISCLOSURE_STORAGE_KEY)

    if (!raw) {
      return {}
    }

    const parsed = JSON.parse(raw) as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>)
        .filter((entry): entry is [string, boolean] => typeof entry[0] === 'string' && typeof entry[1] === 'boolean')
        .slice(-MAX_DISCLOSURE_STATES)
    )
  } catch {
    return {}
  }
}

function persistToolDisclosureStates(states: ToolDisclosureStates) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    const entries = Object.entries(states).slice(-MAX_DISCLOSURE_STATES)

    window.localStorage.setItem(TOOL_DISCLOSURE_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)))
  } catch {
    // Tool disclosure is a local UI preference; ignore storage failures.
  }
}

export function setToolDisclosureOpen(id: string, open: boolean) {
  if (!id) {
    return
  }

  const current = $toolDisclosureStates.get()

  if (current[id] === open) {
    return
  }

  $toolDisclosureStates.set({ ...current, [id]: open })
}
