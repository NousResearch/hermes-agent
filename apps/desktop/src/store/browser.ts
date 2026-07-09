import { atom, computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

import { PREVIEW_PANE_ID, RIGHT_RAIL_BROWSER_TAB_ID, selectRightRailTab } from './layout'
import { setPaneOpen } from './panes'
import { $activeSessionId, $selectedStoredSessionId } from './session'

export interface BrowserSessionState {
  title?: string
  updatedAt: number
  url: string
}

export type BrowserDriveAction = 'act' | 'goBack' | 'goForward' | 'navigate' | 'open' | 'reload' | 'snapshot'

export type BrowserDomActionKind = 'click' | 'press' | 'scroll' | 'select' | 'setValue' | 'type'

export interface BrowserDomActionPayload {
  amount?: number
  direction?: 'down' | 'left' | 'right' | 'up'
  index?: number
  key?: string
  kind: BrowserDomActionKind
  selector?: string
  text?: string
  value?: string
}

export interface BrowserDrivePayload {
  action: BrowserDriveAction
  domAction?: BrowserDomActionPayload
  requestId?: string
  sessionId?: string
  title?: string
  url?: string
}

export interface BrowserDriveCommand extends BrowserDrivePayload {
  id: number
  sessionId: string
}

type BrowserSessionRegistry = Record<string, BrowserSessionState>

const STORAGE_KEY = 'hermes.desktop.browserRail.v1'
const DEFAULT_BROWSER_URL = 'about:blank'
const DRAFT_BROWSER_SESSION_ID = 'draft'

function isBrowserSessionState(value: unknown): value is BrowserSessionState {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    typeof r.url === 'string' &&
    r.url.length > 0 &&
    typeof r.updatedAt === 'number' &&
    Number.isFinite(r.updatedAt) &&
    (r.title === undefined || typeof r.title === 'string')
  )
}

function sanitizeRegistry(value: unknown): BrowserSessionRegistry {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: BrowserSessionRegistry = {}

  for (const [sessionId, state] of Object.entries(value as Record<string, unknown>)) {
    if (sessionId && isBrowserSessionState(state)) {
      out[sessionId] = {
        title: state.title,
        updatedAt: state.updatedAt,
        url: state.url
      }
    }
  }

  return out
}

export const $browserSessionRegistry = persistentAtom<BrowserSessionRegistry>(
  STORAGE_KEY,
  {},
  Codecs.json(sanitizeRegistry)
)

export const $browserSessionId = computed(
  [$activeSessionId, $selectedStoredSessionId],
  (activeSessionId, selectedStoredSessionId) => activeSessionId || selectedStoredSessionId || DRAFT_BROWSER_SESSION_ID
)

export const $browserCurrentState = computed(
  [$browserSessionRegistry, $browserSessionId],
  (registry, sessionId) => registry[sessionId] ?? { updatedAt: 0, url: DEFAULT_BROWSER_URL }
)

export const $browserCurrentSessionHasBrowser = computed(
  [$browserSessionRegistry, $browserSessionId],
  (registry, sessionId) => Boolean(registry[sessionId])
)

export const $browserCurrentUrl = computed($browserCurrentState, state => state.url)
export const $browserCurrentTitle = computed($browserCurrentState, state => state.title || '')
export const $browserDriveCommand = atom<BrowserDriveCommand | null>(null)

let nextBrowserDriveCommandId = 0

export function normalizeBrowserUrl(input: string): string {
  const raw = input.trim()

  if (!raw) {
    return DEFAULT_BROWSER_URL
  }

  if (raw === 'about:blank') {
    return raw
  }

  const withScheme = /^[a-zA-Z][a-zA-Z\d+.-]*:/.test(raw) ? raw : `https://${raw}`

  try {
    const parsed = new URL(withScheme)

    if (!['http:', 'https:', 'about:'].includes(parsed.protocol)) {
      return DEFAULT_BROWSER_URL
    }

    if (parsed.protocol === 'about:' && parsed.pathname !== 'blank') {
      return DEFAULT_BROWSER_URL
    }

    return parsed.toString()
  } catch {
    return DEFAULT_BROWSER_URL
  }
}

function browserSessionKey(sessionId: string | null | undefined): string {
  return sessionId?.trim() || DRAFT_BROWSER_SESSION_ID
}

export function hasBrowserSession(sessionId: string | null | undefined): boolean {
  return Boolean($browserSessionRegistry.get()[browserSessionKey(sessionId)])
}

export function hasCurrentBrowserSession(): boolean {
  return hasBrowserSession($browserSessionId.get())
}

export function ensureBrowserSessionState(sessionId: string | null | undefined): BrowserSessionState {
  const key = browserSessionKey(sessionId)
  const current = $browserSessionRegistry.get()
  const existing = current[key]

  if (existing) {
    return existing
  }

  const next: BrowserSessionState = { updatedAt: Date.now(), url: DEFAULT_BROWSER_URL }
  $browserSessionRegistry.set({ ...current, [key]: next })

  return next
}

export function setBrowserSessionState(
  sessionId: string | null | undefined,
  next: Partial<Pick<BrowserSessionState, 'title' | 'url'>>
): BrowserSessionState {
  const key = browserSessionKey(sessionId)
  const current = $browserSessionRegistry.get()
  const previous = current[key] ?? { updatedAt: 0, url: DEFAULT_BROWSER_URL }
  const url = next.url !== undefined ? normalizeBrowserUrl(next.url) : previous.url
  const title = next.title !== undefined ? next.title : previous.title

  const nextState = {
    title,
    updatedAt: Date.now(),
    url
  }

  $browserSessionRegistry.set({
    ...current,
    [key]: nextState
  })

  return nextState
}

export function setCurrentBrowserState(next: Partial<Pick<BrowserSessionState, 'title' | 'url'>>) {
  setBrowserSessionState($browserSessionId.get(), next)
}

export function showBrowserRailForSession(sessionId: string | null | undefined): boolean {
  if (!hasBrowserSession(sessionId)) {
    return false
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(RIGHT_RAIL_BROWSER_TAB_ID)

  return true
}

export function openBrowserRail(url?: string, sessionId: string | null | undefined = $browserSessionId.get()) {
  if (url !== undefined) {
    setBrowserSessionState(sessionId, { url })
  } else {
    ensureBrowserSessionState(sessionId)
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(RIGHT_RAIL_BROWSER_TAB_ID)
}

export function driveBrowser(payload: BrowserDrivePayload) {
  if (!payload?.action) {
    return
  }

  const sessionId = browserSessionKey(payload.sessionId || $browserSessionId.get())

  const command: BrowserDriveCommand = {
    ...payload,
    id: ++nextBrowserDriveCommandId,
    sessionId
  }

  if (payload.action === 'open') {
    openBrowserRail(payload.url, sessionId)
  } else if (payload.action === 'navigate') {
    if (payload.url !== undefined) {
      setBrowserSessionState(sessionId, { title: payload.title, url: payload.url })
    }

    openBrowserRail(payload.url, sessionId)
  } else {
    openBrowserRail(undefined, sessionId)
  }

  $browserDriveCommand.set(command)
}

export function closeBrowserRailForSession(sessionId: string | null | undefined): boolean {
  const key = browserSessionKey(sessionId)
  const current = $browserSessionRegistry.get()

  if (!current[key]) {
    return false
  }

  const { [key]: _closed, ...next } = current
  $browserSessionRegistry.set(next)

  if ($browserDriveCommand.get()?.sessionId === key) {
    $browserDriveCommand.set(null)
  }

  return true
}

export function closeCurrentBrowserSession(): boolean {
  return closeBrowserRailForSession($browserSessionId.get())
}

export function resetBrowserRegistryForTests() {
  $browserSessionRegistry.set({})
  $browserDriveCommand.set(null)
  nextBrowserDriveCommandId = 0
}

// Wire incoming drive commands from the Electron main process (via preload
// hermesDesktop.browser and 'hermes:browser:drive' channel).
// This is the key piece for "built-in browser" control by the agent / backend.
if (typeof window !== 'undefined') {
  window.hermesDesktop?.browser?.onDrive?.(payload => driveBrowser(payload))
}
