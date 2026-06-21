import { atom, computed } from 'nanostores'

import { $rightRailActiveTabId, RIGHT_RAIL_PREVIEW_TAB_ID, type RightRailTabId, selectRightRailTab } from './layout'
import { $activeSessionId, $selectedStoredSessionId } from './session'
import { setPaneOpen } from './panes'

export interface PreviewTarget {
  binary?: boolean
  byteSize?: number
  /** Inline image bytes (a `data:` URL) when the renderer already holds them —
   * e.g. a pasted/dropped screenshot whose only on-disk copy is a transient
   * path the preview can't reliably re-read. Rendered directly and NOT
   * persisted to the session-preview registry (it would bloat localStorage). */
  dataUrl?: string
  kind: 'file' | 'url'
  label: string
  large?: boolean
  language?: string
  mimeType?: string
  path?: string
  previewKind?: 'binary' | 'html' | 'image' | 'text'
  renderMode?: 'preview' | 'source'
  source: string
  url: string
}

export interface PreviewServerRestart {
  message?: string
  status: 'complete' | 'error' | 'running'
  taskId: string
  url: string
}

export type PreviewRecordSource = 'explicit-link' | 'file-browser' | 'manual' | 'tool-result'

export interface SessionPreviewRecord {
  autoOpen?: boolean
  createdAt: number
  dismissedAt?: number
  id: string
  normalized: PreviewTarget
  sessionId: string
  source: PreviewRecordSource
  target: string
}

type SessionPreviewRegistry = Record<string, SessionPreviewRecord[]>

export interface FilePreviewTab {
  id: `file:${string}`
  target: PreviewTarget
}

const REGISTRY_STORAGE_KEY = 'hermes.desktop.sessionPreviews.v1'
const MAX_RECORDS_PER_SESSION = 1
const MAX_SESSIONS = 120

export const $previewTarget = atom<PreviewTarget | null>(null)
export const $filePreviewTabs = atom<FilePreviewTab[]>([])
export const $filePreviewTarget = computed([$filePreviewTabs, $rightRailActiveTabId], (tabs, activeTabId) => {
  if (!activeTabId.startsWith('file:')) {
    return null
  }

  return tabs.find(tab => tab.id === activeTabId)?.target ?? null
})
export const $previewReloadRequest = atom(0)
export const $previewServerRestart = atom<PreviewServerRestart | null>(null)
export const $previewServerRestartStatus = computed($previewServerRestart, restart => restart?.status ?? 'idle')
export const $sessionPreviewRegistry = atom<SessionPreviewRegistry>(loadSessionPreviewRegistry())

$sessionPreviewRegistry.subscribe(persistSessionPreviewRegistry)

function isSamePreviewTarget(a: PreviewTarget | null, b: PreviewTarget | null): boolean {
  if (a === b) {
    return true
  }

  if (!a || !b) {
    return false
  }

  return (
    a.kind === b.kind &&
    a.label === b.label &&
    a.renderMode === b.renderMode &&
    a.source === b.source &&
    a.url === b.url
  )
}

export function setPreviewTarget(target: PreviewTarget | null) {
  if (isSamePreviewTarget($previewTarget.get(), target)) {
    if (target) {
      selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
    }

    return
  }

  $previewTarget.set(target)

  if (target) {
    setPaneOpen('preview', true)
    selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

export function filePreviewTabId(target: PreviewTarget): `file:${string}` {
  return `file:${target.url}`
}

function openFilePreviewTarget(target: PreviewTarget) {
  const id = filePreviewTabId(target)
  const current = $filePreviewTabs.get()
  const index = current.findIndex(tab => tab.id === id)
  const tab: FilePreviewTab = { id, target }

  $filePreviewTabs.set(index === -1 ? [...current, tab] : current.map((item, i) => (i === index ? tab : item)))
  selectRightRailTab(id)
}

// Manual/file-browser opens are "peeking at a file" → source view in the file
// pane. Tool/explicit-link opens are runnable artifacts → live preview pane.
function isFilePreviewSource(source: PreviewRecordSource): boolean {
  return source === 'file-browser' || source === 'manual'
}

function previewTargetForSource(target: PreviewTarget, source: PreviewRecordSource): PreviewTarget {
  if (target.kind !== 'file' || target.previewKind !== 'html') {
    return target
  }

  return { ...target, renderMode: isFilePreviewSource(source) ? 'source' : 'preview' }
}

function tryOpenFilePreview(target: PreviewTarget, source: PreviewRecordSource): boolean {
  if (target.kind !== 'file' || !isFilePreviewSource(source)) {
    return false
  }

  openFilePreviewTarget(previewTargetForSource(target, source))

  return true
}

function isPreviewTarget(value: unknown): value is PreviewTarget {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    (r.kind === 'file' || r.kind === 'url') &&
    typeof r.label === 'string' &&
    typeof r.source === 'string' &&
    typeof r.url === 'string'
  )
}

function isPreviewRecord(value: unknown): value is SessionPreviewRecord {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    typeof r.createdAt === 'number' &&
    typeof r.id === 'string' &&
    isPreviewTarget(r.normalized) &&
    typeof r.sessionId === 'string' &&
    ['explicit-link', 'file-browser', 'manual', 'tool-result'].includes(String(r.source)) &&
    typeof r.target === 'string' &&
    (r.dismissedAt === undefined || typeof r.dismissedAt === 'number')
  )
}

function loadSessionPreviewRegistry(): SessionPreviewRegistry {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(REGISTRY_STORAGE_KEY)

    if (!raw) {
      return {}
    }

    const parsed = JSON.parse(raw) as unknown

    if (!parsed || typeof parsed !== 'object') {
      return {}
    }

    const out: SessionPreviewRegistry = {}

    for (const [sessionId, records] of Object.entries(parsed as Record<string, unknown>)) {
      if (!Array.isArray(records)) {
        continue
      }

      const valid = records.filter(isPreviewRecord).slice(0, MAX_RECORDS_PER_SESSION)

      if (valid.length > 0) {
        out[sessionId] = valid
      }
    }

    return pruneRegistry(out)
  } catch {
    return {}
  }
}

function persistSessionPreviewRegistry(registry: SessionPreviewRegistry) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    // Drop the inline image bytes before persisting — a screenshot data URL is
    // megabytes and would blow the localStorage quota. On reload the record
    // falls back to reading its `path`/`url`.
    const lean = JSON.stringify(pruneRegistry(registry), (key, value) => (key === 'dataUrl' ? undefined : value))
    window.localStorage.setItem(REGISTRY_STORAGE_KEY, lean)
  } catch {
    // Session previews are a desktop convenience; storage failures are nonfatal.
  }
}

function pruneRegistry(registry: SessionPreviewRegistry): SessionPreviewRegistry {
  const entries = Object.entries(registry)
    .map(
      ([sessionId, records]) =>
        [sessionId, [...records].sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_RECORDS_PER_SESSION)] as const
    )
    .filter(([, records]) => records.length > 0)
    .sort(([, a], [, b]) => (b[0]?.createdAt ?? 0) - (a[0]?.createdAt ?? 0))
    .slice(0, MAX_SESSIONS)

  return Object.fromEntries(entries)
}

function currentPreviewSessionId(): string {
  return $selectedStoredSessionId.get() || $activeSessionId.get() || ''
}

function recordId(sessionId: string, target: PreviewTarget): string {
  return `${sessionId}:${target.url}`
}

export function registerSessionPreview(
  sessionId: string | null | undefined,
  target: PreviewTarget,
  source: PreviewRecordSource,
  rawTarget = target.source
): SessionPreviewRecord | null {
  const id = sessionId?.trim()

  if (!id) {
    return null
  }

  const current = $sessionPreviewRegistry.get()
  const now = Date.now()
  const records = current[id] ?? []
  const existing = records.find(record => record.normalized.url === target.url)
  const normalized = previewTargetForSource(target, source)

  const nextRecord: SessionPreviewRecord = {
    autoOpen: true,
    createdAt: now,
    id: existing?.id || recordId(id, target),
    normalized,
    sessionId: id,
    source,
    target: rawTarget || target.source
  }

  $sessionPreviewRegistry.set(
    pruneRegistry({
      ...current,
      [id]: [nextRecord]
    })
  )

  return nextRecord
}

export function setSessionPreviewTarget(
  sessionId: string | null | undefined,
  target: PreviewTarget,
  source: PreviewRecordSource,
  rawTarget = target.source
): SessionPreviewRecord | null {
  if (tryOpenFilePreview(target, source)) {
    return null
  }

  const record = registerSessionPreview(sessionId, target, source, rawTarget)

  setPreviewTarget(record?.normalized ?? previewTargetForSource(target, source))

  return record
}

export function setCurrentSessionPreviewTarget(
  target: PreviewTarget,
  source: PreviewRecordSource,
  rawTarget = target.source
): SessionPreviewRecord | null {
  return setSessionPreviewTarget(currentPreviewSessionId(), target, source, rawTarget)
}

export function getSessionPreviewRecord(sessionId: string | null | undefined): SessionPreviewRecord | null {
  const id = sessionId?.trim()

  if (!id) {
    return null
  }

  return $sessionPreviewRegistry.get()[id]?.find(record => !record.dismissedAt && record.autoOpen !== false) ?? null
}

export function dismissSessionPreview(sessionId: string | null | undefined, url?: string) {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const current = $sessionPreviewRegistry.get()
  const records = current[id]

  if (!records?.length) {
    return
  }

  const now = Date.now()
  const targetUrl = url || records.find(record => !record.dismissedAt)?.normalized.url

  if (!targetUrl) {
    return
  }

  // The preview rail is a single active file, not a back stack. Dismissing the
  // current preview should leave the rail closed instead of revealing an older
  // record for the same session.
  const dismissedRecords = records.map(record => ({
    ...record,
    autoOpen: false,
    dismissedAt: now
  }))

  $sessionPreviewRegistry.set({
    ...current,
    [id]: dismissedRecords
  })
}

/** User clicked the close X — clear the target and persist dismissal for the current session. */
export function dismissPreviewTarget() {
  const current = $previewTarget.get()

  if (current?.url) {
    dismissSessionPreview(currentPreviewSessionId(), current.url)
  }

  $previewTarget.set(null)

  if ($rightRailActiveTabId.get() === RIGHT_RAIL_PREVIEW_TAB_ID) {
    selectRightRailTab($filePreviewTabs.get()[0]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

function closeFilePreviewTab(tabId: RightRailTabId) {
  if (!tabId.startsWith('file:')) {
    return
  }

  const current = $filePreviewTabs.get()
  const index = current.findIndex(tab => tab.id === tabId)

  if (index === -1) {
    return
  }

  const next = current.filter(tab => tab.id !== tabId)

  $filePreviewTabs.set(next)

  if ($rightRailActiveTabId.get() === tabId) {
    selectRightRailTab(next[Math.min(index, next.length - 1)]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

export function closeRightRailTab(tabId: RightRailTabId) {
  if (tabId === RIGHT_RAIL_PREVIEW_TAB_ID) {
    if ($previewTarget.get()) {
      dismissPreviewTarget()
    }

    return
  }

  closeFilePreviewTab(tabId)
}

export const closeActiveRightRailTab = () => closeRightRailTab($rightRailActiveTabId.get())

/** Dismisses the active preview + every file tab so the rail pane unmounts. */
export function closeRightRail() {
  if ($previewTarget.get()) {
    dismissPreviewTarget()
  }

  $filePreviewTabs.set([])
}

export function clearSessionPreviewRegistry() {
  $sessionPreviewRegistry.set({})
  setPreviewTarget(null)
  $filePreviewTabs.set([])
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
}

export function requestPreviewReload() {
  $previewReloadRequest.set($previewReloadRequest.get() + 1)
}

export function beginPreviewServerRestart(taskId: string, url: string) {
  $previewServerRestart.set({ status: 'running', taskId, url })
}

export function completePreviewServerRestart(taskId: string, text: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId) {
    return
  }

  $previewServerRestart.set({
    ...current,
    message: text,
    status: text.trim().toLowerCase().startsWith('error:') ? 'error' : 'complete'
  })
}

export function progressPreviewServerRestart(taskId: string, text: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId || current.status !== 'running') {
    return
  }

  $previewServerRestart.set({
    ...current,
    message: text
  })
}

export function failPreviewServerRestart(taskId: string, message: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId || current.status !== 'running') {
    return
  }

  $previewServerRestart.set({
    ...current,
    message,
    status: 'error'
  })
}
