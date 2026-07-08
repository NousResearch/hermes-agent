import { atom, computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'
import { normalize } from '@/lib/text'

import {
  $rightRailActiveTabId,
  PREVIEW_PANE_ID,
  RIGHT_RAIL_PREVIEW_TAB_ID,
  type RightRailTabId,
  selectRightRailTab
} from './layout'
import { setPaneOpen } from './panes'
import { $activeSessionId, $selectedStoredSessionId } from './session'

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

export interface PreviewTab {
  id: `preview:${string}`
  target: PreviewTarget
}

export interface FilePreviewTab {
  id: `file:${string}`
  target: PreviewTarget
}

const REGISTRY_STORAGE_KEY = 'hermes.desktop.sessionPreviews.v1'
const TABS_STORAGE_KEY = 'hermes.desktop.filePreviewTabs.v1'
const MAX_RECORDS_PER_SESSION = 8
const MAX_SESSIONS = 120

export const $previewTabs = atom<PreviewTab[]>([])
// Persisted so open file-preview tabs survive a relaunch; content is re-read
// from each target's path/url on demand. Invalid rows are dropped on load and
// inline image bytes (megabytes) are stripped on save, mirroring the registry.
export const $filePreviewTabs = persistentAtom<FilePreviewTab[]>(TABS_STORAGE_KEY, [], {
  decode: raw => {
    const parsed = JSON.parse(raw) as unknown

    return Array.isArray(parsed) ? parsed.filter(isFilePreviewTab) : []
  },
  encode: tabs => JSON.stringify(tabs, (key, value) => (key === 'dataUrl' ? undefined : value))
})

function isPreviewTabId(id: RightRailTabId): id is PreviewTab['id'] {
  return id.startsWith('preview:')
}

export function previewTabId(target: PreviewTarget): PreviewTab['id'] {
  return `preview:${target.url || target.source}`
}

function activePreviewTab(tabs = $previewTabs.get()): PreviewTab | null {
  const activeId = $rightRailActiveTabId.get()

  if (isPreviewTabId(activeId)) {
    const active = tabs.find(tab => tab.id === activeId)

    if (active) {
      return active
    }
  }

  return tabs.at(-1) ?? null
}

function livePreviewTabsFromRecords(records: readonly SessionPreviewRecord[] | undefined): PreviewTab[] {
  return (records ?? [])
    .filter(record => record.autoOpen !== false && !record.dismissedAt)
    .map(record => ({ id: previewTabId(record.normalized), target: record.normalized }))
}

function ensureRightRailActiveTabIsValid() {
  const activeId = $rightRailActiveTabId.get()

  if (activeId.startsWith('file:') && !$filePreviewTabs.get().some(tab => tab.id === activeId)) {
    selectRightRailTab($previewTabs.get()[0]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  } else if (isPreviewTabId(activeId) && !$previewTabs.get().some(tab => tab.id === activeId)) {
    selectRightRailTab($previewTabs.get().at(-1)?.id ?? $filePreviewTabs.get()[0]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

ensureRightRailActiveTabIsValid()

export const $filePreviewTarget = computed([$filePreviewTabs, $rightRailActiveTabId], (tabs, activeTabId) => {
  if (!activeTabId.startsWith('file:')) {
    return null
  }

  return tabs.find(tab => tab.id === activeTabId)?.target ?? null
})

export const $previewTarget = computed([$previewTabs, $rightRailActiveTabId], (tabs, activeTabId) => {
  if (isPreviewTabId(activeTabId)) {
    return tabs.find(tab => tab.id === activeTabId)?.target ?? tabs.at(-1)?.target ?? null
  }

  return tabs.at(-1)?.target ?? null
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

function syncPreviewPaneOpen() {
  setPaneOpen(PREVIEW_PANE_ID, Boolean($previewTabs.get().length > 0 || $filePreviewTabs.get().length > 0))
}

function openPreviewTab(target: PreviewTarget) {
  const id = previewTabId(target)
  const current = $previewTabs.get()
  const index = current.findIndex(tab => tab.id === id)
  const tab: PreviewTab = { id, target }

  $previewTabs.set(index === -1 ? [...current, tab] : current.map((item, i) => (i === index ? tab : item)))
  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(id)
}

export function setPreviewTarget(target: PreviewTarget | null) {
  if (!target) {
    $previewTabs.set([])
    ensureRightRailActiveTabIsValid()
    syncPreviewPaneOpen()

    return
  }

  if (isSamePreviewTarget(activePreviewTab()?.target ?? null, target)) {
    openPreviewTab(target)

    return
  }

  openPreviewTab(target)
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
  setPaneOpen(PREVIEW_PANE_ID, true)
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

function isFilePreviewTab(value: unknown): value is FilePreviewTab {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return typeof r.id === 'string' && r.id.startsWith('file:') && isPreviewTarget(r.target)
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

      const valid = records.filter(isPreviewRecord).slice(-MAX_RECORDS_PER_SESSION)

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
    const pruned = pruneRegistry(registry)

    if (Object.keys(pruned).length === 0) {
      window.localStorage.removeItem(REGISTRY_STORAGE_KEY)

      return
    }

    const lean = JSON.stringify(pruned, (key, value) => (key === 'dataUrl' ? undefined : value))
    window.localStorage.setItem(REGISTRY_STORAGE_KEY, lean)
  } catch {
    // Session previews are a desktop convenience; storage failures are nonfatal.
  }
}

function pruneRegistry(registry: SessionPreviewRegistry): SessionPreviewRegistry {
  const entries = Object.entries(registry)
    .map(
      ([sessionId, records]) =>
        [sessionId, [...records].sort((a, b) => a.createdAt - b.createdAt).slice(-MAX_RECORDS_PER_SESSION)] as const
    )
    .filter(([, records]) => records.length > 0)
    .sort(([, a], [, b]) => (b.at(-1)?.createdAt ?? 0) - (a.at(-1)?.createdAt ?? 0))
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
  const normalized = previewTargetForSource(target, source)
  const existing = records.find(record => record.normalized.url === normalized.url)

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
      [id]: [...records.filter(record => record.normalized.url !== normalized.url), nextRecord]
    })
  )

  return nextRecord
}

export function getSessionPreviewRecords(sessionId: string | null | undefined): SessionPreviewRecord[] {
  const id = sessionId?.trim()

  if (!id) {
    return []
  }

  return ($sessionPreviewRegistry.get()[id] ?? []).filter(record => !record.dismissedAt && record.autoOpen !== false)
}

export function getSessionPreviewRecord(sessionId: string | null | undefined): SessionPreviewRecord | null {
  return getSessionPreviewRecords(sessionId).at(-1) ?? null
}

export function restoreSessionPreviewTabs(sessionId: string | null | undefined) {
  const tabs = livePreviewTabsFromRecords(getSessionPreviewRecords(sessionId))

  $previewTabs.set(tabs)

  if (tabs.length > 0) {
    setPaneOpen(PREVIEW_PANE_ID, true)
    selectRightRailTab(tabs.at(-1)!.id)
  } else {
    ensureRightRailActiveTabIsValid()
    syncPreviewPaneOpen()
  }
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
  const normalized = record?.normalized ?? previewTargetForSource(target, source)

  openPreviewTab(normalized)

  return record
}

export function setCurrentSessionPreviewTarget(
  target: PreviewTarget,
  source: PreviewRecordSource,
  rawTarget = target.source
): SessionPreviewRecord | null {
  return setSessionPreviewTarget(currentPreviewSessionId(), target, source, rawTarget)
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
  const targetUrl = url || activePreviewTab()?.target.url || records.find(record => !record.dismissedAt)?.normalized.url

  if (!targetUrl) {
    return
  }

  const dismissedRecords = records.map(record =>
    record.normalized.url === targetUrl
      ? {
          ...record,
          autoOpen: false,
          dismissedAt: now
        }
      : record
  )

  $sessionPreviewRegistry.set({
    ...current,
    [id]: dismissedRecords
  })
}

function closePreviewTab(tabId: RightRailTabId, persistDismissal = true) {
  const current = $previewTabs.get()

  const targetTab =
    isPreviewTabId(tabId) ? current.find(tab => tab.id === tabId) : tabId === RIGHT_RAIL_PREVIEW_TAB_ID ? activePreviewTab(current) : null

  if (!targetTab) {
    syncPreviewPaneOpen()

    return
  }

  const index = current.findIndex(tab => tab.id === targetTab.id)

  if (persistDismissal) {
    dismissSessionPreview(currentPreviewSessionId(), targetTab.target.url)
  }

  const next = current.filter(tab => tab.id !== targetTab.id)

  $previewTabs.set(next)

  if ($rightRailActiveTabId.get() === targetTab.id || $rightRailActiveTabId.get() === RIGHT_RAIL_PREVIEW_TAB_ID) {
    selectRightRailTab(next[Math.min(index, next.length - 1)]?.id ?? $filePreviewTabs.get()[0]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }

  syncPreviewPaneOpen()
}

/** User clicked a close/hide control — clear the matching or active live-preview tab and persist dismissal. */
export function dismissPreviewTarget(target?: string) {
  const matchingTab = target
    ? $previewTabs
        .get()
        .find(tab => tab.target.source === target || tab.target.url === target || tab.target.path === target)
    : null

  closePreviewTab(matchingTab?.id ?? $rightRailActiveTabId.get())
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
    selectRightRailTab(next[Math.min(index, next.length - 1)]?.id ?? $previewTabs.get().at(-1)?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }

  syncPreviewPaneOpen()
}

export function closeRightRailTab(tabId: RightRailTabId) {
  if (tabId === RIGHT_RAIL_PREVIEW_TAB_ID || isPreviewTabId(tabId)) {
    closePreviewTab(tabId)

    return
  }

  closeFilePreviewTab(tabId)
}

export const closeActiveRightRailTab = () => closeRightRailTab($rightRailActiveTabId.get())

// The rail's visible tab order: live preview tabs first, then the file tabs in
// their stored order. Mirrors `ChatPreviewRail`'s `tabs` memo so "close others /
// to the right" act on what the user actually sees.
function rightRailTabOrder(): RightRailTabId[] {
  return [...$previewTabs.get().map(tab => tab.id), ...$filePreviewTabs.get().map(tab => tab.id)]
}

/** Close every rail tab except `keepId`, then make `keepId` active. */
export function closeOtherRightRailTabs(keepId: RightRailTabId) {
  for (const id of rightRailTabOrder()) {
    if (id !== keepId) {
      closeRightRailTab(id)
    }
  }

  selectRightRailTab(keepId)
}

/** Close every rail tab positioned after `tabId` (VS Code's "Close to the Right"). */
export function closeRightRailTabsToRight(tabId: RightRailTabId) {
  const order = rightRailTabOrder()
  const index = order.indexOf(tabId)

  if (index === -1) {
    return
  }

  for (const id of order.slice(index + 1)) {
    closeRightRailTab(id)
  }
}

/** Dismisses every live preview + every file tab so the rail pane unmounts. */
export function closeRightRail() {
  for (const tab of $previewTabs.get()) {
    dismissSessionPreview(currentPreviewSessionId(), tab.target.url)
  }

  $previewTabs.set([])
  $filePreviewTabs.set([])
  setPaneOpen(PREVIEW_PANE_ID, false)
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
}

export function clearSessionPreviewRegistry() {
  $sessionPreviewRegistry.set({})
  $previewTabs.set([])
  $filePreviewTabs.set([])
  setPaneOpen(PREVIEW_PANE_ID, false)
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
    status: normalize(text).startsWith('error:') ? 'error' : 'complete'
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
