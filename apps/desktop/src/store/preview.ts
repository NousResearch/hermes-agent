import { atom, computed } from 'nanostores'

import { desktopFsCacheKey } from '@/lib/desktop-fs'
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
import {
  clampFloatingGeometry,
  type FloatingGeometry,
  type PreviewSnapSlot,
  type PreviewViewport
} from './preview-surface-layout'
import { $activeGatewayProfile, normalizeProfileKey } from './profile'
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

export type PreviewRecordSource = 'agent-request' | 'explicit-link' | 'file-browser' | 'manual' | 'tool-result'

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

export interface WebPreviewTab {
  id: `preview:${string}`
  target: PreviewTarget
}

export type UtilityPreviewKind = 'host-vnc' | 'terminal'

export interface UtilityPreviewTab {
  id: `utility:${UtilityPreviewKind}`
  kind: UtilityPreviewKind
}

export type PreviewSurfacePlacement = 'docked' | 'floating' | 'maximized' | 'minimized' | PreviewSnapSlot

export interface PreviewSurfaceRestore {
  geometry?: FloatingGeometry
  placement: Exclude<PreviewSurfacePlacement, 'maximized' | 'minimized'>
}

export interface PreviewSurfaceLayout {
  geometry?: FloatingGeometry
  placement: PreviewSurfacePlacement
  restore?: PreviewSurfaceRestore
}

const REGISTRY_STORAGE_KEY = 'hermes.desktop.sessionPreviews.v1'
const TABS_STORAGE_KEY = 'hermes.desktop.filePreviewTabs.v1'
const WEB_TABS_STORAGE_KEY = 'hermes.desktop.webPreviewTabs.v1'
const SURFACE_LAYOUTS_STORAGE_KEY = 'hermes.desktop.previewSurfaceLayouts.v1'
const WORKSPACE_SCOPE_STORAGE_KEY = 'hermes.desktop.previewWorkspaceScope.v1'
const MAX_RECORDS_PER_SESSION = 1
const MAX_SESSIONS = 120

function opaqueScopeId(value: string): string {
  let first = 0x811c9dc5
  let second = 0x9e3779b9

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    first = Math.imul(first ^ code, 0x01000193)
    second = Math.imul(second ^ code, 0x85ebca6b)
  }

  return `scope:${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0).toString(16).padStart(8, '0')}`
}

export function previewWorkspaceScopeId(
  sessionId: string,
  connectionKey = desktopFsCacheKey(),
  profileKey = $activeGatewayProfile.get()
): string {
  return opaqueScopeId(`${connectionKey}\u0000${normalizeProfileKey(profileKey)}\u0000${sessionId.trim()}`)
}

function storedPreviewWorkspaceScope(): string {
  if (typeof window === 'undefined') {
    return ''
  }

  const value = window.localStorage.getItem(WORKSPACE_SCOPE_STORAGE_KEY) || ''

  return /^scope:[a-f0-9]{16}$/.test(value) ? value : ''
}

let activePreviewWorkspaceScope = storedPreviewWorkspaceScope()

export const $previewTarget = atom<PreviewTarget | null>(null)
export const $utilityPreviewTabs = atom<UtilityPreviewTab[]>([])
export const $webPreviewTabs = persistentAtom<WebPreviewTab[]>(WEB_TABS_STORAGE_KEY, [], {
  decode: raw => {
    const parsed = JSON.parse(raw) as unknown

    return Array.isArray(parsed)
      ? parsed.filter(isWebPreviewTab).filter(isPersistableWebPreviewTab).map(migrateWebPreviewTab)
      : []
  },
  encode: tabs => JSON.stringify(tabs.filter(isPersistableWebPreviewTab))
})
// Persisted so open file-preview tabs survive a relaunch; content is re-read
// from each target's path/url on demand. Invalid rows are dropped on load and
// inline image bytes (megabytes) are stripped on save, mirroring the registry.
export const $filePreviewTabs = persistentAtom<FilePreviewTab[]>(TABS_STORAGE_KEY, [], {
  decode: raw => {
    const parsed = JSON.parse(raw) as unknown

    return Array.isArray(parsed) ? parsed.filter(isFilePreviewTab).map(migrateFilePreviewTab) : []
  },
  encode: tabs => JSON.stringify(tabs, (key, value) => (key === 'dataUrl' ? undefined : value))
})
export const $previewSurfaceLayouts = persistentAtom<Partial<Record<RightRailTabId, PreviewSurfaceLayout>>>(
  SURFACE_LAYOUTS_STORAGE_KEY,
  {},
  {
    decode: raw => {
      const parsed = JSON.parse(raw) as unknown

      if (!parsed || typeof parsed !== 'object') {
        return {}
      }

      return Object.fromEntries(
        Object.entries(parsed as Record<string, unknown>).filter(
          ([tabId, layout]) => isPersistableLayoutTabId(tabId) && isPreviewSurfaceLayout(layout)
        )
      ) as Partial<Record<RightRailTabId, PreviewSurfaceLayout>>
    },
    encode: layouts =>
      JSON.stringify(Object.fromEntries(Object.entries(layouts).filter(([tabId]) => isPersistableLayoutTabId(tabId))))
  }
)

// Rewrite migrated/sanitized persistence immediately so credentials from a
// legacy URL-derived tab id cannot remain dormant in localStorage.
$filePreviewTabs.set([...$filePreviewTabs.get()])
$webPreviewTabs.set([...$webPreviewTabs.get()])
$previewSurfaceLayouts.set({ ...$previewSurfaceLayouts.get() })

// Drop a restored active tab that did not survive validation so the rail never
// points at a tab that is not there.
const restoredActiveTabId = $rightRailActiveTabId.get()

if (
  (restoredActiveTabId.startsWith('file:') && !$filePreviewTabs.get().some(tab => tab.id === restoredActiveTabId)) ||
  (restoredActiveTabId.startsWith('preview:') && !$webPreviewTabs.get().some(tab => tab.id === restoredActiveTabId)) ||
  (restoredActiveTabId.startsWith('utility:') && !$utilityPreviewTabs.get().some(tab => tab.id === restoredActiveTabId))
) {
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
}

export function setPreviewWorkspaceScope(
  sessionId: string,
  connectionKey = desktopFsCacheKey(),
  profileKey = $activeGatewayProfile.get()
): string {
  const id = sessionId.trim()
  const nextScope = id ? previewWorkspaceScopeId(id, connectionKey, profileKey) : ''

  if (nextScope === activePreviewWorkspaceScope) {
    return nextScope
  }

  activePreviewWorkspaceScope = nextScope

  // Clear session-scoped preview/file/web surfaces while preserving global
  // utility tabs. Terminal shells and the host display belong to the backend
  // connection, not to one chat, so switching sessions must not close them.
  const utilityTabs = $utilityPreviewTabs.get()
  const utilityIds = new Set<RightRailTabId>(utilityTabs.map(tab => tab.id))

  const utilityLayouts = Object.fromEntries(
    Object.entries($previewSurfaceLayouts.get()).filter(([tabId]) => utilityIds.has(tabId as RightRailTabId))
  ) as Partial<Record<RightRailTabId, PreviewSurfaceLayout>>

  const currentActiveTabId = $rightRailActiveTabId.get()

  const nextActiveUtilityId = utilityIds.has(currentActiveTabId)
    ? currentActiveTabId
    : (utilityTabs.at(-1)?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)

  // Clear the old scope's persisted surfaces before committing the new marker.
  // If the renderer is interrupted between these synchronous writes, a later
  // startup can restore an empty workspace, never old tabs under a new scope.
  $previewTarget.set(null)
  $webPreviewTabs.set([])
  $filePreviewTabs.set([])
  $previewSurfaceLayouts.set(utilityLayouts)
  setPaneOpen(PREVIEW_PANE_ID, utilityTabs.length > 0)
  selectRightRailTab(nextActiveUtilityId)

  if (typeof window !== 'undefined') {
    if (nextScope) {
      window.localStorage.setItem(WORKSPACE_SCOPE_STORAGE_KEY, nextScope)
    } else {
      window.localStorage.removeItem(WORKSPACE_SCOPE_STORAGE_KEY)
    }
  }

  return nextScope
}

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

function showLivePreviewTab() {
  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
}

export function utilityPreviewTabId(kind: UtilityPreviewKind): UtilityPreviewTab['id'] {
  return `utility:${kind}`
}

export function openUtilityPreviewTab(kind: UtilityPreviewKind) {
  const id = utilityPreviewTabId(kind)
  const current = $utilityPreviewTabs.get()

  if (!current.some(tab => tab.id === id)) {
    $utilityPreviewTabs.set([...current, { id, kind }])
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(id)
}

export function setPreviewTarget(target: PreviewTarget | null) {
  if (isSamePreviewTarget($previewTarget.get(), target)) {
    if (target) {
      showLivePreviewTab()
    }

    return
  }

  $previewTarget.set(target)

  if (target) {
    showLivePreviewTab()
  }
}

function createOpaquePreviewTabId(prefix: 'file' | 'preview'): `file:${string}` | `preview:${string}` {
  const uuid = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

  return `${prefix}:tab-${uuid}`
}

export function filePreviewTabId(_target: PreviewTarget): `file:${string}` {
  return createOpaquePreviewTabId('file') as `file:${string}`
}

export function webPreviewTabId(_target: PreviewTarget): `preview:${string}` {
  return createOpaquePreviewTabId('preview') as `preview:${string}`
}

function openWebPreviewTarget(target: PreviewTarget) {
  const current = $webPreviewTabs.get()
  const index = current.findIndex(tab => isSamePreviewTarget(tab.target, target))
  const id = index === -1 ? webPreviewTabId(target) : current[index]!.id
  const tab: WebPreviewTab = { id, target }

  $webPreviewTabs.set(index === -1 ? [...current, tab] : current.map((item, i) => (i === index ? tab : item)))
  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(id)
}

function openFilePreviewTarget(target: PreviewTarget) {
  const current = $filePreviewTabs.get()
  const index = current.findIndex(tab => isSamePreviewTarget(tab.target, target))
  const id = index === -1 ? filePreviewTabId(target) : current[index]!.id
  const tab: FilePreviewTab = { id, target }

  $filePreviewTabs.set(index === -1 ? [...current, tab] : current.map((item, i) => (i === index ? tab : item)))
  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(id)
}

// Manual/file-browser opens are "peeking at a file" → source view in the file
// pane. Tool/explicit-link opens are runnable artifacts → live preview pane.
function isFilePreviewSource(source: PreviewRecordSource): boolean {
  return source === 'agent-request' || source === 'file-browser' || source === 'manual'
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

function isWebPreviewTab(value: unknown): value is WebPreviewTab {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return typeof r.id === 'string' && r.id.startsWith('preview:') && isPreviewTarget(r.target)
}

function isOpaquePreviewTabId(id: string, prefix: 'file' | 'preview'): boolean {
  return new RegExp(`^${prefix}:tab-[a-z0-9-]+$`, 'i').test(id)
}

function migrateFilePreviewTab(tab: FilePreviewTab): FilePreviewTab {
  return isOpaquePreviewTabId(tab.id, 'file') ? tab : { ...tab, id: filePreviewTabId(tab.target) }
}

function migrateWebPreviewTab(tab: WebPreviewTab): WebPreviewTab {
  return isOpaquePreviewTabId(tab.id, 'preview') ? tab : { ...tab, id: webPreviewTabId(tab.target) }
}

function isPersistableWebPreviewTarget(target: PreviewTarget): boolean {
  if (target.dataUrl) {
    return false
  }

  try {
    const url = new URL(target.url)

    // Credential-bearing, stateful, and non-network URLs remain usable for
    // this process but are deliberately omitted from localStorage.
    return (
      (url.protocol === 'http:' || url.protocol === 'https:') &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash
    )
  } catch {
    return false
  }
}

function isPersistableWebPreviewTab(tab: WebPreviewTab): boolean {
  return isPersistableWebPreviewTarget(tab.target)
}

function isPersistableLayoutTabId(tabId: string): boolean {
  return (
    tabId === RIGHT_RAIL_PREVIEW_TAB_ID || isOpaquePreviewTabId(tabId, 'file') || isOpaquePreviewTabId(tabId, 'preview')
  )
}

function isFloatingGeometry(value: unknown): value is FloatingGeometry {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return ['height', 'width', 'x', 'y'].every(key => typeof r[key] === 'number' && Number.isFinite(r[key]))
}

const SURFACE_PLACEMENTS: readonly PreviewSurfacePlacement[] = [
  'docked',
  'floating',
  'minimized',
  'maximized',
  'left-half',
  'right-half',
  'top-half',
  'bottom-half',
  'left-third',
  'center-third',
  'right-third',
  'left-two-thirds',
  'right-two-thirds',
  'top-left-quarter',
  'top-right-quarter',
  'bottom-left-quarter',
  'bottom-right-quarter'
]

function isPreviewSurfaceRestore(value: unknown): value is PreviewSurfaceRestore {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    SURFACE_PLACEMENTS.includes(r.placement as PreviewSurfacePlacement) &&
    r.placement !== 'maximized' &&
    r.placement !== 'minimized' &&
    (r.geometry === undefined || isFloatingGeometry(r.geometry))
  )
}

function isPreviewSurfaceLayout(value: unknown): value is PreviewSurfaceLayout {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    SURFACE_PLACEMENTS.includes(r.placement as PreviewSurfacePlacement) &&
    (r.geometry === undefined || isFloatingGeometry(r.geometry)) &&
    (r.restore === undefined || isPreviewSurfaceRestore(r.restore))
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
    ['agent-request', 'explicit-link', 'file-browser', 'manual', 'tool-result'].includes(String(r.source)) &&
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

    for (const [scopeId, records] of Object.entries(parsed as Record<string, unknown>)) {
      if (!/^scope:[a-f0-9]{16}$/.test(scopeId) || !Array.isArray(records)) {
        continue
      }

      const valid = records
        .filter(isPreviewRecord)
        .filter(record => record.normalized.kind !== 'url' || isPersistableWebPreviewTarget(record.normalized))
        .map(record =>
          /^session-preview:tab-[a-z0-9-]+$/i.test(record.id)
            ? record
            : { ...record, id: createOpaqueSessionPreviewRecordId() }
        )
        .slice(0, MAX_RECORDS_PER_SESSION)

      if (valid.length > 0) {
        out[scopeId] = valid
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
    const pruned = pruneRegistry(registry)

    if (Object.keys(pruned).length === 0) {
      window.localStorage.removeItem(REGISTRY_STORAGE_KEY)

      return
    }

    // Drop the inline image bytes before persisting — a screenshot data URL is
    // megabytes and would blow the localStorage quota. On reload the record
    // falls back to reading its `path`/`url`.
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

function createOpaqueSessionPreviewRecordId(): string {
  const uuid = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

  return `session-preview:tab-${uuid}`
}

function sessionRegistryKey(sessionId: string): string {
  return previewWorkspaceScopeId(sessionId)
}

export function registerSessionPreview(
  sessionId: string | null | undefined,
  target: PreviewTarget,
  source: PreviewRecordSource,
  rawTarget = target.source
): SessionPreviewRecord | null {
  const id = sessionId?.trim()

  if (!id || (target.kind === 'url' && !isPersistableWebPreviewTarget(target))) {
    return null
  }

  const current = $sessionPreviewRegistry.get()
  const scopeId = sessionRegistryKey(id)
  const now = Date.now()
  const records = current[scopeId] ?? []
  const existing = records.find(record => record.normalized.url === target.url)
  const normalized = previewTargetForSource(target, source)

  const nextRecord: SessionPreviewRecord = {
    autoOpen: true,
    createdAt: now,
    id: existing?.id || createOpaqueSessionPreviewRecordId(),
    normalized,
    sessionId: id,
    source,
    target: rawTarget || target.source
  }

  $sessionPreviewRegistry.set(
    pruneRegistry({
      ...current,
      [scopeId]: [nextRecord]
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
  const normalized = record?.normalized ?? previewTargetForSource(target, source)

  if (normalized.kind === 'url') {
    openWebPreviewTarget(normalized)
  } else {
    setPreviewTarget(normalized)
  }

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

  const records = $sessionPreviewRegistry.get()[sessionRegistryKey(id)]

  return records?.find(record => !record.dismissedAt && record.autoOpen !== false) ?? null
}

export function previewSurfacePlacementForTab(tabId: RightRailTabId): PreviewSurfacePlacement {
  return $previewSurfaceLayouts.get()[tabId]?.placement ?? 'docked'
}

function removeRightRailTabLayout(tabId: RightRailTabId) {
  const next = { ...$previewSurfaceLayouts.get() }
  delete next[tabId]
  $previewSurfaceLayouts.set(next)
}

function previewTabIds(): RightRailTabId[] {
  const ids: RightRailTabId[] = []

  if ($previewTarget.get()) {
    ids.push(RIGHT_RAIL_PREVIEW_TAB_ID)
  }

  ids.push(...$webPreviewTabs.get().map(tab => tab.id))
  ids.push(...$filePreviewTabs.get().map(tab => tab.id))
  ids.push(...$utilityPreviewTabs.get().map(tab => tab.id))

  return ids
}

function activateAvailableTab(excluding?: RightRailTabId, preserveExcluded = false) {
  const candidates = previewTabIds().filter(id => id !== excluding)
  const next = candidates.find(id => previewSurfacePlacementForTab(id) !== 'minimized') ?? candidates[0]

  if (next) {
    selectRightRailTab(next)
  } else if (!preserveExcluded) {
    selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

export function setRightRailTabPlacement(tabId: RightRailTabId, placement: PreviewSurfacePlacement) {
  const current = $previewSurfaceLayouts.get()[tabId]

  if (placement === 'docked') {
    removeRightRailTabLayout(tabId)
  } else {
    $previewSurfaceLayouts.set({
      ...$previewSurfaceLayouts.get(),
      [tabId]: { ...current, placement }
    })
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(tabId)
}

export function setRightRailTabFloatingGeometry(
  tabId: RightRailTabId,
  geometry: FloatingGeometry,
  viewport: PreviewViewport
) {
  const current = $previewSurfaceLayouts.get()[tabId]
  const clamped = clampFloatingGeometry(geometry, viewport)

  $previewSurfaceLayouts.set({
    ...$previewSurfaceLayouts.get(),
    [tabId]: { ...current, geometry: clamped, placement: current?.placement ?? 'floating' }
  })
}

export function dockRightRailTab(tabId: RightRailTabId) {
  setRightRailTabPlacement(tabId, 'docked')
}

export function detachRightRailTab(tabId: RightRailTabId) {
  setRightRailTabPlacement(tabId, 'floating')
}

function restorableLayout(layout: PreviewSurfaceLayout | undefined): PreviewSurfaceRestore {
  if (layout?.placement === 'minimized' || layout?.placement === 'maximized') {
    return layout.restore ?? { geometry: layout.geometry, placement: 'floating' }
  }

  return {
    geometry: layout?.geometry,
    placement: layout?.placement ?? 'docked'
  }
}

export function maximizeRightRailTab(tabId: RightRailTabId) {
  const current = $previewSurfaceLayouts.get()[tabId]

  $previewSurfaceLayouts.set({
    ...$previewSurfaceLayouts.get(),
    [tabId]: {
      geometry: current?.geometry,
      placement: 'maximized',
      restore: restorableLayout(current)
    }
  })
  selectRightRailTab(tabId)
}

export function minimizeRightRailTab(tabId: RightRailTabId) {
  const current = $previewSurfaceLayouts.get()[tabId]

  $previewSurfaceLayouts.set({
    ...$previewSurfaceLayouts.get(),
    [tabId]: {
      geometry: current?.geometry,
      placement: 'minimized',
      restore: restorableLayout(current)
    }
  })

  if ($rightRailActiveTabId.get() === tabId) {
    activateAvailableTab(tabId, true)
  }
}

export function restoreRightRailTab(tabId: RightRailTabId) {
  const current = $previewSurfaceLayouts.get()[tabId]
  const restore = current?.restore

  if (!restore || restore.placement === 'docked') {
    removeRightRailTabLayout(tabId)
  } else {
    $previewSurfaceLayouts.set({
      ...$previewSurfaceLayouts.get(),
      [tabId]: { geometry: restore.geometry ?? current?.geometry, placement: restore.placement }
    })
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(tabId)
}

export function snapRightRailTab(tabId: RightRailTabId, placement: PreviewSnapSlot) {
  const current = $previewSurfaceLayouts.get()[tabId]

  $previewSurfaceLayouts.set({
    ...$previewSurfaceLayouts.get(),
    [tabId]: {
      geometry: current?.geometry,
      placement,
      restore: restorableLayout(current)
    }
  })
  selectRightRailTab(tabId)
}

export function dismissSessionPreview(sessionId: string | null | undefined, url?: string) {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const current = $sessionPreviewRegistry.get()
  const scopeId = sessionRegistryKey(id)
  const records = current[scopeId]

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
    [scopeId]: dismissedRecords
  })
}

function dismissSessionPreviewsForUrl(url: string) {
  const current = $sessionPreviewRegistry.get()
  const now = Date.now()
  let changed = false
  const next: SessionPreviewRegistry = {}

  for (const [sessionId, records] of Object.entries(current)) {
    next[sessionId] = records.map(record => {
      if (record.normalized.url !== url || record.dismissedAt || record.autoOpen === false) {
        return record
      }

      changed = true

      return { ...record, autoOpen: false, dismissedAt: now }
    })
  }

  if (changed) {
    $sessionPreviewRegistry.set(next)
  }
}

/** User clicked the close X — clear the target and persist dismissal for the current session. */
export function dismissPreviewTarget() {
  const current = $previewTarget.get()

  if (current?.url) {
    dismissSessionPreview(currentPreviewSessionId(), current.url)
  }

  $previewTarget.set(null)
  removeRightRailTabLayout(RIGHT_RAIL_PREVIEW_TAB_ID)

  if ($rightRailActiveTabId.get() === RIGHT_RAIL_PREVIEW_TAB_ID) {
    activateAvailableTab(RIGHT_RAIL_PREVIEW_TAB_ID)
  }

  setPaneOpen(PREVIEW_PANE_ID, previewTabIds().length > 0)
}

function closeWebPreviewTab(tabId: RightRailTabId) {
  if (!tabId.startsWith('preview:')) {
    return
  }

  const current = $webPreviewTabs.get()
  const closing = current.find(tab => tab.id === tabId)

  if (!closing) {
    return
  }

  dismissSessionPreviewsForUrl(closing.target.url)
  $webPreviewTabs.set(current.filter(tab => tab.id !== tabId))
  removeRightRailTabLayout(tabId)

  if ($rightRailActiveTabId.get() === tabId) {
    activateAvailableTab(tabId)
  }

  setPaneOpen(PREVIEW_PANE_ID, previewTabIds().length > 0)
}

function closeFilePreviewTab(tabId: RightRailTabId) {
  if (!tabId.startsWith('file:')) {
    return
  }

  const current = $filePreviewTabs.get()

  if (!current.some(tab => tab.id === tabId)) {
    return
  }

  $filePreviewTabs.set(current.filter(tab => tab.id !== tabId))
  removeRightRailTabLayout(tabId)

  if ($rightRailActiveTabId.get() === tabId) {
    activateAvailableTab(tabId)
  }

  setPaneOpen(PREVIEW_PANE_ID, previewTabIds().length > 0)
}

function closeUtilityPreviewTab(tabId: RightRailTabId) {
  if (!tabId.startsWith('utility:')) {
    return
  }

  const current = $utilityPreviewTabs.get()

  if (!current.some(tab => tab.id === tabId)) {
    return
  }

  $utilityPreviewTabs.set(current.filter(tab => tab.id !== tabId))
  removeRightRailTabLayout(tabId)

  if ($rightRailActiveTabId.get() === tabId) {
    activateAvailableTab(tabId)
  }

  setPaneOpen(PREVIEW_PANE_ID, previewTabIds().length > 0)
}

export function closeRightRailTab(tabId: RightRailTabId) {
  if (tabId === RIGHT_RAIL_PREVIEW_TAB_ID) {
    if ($previewTarget.get()) {
      dismissPreviewTarget()
    }

    return
  }

  if (tabId.startsWith('preview:')) {
    closeWebPreviewTab(tabId)
  } else if (tabId.startsWith('utility:')) {
    closeUtilityPreviewTab(tabId)
  } else {
    closeFilePreviewTab(tabId)
  }
}

export const closeActiveRightRailTab = () => closeRightRailTab($rightRailActiveTabId.get())

function rightRailTabOrder(): RightRailTabId[] {
  return previewTabIds()
}

/** Close every rail tab except `keepId`, then make `keepId` active. */
export function closeOtherRightRailTabs(keepId: RightRailTabId) {
  for (const id of rightRailTabOrder()) {
    if (id !== keepId) {
      closeRightRailTab(id)
    }
  }

  restoreRightRailTab(keepId)
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

/** Dismisses every preview surface so the rail pane unmounts. */
export function closeRightRail() {
  if ($previewTarget.get()) {
    dismissPreviewTarget()
  }

  $webPreviewTabs.set([])
  $filePreviewTabs.set([])
  $utilityPreviewTabs.set([])
  $previewSurfaceLayouts.set({})
  setPaneOpen(PREVIEW_PANE_ID, false)
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
}

export function clearSessionPreviewRegistry() {
  activePreviewWorkspaceScope = ''

  if (typeof window !== 'undefined') {
    window.localStorage.removeItem(WORKSPACE_SCOPE_STORAGE_KEY)
  }

  $sessionPreviewRegistry.set({})
  $previewTarget.set(null)
  $webPreviewTabs.set([])
  $filePreviewTabs.set([])
  $utilityPreviewTabs.set([])
  $previewSurfaceLayouts.set({})
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
