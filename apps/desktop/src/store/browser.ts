import { Codecs, persistentAtom } from '@/lib/persisted'

import { dropBrowserGuestState } from './browser-guest-state'
import {
  $rightRailActiveTabId,
  type BrowserRightRailTabId,
  PREVIEW_PANE_ID,
  RIGHT_RAIL_BROWSER_TAB_PREFIX,
  RIGHT_RAIL_PREVIEW_TAB_ID,
  selectRightRailTab
} from './layout'
import { setPaneOpen } from './panes'

export type BrowserTabId = BrowserRightRailTabId
export type BrowserControlMode = 'idle' | 'observe' | 'control' | 'paused'

export type BrowserConsoleLevel = 'debug' | 'error' | 'info' | 'log' | 'warn'
export type BrowserConsoleSource = 'console' | 'exception' | 'page' | 'agent'

export interface BrowserConsoleEntry {
  at: number
  level: BrowserConsoleLevel
  line?: number
  message: string
  source: BrowserConsoleSource
  sourceId?: string
  url?: string
}

export type BrowserNetworkEventType = 'load-error' | 'navigation' | 'request' | 'response'

export type BrowserActionStatus = 'error' | 'success'

export interface BrowserActionEvent {
  at: number
  command: string
  error?: string
  result?: string
  status: BrowserActionStatus
  target?: string
}

export interface BrowserScreenshotEntry {
  at: number
  dataUrl: string
  title?: string
  url?: string
}

export interface BrowserNetworkEvent {
  at: number
  error?: string
  method?: string
  status?: number
  type: BrowserNetworkEventType
  url: string
}

export type BrowserErrorKind = 'agent' | 'blocked-scheme' | 'certificate' | 'crash' | 'load' | 'timeout'

export interface BrowserErrorState {
  code?: number | string
  kind: BrowserErrorKind
  message: string
  title: string
  url?: string
}

export interface BrowserTabState {
  actionCount: number
  agentError?: string
  automationBound?: boolean
  browserError?: BrowserErrorState
  canGoBack: boolean
  canGoForward: boolean
  consoleCount: number
  controlMode: BrowserControlMode
  crashed?: boolean
  createdAt: number
  id: BrowserTabId
  loading: boolean
  networkCount: number
  originAllowlist?: string[]
  partition?: string
  profile?: string
  screenshotCount: number
  sessionId: string
  title: string
  updatedAt: number
  url: string
}

export interface CreateBrowserTabRequest {
  originAllowlist?: string[]
  partition?: string
  profile?: string
  sessionId: string
  title?: string
  url?: string
}

export interface OpenBrowserTabRequest extends Omit<CreateBrowserTabRequest, 'profile' | 'sessionId'> {
  profile?: string | null
  sessionId?: string | null
}

export const BROWSER_TAB_LIMIT = 8
export const BROWSER_OBSERVABILITY_LIMIT = 200
export const BROWSER_SCREENSHOT_HISTORY_LIMIT = 20
const BROWSER_OBSERVABILITY_FIELD_LIMIT = 4096
const BROWSER_OBSERVABILITY_METHOD_LIMIT = 128
export const BROWSER_TABS_STORAGE_KEY = 'hermes.desktop.browserTabs.v1'
export const CURSOR_BROWSER_ENABLED_STORAGE_KEY = 'hermes.desktop.browser.enabled'

const browserActionEvents = new Map<BrowserTabId, BrowserActionEvent[]>()
const browserConsoleEntries = new Map<BrowserTabId, BrowserConsoleEntry[]>()
const browserNetworkEvents = new Map<BrowserTabId, BrowserNetworkEvent[]>()
const browserScreenshotEntries = new Map<BrowserTabId, BrowserScreenshotEntry[]>()

export const $browserEnabled = persistentAtom(CURSOR_BROWSER_ENABLED_STORAGE_KEY, false, Codecs.bool)

export const $browserTabs = persistentAtom<BrowserTabState[]>(BROWSER_TABS_STORAGE_KEY, [], {
  decode: raw => sanitizeBrowserTabs(JSON.parse(raw) as unknown, { resetControlConsent: true }),
  encode: tabs => {
    const sanitized = sanitizeBrowserTabs(tabs)

    return sanitized.length === 0 ? null : JSON.stringify(sanitized)
  }
})

export function setBrowserEnabled(enabled: boolean): void {
  $browserEnabled.set(enabled)
}

export function toggleBrowserEnabled(): boolean {
  const next = !$browserEnabled.get()
  setBrowserEnabled(next)

  return next
}

export function isBrowserTabId(value: unknown): value is BrowserTabId {
  return typeof value === 'string' && value.startsWith(RIGHT_RAIL_BROWSER_TAB_PREFIX) && value.length > RIGHT_RAIL_BROWSER_TAB_PREFIX.length
}

export function createBrowserTab(request: CreateBrowserTabRequest): BrowserTabState {
  const now = Date.now()
  const url = request.url?.trim() || 'about:blank'
  const profile = cleanOptionalString(request.profile)
  const sessionId = request.sessionId?.trim() || 'desktop'

  const tab: BrowserTabState = {
    actionCount: 0,
    canGoBack: false,
    canGoForward: false,
    consoleCount: 0,
    controlMode: 'idle',
    createdAt: now,
    id: newBrowserTabId(),
    loading: false,
    networkCount: 0,
    originAllowlist: cleanStringArray(request.originAllowlist),
    partition: cleanBrowserPartition(request.partition, profile, sessionId),
    profile,
    screenshotCount: 0,
    sessionId,
    title: request.title?.trim() || titleForUrl(url),
    updatedAt: now,
    url
  }

  $browserTabs.set(capBrowserTabs([...$browserTabs.get(), tab]))
  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(tab.id)

  return tab
}

export function openBrowserTab(request: OpenBrowserTabRequest = {}): BrowserTabState | null {
  if (!$browserEnabled.get()) {
    return null
  }

  return createBrowserTabFromOpenRequest(request)
}

export function enableBrowserAndOpenTab(request: OpenBrowserTabRequest = {}): BrowserTabState {
  setBrowserEnabled(true)

  return createBrowserTabFromOpenRequest(request)
}

function createBrowserTabFromOpenRequest(request: OpenBrowserTabRequest): BrowserTabState {
  return createBrowserTab({
    profile: request.profile ?? undefined,
    sessionId: request.sessionId?.trim() || 'desktop',
    title: request.title?.trim() || 'Browser',
    url: request.url?.trim() || 'about:blank'
  })
}

export function updateBrowserTab(tabId: BrowserTabId, patch: Partial<BrowserTabState>): BrowserTabState | null {
  let updated: BrowserTabState | null = null
  const hasAgentError = Object.prototype.hasOwnProperty.call(patch, 'agentError')
  const hasBrowserError = Object.prototype.hasOwnProperty.call(patch, 'browserError')

  $browserTabs.set(
    $browserTabs.get().map(tab => {
      if (tab.id !== tabId) {
        return tab
      }

      const nextTab = sanitizeBrowserTab({
        ...tab,
        actionCount: patch.actionCount ?? tab.actionCount,
        agentError: hasAgentError ? cleanOptionalString(patch.agentError) : tab.agentError,
        automationBound: patch.automationBound ?? tab.automationBound,
        browserError: hasBrowserError ? sanitizeBrowserError(patch.browserError) : tab.browserError,
        canGoBack: patch.canGoBack ?? tab.canGoBack,
        canGoForward: patch.canGoForward ?? tab.canGoForward,
        consoleCount: patch.consoleCount ?? tab.consoleCount,
        controlMode: patch.controlMode ?? tab.controlMode,
        crashed: patch.crashed ?? tab.crashed,
        loading: patch.loading ?? tab.loading,
        networkCount: patch.networkCount ?? tab.networkCount,
        originAllowlist: patch.originAllowlist ?? tab.originAllowlist,
        partition: tab.partition,
        profile: tab.profile,
        screenshotCount: patch.screenshotCount ?? tab.screenshotCount,
        sessionId: tab.sessionId,
        title: patch.title ?? tab.title,
        updatedAt: Date.now(),
        url: patch.url ?? tab.url
      })

      updated = nextTab ?? tab

      return updated
    })
  )

  return updated
}

export function moveBrowserTab(tabId: BrowserTabId, direction: 'left' | 'right'): void {
  const tabs = $browserTabs.get()
  const index = tabs.findIndex(tab => tab.id === tabId)

  if (index === -1) {
    return
  }

  const targetIndex = direction === 'left' ? index - 1 : index + 1

  if (targetIndex < 0 || targetIndex >= tabs.length) {
    return
  }

  const next = [...tabs]
  const [tab] = next.splice(index, 1)

  if (!tab) {
    return
  }

  next.splice(targetIndex, 0, tab)
  $browserTabs.set(next)
}

export function closeBrowserTab(tabId: BrowserTabId): void {
  const current = $browserTabs.get()
  const index = current.findIndex(tab => tab.id === tabId)

  if (index === -1) {
    return
  }

  const next = current.filter(tab => tab.id !== tabId)
  dropBrowserObservability(tabId)
  dropBrowserGuestState(tabId)
  $browserTabs.set(next)

  if (next.length === 0) {
    if ($rightRailActiveBrowserTabId() === tabId) {
      selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
    }

    return
  }

  if ($rightRailActiveBrowserTabId() === tabId) {
    selectRightRailTab(next[Math.min(index, next.length - 1)]?.id ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }
}

export function clearBrowserTabs(): void {
  browserActionEvents.clear()
  browserConsoleEntries.clear()
  browserNetworkEvents.clear()
  browserScreenshotEntries.clear()
  $browserTabs.set([])
  selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
  setPaneOpen(PREVIEW_PANE_ID, false)
}

export function appendBrowserConsoleEntry(
  tabId: BrowserTabId,
  entry: Omit<BrowserConsoleEntry, 'at'> & { at?: number }
): BrowserConsoleEntry {
  const nextEntry: BrowserConsoleEntry = {
    at: typeof entry.at === 'number' ? entry.at : Date.now(),
    level: sanitizeConsoleLevel(entry.level),
    line: typeof entry.line === 'number' ? entry.line : undefined,
    message: capObservabilityString(entry.message, BROWSER_OBSERVABILITY_FIELD_LIMIT) ?? '',
    source: sanitizeConsoleSource(entry.source),
    sourceId: capObservabilityString(entry.sourceId, BROWSER_OBSERVABILITY_FIELD_LIMIT),
    url: capObservabilityString(entry.url, BROWSER_OBSERVABILITY_FIELD_LIMIT)
  }

  const next = capObservabilityEntries([...(browserConsoleEntries.get(tabId) ?? []), nextEntry])
  browserConsoleEntries.set(tabId, next)
  updateBrowserTab(tabId, { consoleCount: next.length })

  return nextEntry
}

export function getBrowserConsoleEntries(tabId: BrowserTabId): BrowserConsoleEntry[] {
  return [...(browserConsoleEntries.get(tabId) ?? [])]
}

export function clearBrowserConsoleEntries(tabId: BrowserTabId): void {
  browserConsoleEntries.delete(tabId)
  updateBrowserTab(tabId, { consoleCount: 0 })
}

export function appendBrowserNetworkEvent(
  tabId: BrowserTabId,
  event: Omit<BrowserNetworkEvent, 'at'> & { at?: number }
): BrowserNetworkEvent {
  const nextEvent: BrowserNetworkEvent = {
    at: typeof event.at === 'number' ? event.at : Date.now(),
    error: capObservabilityString(event.error, BROWSER_OBSERVABILITY_FIELD_LIMIT),
    method: capObservabilityString(event.method, BROWSER_OBSERVABILITY_METHOD_LIMIT),
    status: typeof event.status === 'number' ? event.status : undefined,
    type: sanitizeNetworkEventType(event.type),
    url: capObservabilityString(event.url, BROWSER_OBSERVABILITY_FIELD_LIMIT) ?? ''
  }

  const next = capObservabilityEntries([...(browserNetworkEvents.get(tabId) ?? []), nextEvent])
  browserNetworkEvents.set(tabId, next)
  updateBrowserTab(tabId, { networkCount: next.length })

  return nextEvent
}

export function getBrowserNetworkEvents(tabId: BrowserTabId): BrowserNetworkEvent[] {
  return [...(browserNetworkEvents.get(tabId) ?? [])]
}

export function clearBrowserNetworkEvents(tabId: BrowserTabId): void {
  browserNetworkEvents.delete(tabId)
  updateBrowserTab(tabId, { networkCount: 0 })
}

export function appendBrowserActionEvent(
  tabId: BrowserTabId,
  event: Omit<BrowserActionEvent, 'at'> & { at?: number }
): BrowserActionEvent {
  const nextEvent: BrowserActionEvent = {
    at: typeof event.at === 'number' ? event.at : Date.now(),
    command: capObservabilityString(event.command, BROWSER_OBSERVABILITY_METHOD_LIMIT) ?? 'unknown',
    error: capObservabilityString(event.error, BROWSER_OBSERVABILITY_FIELD_LIMIT),
    result: capObservabilityString(event.result, BROWSER_OBSERVABILITY_FIELD_LIMIT),
    status: event.status === 'error' ? 'error' : 'success',
    target: capObservabilityString(event.target, BROWSER_OBSERVABILITY_FIELD_LIMIT)
  }

  const next = capObservabilityEntries([...(browserActionEvents.get(tabId) ?? []), nextEvent])

  browserActionEvents.set(tabId, next)
  updateBrowserTab(tabId, { actionCount: next.length })

  return nextEvent
}

export function getBrowserActionEvents(tabId: BrowserTabId): BrowserActionEvent[] {
  return [...(browserActionEvents.get(tabId) ?? [])]
}

export function clearBrowserActionEvents(tabId: BrowserTabId): void {
  browserActionEvents.delete(tabId)
  updateBrowserTab(tabId, { actionCount: 0 })
}

export function appendBrowserScreenshotEntry(
  tabId: BrowserTabId,
  entry: Omit<BrowserScreenshotEntry, 'at'> & { at?: number }
): BrowserScreenshotEntry {
  const nextEntry: BrowserScreenshotEntry = {
    at: typeof entry.at === 'number' ? entry.at : Date.now(),
    dataUrl: capObservabilityString(entry.dataUrl, BROWSER_OBSERVABILITY_FIELD_LIMIT * 4) ?? '',
    title: capObservabilityString(entry.title, BROWSER_OBSERVABILITY_FIELD_LIMIT),
    url: capObservabilityString(entry.url, BROWSER_OBSERVABILITY_FIELD_LIMIT)
  }

  const next = [...(browserScreenshotEntries.get(tabId) ?? []), nextEntry].slice(-BROWSER_SCREENSHOT_HISTORY_LIMIT)

  browserScreenshotEntries.set(tabId, next)
  updateBrowserTab(tabId, { screenshotCount: next.length })

  return nextEntry
}

export function getBrowserScreenshotEntries(tabId: BrowserTabId): BrowserScreenshotEntry[] {
  return [...(browserScreenshotEntries.get(tabId) ?? [])]
}

export function clearBrowserScreenshotEntries(tabId: BrowserTabId): void {
  browserScreenshotEntries.delete(tabId)
  updateBrowserTab(tabId, { screenshotCount: 0 })
}

function $rightRailActiveBrowserTabId(): BrowserTabId | null {
  const current = $rightRailActiveTabId.get()

  return isBrowserTabId(current) ? current : null
}

function capBrowserTabs(tabs: BrowserTabState[]): BrowserTabState[] {
  return tabs.slice(-BROWSER_TAB_LIMIT)
}

function newBrowserTabId(): BrowserTabId {
  const random = globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

  return `${RIGHT_RAIL_BROWSER_TAB_PREFIX}${random}`
}

interface SanitizeBrowserTabsOptions {
  resetControlConsent?: boolean
}

function sanitizeBrowserTabs(value: unknown, options: SanitizeBrowserTabsOptions = {}): BrowserTabState[] {
  if (!Array.isArray(value)) {
    return []
  }

  return capBrowserTabs(value.map(tab => sanitizeBrowserTab(tab, options)).filter((tab): tab is BrowserTabState => Boolean(tab)))
}

function sanitizeBrowserTab(value: unknown, options: SanitizeBrowserTabsOptions = {}): BrowserTabState | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as Record<string, unknown>

  if (!isBrowserTabId(raw.id) || typeof raw.sessionId !== 'string' || typeof raw.url !== 'string') {
    return null
  }

  const now = Date.now()
  const url = raw.url.trim() || 'about:blank'
  const profile = cleanOptionalString(raw.profile)
  const sessionId = raw.sessionId.trim() || 'desktop'

  return {
    actionCount: nonNegativeInteger(raw.actionCount),
    agentError: cleanOptionalString(raw.agentError),
    automationBound: typeof raw.automationBound === 'boolean' ? raw.automationBound : undefined,
    browserError: sanitizeBrowserError(raw.browserError),
    canGoBack: raw.canGoBack === true,
    canGoForward: raw.canGoForward === true,
    consoleCount: nonNegativeInteger(raw.consoleCount),
    controlMode: sanitizeControlMode(raw.controlMode, options),
    crashed: typeof raw.crashed === 'boolean' ? raw.crashed : undefined,
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : now,
    id: raw.id,
    loading: raw.loading === true,
    networkCount: nonNegativeInteger(raw.networkCount),
    originAllowlist: cleanStringArray(Array.isArray(raw.originAllowlist) ? raw.originAllowlist : undefined),
    partition: cleanBrowserPartition(raw.partition, profile, sessionId),
    profile,
    screenshotCount: nonNegativeInteger(raw.screenshotCount),
    sessionId,
    title: typeof raw.title === 'string' && raw.title.trim() ? raw.title.trim() : titleForUrl(url),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : now,
    url
  }
}

function sanitizeControlMode(value: unknown, options: SanitizeBrowserTabsOptions = {}): BrowserControlMode {
  if (options.resetControlConsent && (value === 'observe' || value === 'control')) {
    return 'idle'
  }

  return value === 'observe' || value === 'control' || value === 'paused' ? value : 'idle'
}

function sanitizeBrowserError(value: unknown): BrowserErrorState | undefined {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const raw = value as Record<string, unknown>
  const title = cleanOptionalString(raw.title)
  const message = cleanOptionalString(raw.message)
  const kind = raw.kind

  if (!title || !message || !isBrowserErrorKind(kind)) {
    return undefined
  }

  return {
    code: typeof raw.code === 'string' || typeof raw.code === 'number' ? raw.code : undefined,
    kind,
    message,
    title,
    url: cleanOptionalString(raw.url)
  }
}

function isBrowserErrorKind(value: unknown): value is BrowserErrorKind {
  return value === 'agent' || value === 'blocked-scheme' || value === 'certificate' || value === 'crash' || value === 'load' || value === 'timeout'
}

function nonNegativeInteger(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : 0
}

function cleanOptionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

export function browserPartitionForScope(profile: string | undefined, sessionId: string | undefined): string {
  return `persist:hermes-browser:${partitionToken(profile || 'default')}:${partitionToken(sessionId || 'desktop')}`
}

export function browserPartitionForTab(tab: Pick<BrowserTabState, 'profile' | 'sessionId'>): string {
  return browserPartitionForScope(tab.profile, tab.sessionId)
}

function cleanBrowserPartition(value: unknown, profile: string | undefined, sessionId: string | undefined): string {
  const canonical = browserPartitionForScope(profile, sessionId)
  const cleaned = cleanOptionalString(value)

  return cleaned === canonical ? cleaned : canonical
}

function partitionToken(value: string): string {
  const token = value.trim().replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 96)

  return token || 'default'
}

function cleanStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined
  }

  const cleaned = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0).map(item => item.trim())

  return cleaned.length > 0 ? [...new Set(cleaned)] : undefined
}

function dropBrowserObservability(tabId: BrowserTabId): void {
  browserActionEvents.delete(tabId)
  browserConsoleEntries.delete(tabId)
  browserNetworkEvents.delete(tabId)
  browserScreenshotEntries.delete(tabId)
}

function capObservabilityEntries<T>(entries: T[]): T[] {
  return entries.slice(-BROWSER_OBSERVABILITY_LIMIT)
}

function capObservabilityString(value: unknown, limit: number): string | undefined {
  if (typeof value !== 'string') {
    return undefined
  }

  const cleaned = value.trim()

  if (!cleaned) {
    return undefined
  }

  if (cleaned.length <= limit) {
    return cleaned
  }

  return `${cleaned.slice(0, Math.max(0, limit - 32))}… [truncated ${cleaned.length - limit} chars]`
}

function sanitizeConsoleLevel(value: unknown): BrowserConsoleLevel {
  return value === 'debug' || value === 'error' || value === 'info' || value === 'warn' ? value : 'log'
}

function sanitizeConsoleSource(value: unknown): BrowserConsoleSource {
  return value === 'exception' || value === 'page' || value === 'agent' ? value : 'console'
}

function sanitizeNetworkEventType(value: unknown): BrowserNetworkEventType {
  return value === 'load-error' || value === 'request' || value === 'response' ? value : 'navigation'
}

function titleForUrl(url: string): string {
  try {
    const parsed = new URL(url)

    return parsed.hostname || parsed.href
  } catch {
    return url
  }
}
