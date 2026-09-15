/**
 * Hermes Workstation Browser Runtime
 *
 * First-class Chromium runtime hosted inside Hermes Desktop. Electron already
 * ships Chromium, so Workstation does not require Chrome/Edge as a product
 * dependency.
 *
 * The runtime deliberately has two control surfaces over ONE WebContentsView
 * pool:
 *   1. Electron IPC for the Browser page the human sees.
 *   2. A loopback-only bearer-authenticated controller for Hermes browser_*
 *      tools and Kanban workers. The controller file is user-local and never
 *      exposed on LAN.
 *
 * Browser tabs survive route changes and continue in the background. Task ids
 * are bound to tabs so a background worker never steals the user's active tab.
 * Once a task owns an internal tab, recovery is expected to reconnect to this
 * runtime rather than silently moving the task to a different browser.
 *
 * The attach/detach + lifecycle design is adapted from browser-use/desktop's
 * BrowserPool (MIT). See workstation/THIRD_PARTY_NOTICES.md.
 */

import crypto from 'node:crypto'
import fs from 'node:fs'
import http, { type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import os from 'node:os'
import path from 'node:path'

import { app, BrowserWindow, ipcMain, session, type Session, type WebContents, WebContentsView } from 'electron'

import {
  BrowserSessionStateFilePersistence,
  type BrowserSessionStateSnapshot,
  type BrowserSessionTab,
  type BrowserSessionTabRecoveryReason,
  type BrowserSessionTabRecoveryState,
  safeRestorableUrlMetadata,
  safeTitleMetadata
} from './workstation-browser-session-state'
import { type BrowserTask, BrowserTaskLifecycle, type BrowserTaskSeed } from './workstation-browser-task'
import {
  buildWorkstationResourceSnapshot,
  WORKSTATION_EVENT_SCHEMA_VERSION,
  type WorkstationEventSnapshot,
  type WorkstationResourceSnapshot
} from './workstation-browser-resources'

const CACHE_CHECK_INTERVAL_MS = 30 * 60 * 1000
const DEFAULT_CACHE_MAX_MB = 512
const DEFAULT_BACKGROUND_FRAME_RATE = 6
const DEFAULT_VISIBLE_FRAME_RATE = 60
const DEFAULT_BROWSER_WIDTH = 1280
const DEFAULT_BROWSER_HEIGHT = 800
const CONTROL_FILE_VERSION = 1
const MAX_CONTROL_BODY_BYTES = 512 * 1024
const MAX_CONTROLLER_SESSION_ID_CHARS = 256
const COMPACT_TEXT_CHARS = 8_000
const FULL_TEXT_CHARS = 24_000
const COMPACT_ELEMENTS = 120
const FULL_ELEMENTS = 400
const MAX_EVENTS_ENDPOINT = 200

export function getStandardChromeUserAgent(): string {
  const plat = process.platform

  if (plat === 'darwin') {
    return 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
  }

  if (plat === 'linux') {
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
  }

  return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

export type WorkstationBrowserControlOwner = 'agent' | 'human'

export interface WorkstationBrowserBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface WorkstationBrowserTabState {
  id: string
  title: string
  url: string
  active: boolean
  loading: boolean
  canGoBack: boolean
  canGoForward: boolean
  crashed: boolean
  ownerTaskId: string | null
}

export interface WorkstationDownloadItem {
  id: string
  filename: string
  savePath: string
  totalBytes: number
  receivedBytes: number
  state: 'progressing' | 'completed' | 'cancelled' | 'interrupted'
  url: string
}

export interface WorkstationBrowserState {
  runtime: 'electron-chromium'
  ready: boolean
  attached: boolean
  viewportHost: 'hub' | 'chat' | string | null
  backgroundCapable: true
  paused: boolean
  controlOwner: WorkstationBrowserControlOwner
  controlReady: boolean
  profilePath: string
  cacheBytes: number | null
  activeTabId: string | null
  tabs: WorkstationBrowserTabState[]
  tasks: BrowserTask[]
  downloads: WorkstationDownloadItem[]
  lastError: string | null
}

interface BrowserEntry {
  id: string
  view: WebContentsView
  loading: boolean
  crashed: boolean
  ownerTaskId: string | null
  safeUrl: string | null
  safeTitle: string | null
  recoveryState: BrowserSessionTabRecoveryState
  recoveryReason: BrowserSessionTabRecoveryReason
}

export interface WorkstationControllerError {
  error_code:
    | 'STALE_REF'
    | 'NO_BOUND_TAB'
    | 'AUTH_REQUIRED'
    | 'USER_CONTROL_ACTIVE'
    | 'CAPABILITY_MISSING'
    | 'TIMEOUT'
    | 'INVALID_ARGUMENT'
    | 'CONTROLLER_DOWN'
  message: string
  retryable: boolean
  retry_after_ms: number
  state_changed: boolean
  recommended_action: string
  resource_ref?: string | undefined
  details: Record<string, unknown>
}

/** Map controller failures to remediation data without changing task ownership. */
export function normalizeWorkstationControllerError(error: unknown, resourceRef?: string): WorkstationControllerError {
  const message = error instanceof Error ? error.message : String(error)
  const lower = message.toLowerCase()
  const base = {
    message,
    retry_after_ms: 0,
    resource_ref: resourceRef,
    details: {}
  }

  if (lower.includes('human control')) {
    return { ...base, error_code: 'USER_CONTROL_ACTIVE', retryable: true, state_changed: true, recommended_action: 'WAIT_FOR_RELEASE' }
  }
  if (lower.includes('no_bound_browser_tab') || lower.includes('no active tab')) {
    return { ...base, error_code: 'NO_BOUND_TAB', retryable: true, state_changed: true, recommended_action: 'BIND_OR_NAVIGATE' }
  }
  if (lower.includes('ref_required') || lower.includes('element_unavailable') || lower.includes('browser_tab_destroyed')) {
    return { ...base, error_code: 'STALE_REF', retryable: true, state_changed: true, recommended_action: 'RESNAPSHOT' }
  }
  if (lower.includes('timed out') || lower.includes('timeout')) {
    return { ...base, error_code: 'TIMEOUT', retryable: true, state_changed: false, recommended_action: 'RETRY_WITH_BACKOFF' }
  }
  if (lower.includes('unavailable') || lower.includes('could not bind')) {
    return { ...base, error_code: 'CONTROLLER_DOWN', retryable: true, state_changed: true, recommended_action: 'RECONCILE_CONTROLLER' }
  }
  if (lower.includes('unsupported') || lower.includes('invalid') || lower.includes('required')) {
    return { ...base, error_code: 'INVALID_ARGUMENT', retryable: false, state_changed: false, recommended_action: 'CORRECT_REQUEST' }
  }
  return { ...base, error_code: 'CAPABILITY_MISSING', retryable: false, state_changed: false, recommended_action: 'ESCALATE' }
}

interface ControlHandle {
  server: Server
  url: string
  token: string
  controlPath: string
}

interface BrowserControlRequest {
  action?: unknown
  arguments?: unknown
  task_id?: unknown
  session_id?: unknown
  kanban_card_id?: unknown
  card_id?: unknown
  run_id?: unknown
}

interface PageInventory {
  url: string
  title: string
  text: string
  totalTextChars: number
  truncated: boolean
  elements: Array<{
    ref: string
    tag: string
    role: string
    label: string
    value?: string
    disabled: boolean
  }>
}

interface BrowserTaskShowContext {
  window: BrowserWindow
  bounds: WorkstationBrowserBounds
  host?: string
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function cacheLimitBytes(): number {
  const raw = Number(process.env.HERMES_WORKSTATION_BROWSER_CACHE_MAX_MB ?? DEFAULT_CACHE_MAX_MB)
  const mb = Number.isFinite(raw) && raw > 32 ? raw : DEFAULT_CACHE_MAX_MB

  return Math.round(mb * 1024 * 1024)
}

function backgroundFrameRate(): number {
  const raw = Number(process.env.HERMES_WORKSTATION_BROWSER_BACKGROUND_FPS ?? DEFAULT_BACKGROUND_FRAME_RATE)

  if (!Number.isFinite(raw)) {
    return DEFAULT_BACKGROUND_FRAME_RATE
  }

  return Math.max(1, Math.min(30, Math.round(raw)))
}

function workstationBasePath(): string {
  if (process.env.HERMES_WORKSTATION_HOME?.trim()) {
    return path.resolve(process.env.HERMES_WORKSTATION_HOME.trim())
  }

  if (process.platform === 'win32') {
    const base = process.env.LOCALAPPDATA?.trim() || path.dirname(app.getPath('userData'))

    return path.join(base, 'HermesWorkstation')
  }

  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', 'HermesWorkstation')
  }

  const base = process.env.XDG_CONFIG_HOME?.trim() || path.join(os.homedir(), '.config')

  return path.join(base, 'HermesWorkstation')
}

export function workstationBrowserProfilePath(): string {
  if (process.env.HERMES_WORKSTATION_BROWSER_PROFILE?.trim()) {
    return path.resolve(process.env.HERMES_WORKSTATION_BROWSER_PROFILE.trim())
  }

  return path.join(workstationBasePath(), 'Browser', 'User Data')
}

export function workstationBrowserControlPath(): string {
  if (process.env.HERMES_WORKSTATION_BROWSER_CONTROL_FILE?.trim()) {
    return path.resolve(process.env.HERMES_WORKSTATION_BROWSER_CONTROL_FILE.trim())
  }

  return path.join(workstationBasePath(), 'Runtime', 'browser-control.json')
}

export function workstationBrowserTaskStatePath(): string {
  if (process.env.HERMES_WORKSTATION_BROWSER_TASK_FILE?.trim()) {
    return path.resolve(process.env.HERMES_WORKSTATION_BROWSER_TASK_FILE.trim())
  }

  return path.join(workstationBasePath(), 'Runtime', 'browser-tasks.json')
}

export function workstationBrowserSessionStatePath(): string {
  return path.join(workstationBasePath(), 'Runtime', 'browser-session.json')
}

function screenshotDirectory(): string {
  return path.join(workstationBasePath(), 'Browser', 'Screenshots')
}

function historyState(wc: WebContents): { canGoBack: boolean; canGoForward: boolean } {
  try {
    return {
      canGoBack: wc.navigationHistory.canGoBack(),
      canGoForward: wc.navigationHistory.canGoForward()
    }
  } catch {
    return { canGoBack: false, canGoForward: false }
  }
}

export function normalizeWorkstationBrowserTarget(value: string): string {
  const raw = String(value ?? '').trim()

  if (!raw) {
    return 'about:blank'
  }

  if (raw === 'about:blank') {
    return raw
  }

  try {
    const parsed = new URL(raw)

    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.toString()
    }
  } catch {
    // Fall through to hostname/search heuristics.
  }

  const localish = /^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/.*)?$/i.test(raw)

  if (localish) {
    return `http://${raw}`
  }

  const hostish = /^[a-z0-9.-]+\.[a-z]{2,}(?::\d+)?(?:\/.*)?$/i.test(raw)

  if (hostish) {
    return `https://${raw}`
  }

  return `https://duckduckgo.com/?q=${encodeURIComponent(raw)}`
}

function blockedSensitiveNetworkUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    const host = parsed.hostname.replace(/^\[|\]$/g, '').toLowerCase()

    if (host === 'metadata.google.internal' || host === 'metadata.goog' || host === '100.100.100.200') {
      return true
    }

    if (host.startsWith('169.254.')) {
      return true
    }

    if (host === 'fd00:ec2::254') {
      return true
    }

    if (host.startsWith('::ffff:169.254.')) {
      return true
    }

    if (host === '::ffff:100.100.100.200') {
      return true
    }

    return false
  } catch {
    return false
  }
}

function permittedTopLevelUrl(url: string): boolean {
  if (url === 'about:blank') {
    return true
  }

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return false
  }

  return !blockedSensitiveNetworkUrl(url)
}

function sendJson(res: ServerResponse, status: number, body: Record<string, unknown>): void {
  const raw = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(raw),
    'cache-control': 'no-store'
  })
  res.end(raw)
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = ''
    let bytes = 0
    req.setEncoding('utf-8')
    req.on('data', (chunk: string) => {
      bytes += Buffer.byteLength(chunk)

      if (bytes > MAX_CONTROL_BODY_BYTES) {
        reject(new Error('request body too large'))
        req.destroy()

        return
      }

      body += chunk
    })
    req.on('end', () => resolve(body))
    req.on('error', reject)
  })
}

function authorized(req: IncomingMessage, token: string): boolean {
  return req.headers.authorization === `Bearer ${token}`
}

function controllerSessionIdentity(value: unknown): string | null {
  if (value === undefined || value === null) {
    return null
  }

  if (typeof value !== 'string') {
    throw new Error('invalid session identity')
  }

  const normalized = value.trim()

  // eslint-disable-next-line no-control-regex
  if (!normalized || normalized.length > MAX_CONTROLLER_SESSION_ID_CHARS || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new Error('invalid session identity')
  }

  return normalized
}

function controllerBoundedIdentity(value: unknown, name: string): string | null {
  if (value === undefined || value === null) {
    return null
  }

  if (typeof value !== 'string') {
    throw new Error(`invalid ${name}`)
  }
  const normalized = value.trim()

  // eslint-disable-next-line no-control-regex
  if (!normalized || normalized.length > MAX_CONTROLLER_SESSION_ID_CHARS || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new Error(`invalid ${name}`)
  }

  return normalized
}

function atomicWritePrivateJson(filePath: string, value: unknown): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  const temp = `${filePath}.${process.pid}.tmp`
  fs.writeFileSync(temp, JSON.stringify(value, null, 2), { encoding: 'utf-8', mode: 0o600 })

  try {
    fs.renameSync(temp, filePath)
  } catch (err: any) {
    if (err && (err.code === 'EPERM' || err.code === 'EBUSY')) {
      fs.copyFileSync(temp, filePath)
      fs.rmSync(temp, { force: true })
    } else {
      throw err
    }
  }

  try {
    fs.chmodSync(filePath, 0o600)
  } catch {
    // Best effort on Windows / filesystems without POSIX permissions.
  }
}

function removeOwnedControlFile(filePath: string, token: string): void {
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, 'utf-8')) as { token?: unknown }

    if (parsed.token !== token) {
      return
    }
    fs.rmSync(filePath, { force: true })
  } catch {
    // Missing/malformed files are safe to ignore at shutdown.
  }
}

function inventoryScript(maxText: number, maxElements: number): string {
  return `(async function () {
    var w = window;
    var state = w.__hermesWorkstationRefs;
    if (!state || state.href !== location.href) {
      state = { href: location.href, counter: 0, byRef: new Map(), byElement: new WeakMap() };
      w.__hermesWorkstationRefs = state;
    }

    var roots = [document];
    var candidates = [];
    var seen = new Set();
    while (roots.length) {
      var root = roots.shift();
      if (!root || !root.querySelectorAll) continue;
      var all = Array.from(root.querySelectorAll('*'));
      for (var i = 0; i < all.length; i++) {
        var el = all[i];
        if (el.shadowRoot) roots.push(el.shadowRoot);
        if (el.tagName === 'IFRAME') {
          try { if (el.contentDocument) roots.push(el.contentDocument); } catch (err) {}
        }
        if (seen.has(el)) continue;
        seen.add(el);
        var tag = String(el.tagName || '').toLowerCase();
        var role = String(el.getAttribute && el.getAttribute('role') || '').toLowerCase();
        var interactive = ['a','button','input','textarea','select','summary','option'].includes(tag) ||
          ['button','link','textbox','searchbox','checkbox','radio','combobox','menuitem','option','switch','slider','tab','treeitem','spinbutton'].includes(role) ||
          (typeof el.tabIndex === 'number' && el.tabIndex >= 0);
        if (interactive) candidates.push(el);
      }
    }

    var out = [];
    for (var j = 0; j < candidates.length && out.length < ${maxElements}; j++) {
      var el = candidates[j];
      var rect = el.getBoundingClientRect();
      var style = getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden' || rect.width < 1 || rect.height < 1) continue;
      var ref = state.byElement.get(el);
      if (!ref) {
        ref = '@e' + (++state.counter);
        state.byElement.set(el, ref);
        state.byRef.set(ref, el);
      }
      var tag = String(el.tagName || '').toLowerCase();
      var role = String(el.getAttribute('role') || tag || 'element');
      var type = String(el.getAttribute('type') || '').toLowerCase();
      var rawValue = 'value' in el ? String(el.value || '') : '';
      var value = type === 'password' && rawValue ? '[REDACTED]' : rawValue;
      var label = String(
        el.getAttribute('aria-label') ||
        el.getAttribute('alt') ||
        el.getAttribute('title') ||
        el.getAttribute('placeholder') ||
        el.innerText ||
        el.textContent ||
        el.getAttribute('name') ||
        ''
      ).replace(/\\s+/g, ' ').trim().slice(0, 240);
      out.push({
        ref: ref,
        tag: tag,
        role: role,
        label: label,
        value: value ? value.slice(0, 240) : undefined,
        disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true'
      });
    }

    var bodyText = String(document.body && document.body.innerText || '').replace(/\\u0000/g, '');
    return {
      url: location.href,
      title: document.title || '',
      text: bodyText.slice(0, ${maxText}),
      totalTextChars: bodyText.length,
      truncated: bodyText.length > ${maxText} || candidates.length > ${maxElements},
      elements: out
    };
  })()`
}

function pointScript(ref: string, focus: boolean): string {
  return `(function () {
    var state = window.__hermesWorkstationRefs;
    var el = state && state.byRef && state.byRef.get(${JSON.stringify(ref)});
    if (!el || !el.isConnected) return { success: false, error: 'stale_or_unknown_ref' };
    try { el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'auto' }); } catch (err) {}
    if (${focus ? 'true' : 'false'}) { try { el.focus({ preventScroll: true }); } catch (err) { try { el.focus(); } catch (err2) {} } }
    var r = el.getBoundingClientRect();
    if (!r || r.width < 1 || r.height < 1) return { success: false, error: 'element_not_visible' };
    return { success: true, x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
  })()`
}

function formatInventory(inv: PageInventory, full: boolean): string {
  const lines: string[] = []
  lines.push(`URL: ${inv.url}`)

  if (inv.title) {
    lines.push(`Title: ${inv.title}`)
  }
  lines.push('')

  if (inv.elements.length) {
    lines.push('Interactive elements:')

    for (const item of inv.elements) {
      const label = item.label ? ` "${item.label.replace(/"/g, '\\"')}"` : ''
      const value = item.value ? ` value="${item.value.replace(/"/g, '\\"')}"` : ''
      const disabled = item.disabled ? ' disabled' : ''
      lines.push(`- [${item.ref}] ${item.role || item.tag}${label}${value}${disabled}`)
    }
  }

  if (full || inv.text.trim()) {
    lines.push('', 'Page text:', inv.text.trim())
  }

  if (inv.truncated) {
    lines.push('', '[Snapshot truncated by Hermes Workstation budget]')
  }

  return lines.join('\n').trim()
}

export class WorkstationBrowserRuntime {
  private browserSession: Session | null = null
  private entries = new Map<string, BrowserEntry>()
  private taskTabs = new Map<string, string>()
  private activeTabId: string | null = null
  private ownerWindow: BrowserWindow | null = null
  private viewportGeometryWindow: BrowserWindow | null = null
  private viewportGeometryListener: (() => void) | null = null
  private attached = false
  private viewportHost: 'hub' | 'chat' | string | null = null
  private preferredTaskId: string | null = null
  private bounds: WorkstationBrowserBounds | null = null
  private paused = false
  private controlOwner: WorkstationBrowserControlOwner = 'agent'
  private lastError: string | null = null
  private viewVisible = true
  private cacheBytes: number | null = null
  private cacheTimer: NodeJS.Timeout | null = null
  private control: ControlHandle | null = null
  private downloads: WorkstationDownloadItem[] = []
  private browserTasks: BrowserTaskLifecycle<BrowserEntry, BrowserTaskShowContext> | null = null
  private browserTasksRestored = false
  private browserSessionState: BrowserSessionStateFilePersistence | null = null
  private browserSessionStateRestored = false
  private browserSessionStateRestoring = false
  private browserSessionPersistenceSuppressed = false
  private pendingSessionTabs = new Map<string, BrowserSessionTab>()
  private restoredTabOrder: string[] = []
  private restoredLogicalActiveTabId: string | null = null

  constructor(browserSessionState: BrowserSessionStateFilePersistence | null = null) {
    this.browserSessionState = browserSessionState
  }

  ensure(): WorkstationBrowserState {
    this.ensureSession()
    this.ensureBrowserSessionStateRestored()

    if (!this.activeTabId || !this.entries.has(this.activeTabId)) {
      const restoredLogicalActiveTabId = this.restoredLogicalActiveTabId
      this.withBrowserSessionProjectionSuppressed(() => this.createTab('about:blank', true))

      if (restoredLogicalActiveTabId && this.pendingSessionTabs.has(restoredLogicalActiveTabId)) {
        this.restoredLogicalActiveTabId = restoredLogicalActiveTabId
      }

      this.persistBrowserSessionState()
    }

    void this.refreshCacheSize()
    this.emitState()

    return this.state()
  }

  state(): WorkstationBrowserState {
    return {
      runtime: 'electron-chromium',
      ready: this.browserSession !== null,
      attached: this.attached,
      viewportHost: this.attached ? this.viewportHost : null,
      backgroundCapable: true,
      paused: this.paused,
      controlOwner: this.controlOwner,
      controlReady: this.control !== null,
      profilePath: workstationBrowserProfilePath(),
      cacheBytes: this.cacheBytes,
      activeTabId: this.activeTabId,
      tabs: Array.from(this.entries.values()).map(entry => this.tabState(entry)),
      tasks: this.listTasks(),
      downloads: [...this.downloads],
      lastError: this.lastError
    }
  }

  resources(): WorkstationResourceSnapshot {
    this.ensureBrowserSessionStateRestored()

    return buildWorkstationResourceSnapshot(this.state(), taskId => this.getTaskJournal(taskId))
  }

  events(taskId: string | null = null, limit = MAX_EVENTS_ENDPOINT): WorkstationEventSnapshot {
    this.ensureBrowserSessionStateRestored()
    const boundedLimit = Number.isFinite(limit)
      ? Math.min(MAX_EVENTS_ENDPOINT, Math.max(1, Math.trunc(limit)))
      : MAX_EVENTS_ENDPOINT
    const selectedTaskId = typeof taskId === 'string' && taskId.trim() ? taskId.trim() : null
    const knownTaskIds = this.listTasks().map(task => task.taskId)
    const taskIds = selectedTaskId ? (knownTaskIds.includes(selectedTaskId) ? [selectedTaskId] : []) : knownTaskIds
    const events = taskIds
      .flatMap(candidate => this.getTaskJournal(candidate))
      .sort((left, right) => String(left.timestamp ?? '').localeCompare(String(right.timestamp ?? '')))
      .slice(-boundedLimit)

    return {
      schema_version: WORKSTATION_EVENT_SCHEMA_VERSION,
      runtime: 'electron-chromium',
      generated_at: new Date().toISOString(),
      task_id: selectedTaskId,
      events
    }
  }

  getSession(): Session {
    this.ensureSession()

    return this.browserSession!
  }

  getActiveWebContents(): WebContents | null {
    if (!this.activeTabId) {
      return null
    }

    return this.entries.get(this.activeTabId)?.view.webContents ?? null
  }

  getWebContents(tabId: string): WebContents | null {
    return this.entries.get(tabId)?.view.webContents ?? null
  }

  createTask(seed: BrowserTaskSeed = {}): BrowserTask {
    this.ensureSession()
    this.ensureBrowserSessionStateRestored()
    const task = this.withBrowserSessionProjectionSuppressed(() => this.taskLifecycle().createTask(seed))
    this.persistBrowserSessionState()

    return task
  }

  showTask(taskId: string, window: BrowserWindow, bounds: WorkstationBrowserBounds, host = 'hub'): BrowserTask {
    this.ensureSession()
    this.ensureBrowserSessionStateRestored()

    const task = this.withBrowserSessionProjectionSuppressed(() =>
      this.taskLifecycle().showTask(taskId, { window, bounds, host })
    )

    this.persistBrowserSessionState()

    return task
  }

  hideTask(taskId: string): BrowserTask {
    this.ensureBrowserSessionStateRestored()
    const task = this.withBrowserSessionProjectionSuppressed(() => this.taskLifecycle().hideTask(taskId))
    this.persistBrowserSessionState()

    return task
  }

  parkTask(taskId: string): BrowserTask {
    this.ensureBrowserSessionStateRestored()
    const task = this.withBrowserSessionProjectionSuppressed(() => this.taskLifecycle().parkTask(taskId))
    this.persistBrowserSessionState()

    return task
  }

  destroyTask(taskId: string): boolean {
    this.ensureBrowserSessionStateRestored()

    try {
      const destroyed = this.withBrowserSessionProjectionSuppressed(() => this.taskLifecycle().destroyTask(taskId))

      if (destroyed) {
        this.removePendingTaskTab(taskId)
        this.taskTabs.delete(taskId)
      }

      this.persistBrowserSessionState()
      this.emitState()

      return destroyed
    } catch (error) {
      // BrowserTaskLifecycle applies explicit destroy in process before it
      // persists the composite snapshot. If durability fails, keep surfacing
      // that error, but finish the corresponding process-local cleanup so a
      // later task with the same id cannot inherit stale recovery metadata.
      if (!this.taskLifecycle().task(taskId)) {
        this.removePendingTaskTab(taskId)
      }
      this.emitState()
      throw error
    }
  }

  clearParkedTasks(): number {
    this.ensureBrowserSessionStateRestored()
    const parked = this.taskLifecycle()
      .listTasks()
      .filter(task => task.status === 'parked')
    let count = 0

    for (const task of parked) {
      try {
        if (this.destroyTask(task.taskId)) {
          count++
        }
      } catch {
        // Individual destroy failures do not stop bulk clearing.
      }
    }

    this.emitState()

    return count
  }

  listTasks(): BrowserTask[] {
    this.ensureBrowserSessionStateRestored()

    return this.taskLifecycle().listTasks()
  }

  createTab(target = 'about:blank', activate = true, ownerTaskId: string | null = null): WorkstationBrowserState {
    this.ensureSession()

    if (!this.browserSessionStateRestoring) {
      this.ensureBrowserSessionStateRestored()
    }

    return this.createTabEntry(target, activate, ownerTaskId)
  }

  private createTabEntry(
    target: string,
    activate: boolean,
    ownerTaskId: string | null,
    restoredTab: BrowserSessionTab | null = null
  ): WorkstationBrowserState {
    const url = normalizeWorkstationBrowserTarget(target)

    if (ownerTaskId) {
      const mapped = this.taskTabs.get(ownerTaskId)
      const existing = mapped ? this.entries.get(mapped) : null

      if (existing && !existing.view.webContents.isDestroyed()) {
        if (activate) {
          this.activateTab(existing.id)
        } else {
          this.parkEntry(existing)
        }

        if (url !== 'about:blank' && existing.view.webContents.getURL() !== url) {
          this.updateEntrySafeMetadata(existing, url, existing.view.webContents.getTitle())
          void existing.view.webContents.loadURL(url).catch(error => this.recordError(error))
        }

        this.persistBrowserSessionState()
        this.emitState()

        return this.state()
      }

      if (mapped) {
        this.taskTabs.delete(ownerTaskId)
      }
    }

    const requestedId = restoredTab?.id ?? crypto.randomUUID()
    const id = this.entries.has(requestedId) ? crypto.randomUUID() : requestedId

    const view = new WebContentsView({
      webPreferences: {
        session: this.browserSession!,
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        spellcheck: true,
        backgroundThrottling: false
      }
    })

    view.setBackgroundColor('#111111')

    if (typeof view.webContents.setUserAgent === 'function') {
      view.webContents.setUserAgent(getStandardChromeUserAgent())
    }

    const safeUrl = restoredTab?.safeUrl ?? safeRestorableUrlMetadata(url)
    const safeTitle = restoredTab?.safeTitle ?? null

    const entry: BrowserEntry = {
      id,
      view,
      loading: false,
      crashed: false,
      ownerTaskId,
      safeUrl,
      safeTitle,
      recoveryState: restoredTab?.recoveryState ?? 'live',
      recoveryReason: restoredTab?.recoveryReason ?? (safeUrl !== url ? 'unsafe-metadata' : null)
    }

    this.entries.set(id, entry)
    this.pendingSessionTabs.delete(id)
    this.applyFrameRate(entry, false)

    if (ownerTaskId) {
      this.taskTabs.set(ownerTaskId, id)
    }
    this.wireEntry(entry)

    if (activate || !this.activeTabId) {
      this.activateTab(id)
    }

    if (url !== 'about:blank') {
      void view.webContents.loadURL(url).catch(error => this.recordError(error))
    }

    this.reconcileRestoredEntryOrder()
    this.persistBrowserSessionState()
    this.emitState()

    return this.state()
  }

  closeTab(tabId: string): WorkstationBrowserState {
    const entry = this.entries.get(tabId)

    if (!entry) {
      return this.state()
    }

    const wasActive = this.activeTabId === tabId

    if (entry.ownerTaskId) {
      this.rememberPendingSessionTab(entry, 'stale', 'page-gone')
    }
    this.discardEntry(entry)

    if (wasActive) {
      const replacement = this.entries.values().next().value as BrowserEntry | undefined

      if (replacement) {
        this.activateTab(replacement.id)
      }
    }

    if (this.entries.size === 0) {
      this.createTab('about:blank', true)
    }
    this.persistBrowserSessionState()
    this.emitState()

    return this.state()
  }

  activateTab(tabId: string): WorkstationBrowserState {
    const entry = this.entries.get(tabId)

    if (!entry) {
      throw new Error(`Unknown Hermes Browser tab: ${tabId}`)
    }

    if (this.activeTabId === tabId) {
      return this.state()
    }

    const wasAttached = this.attached && this.ownerWindow && !this.ownerWindow.isDestroyed() && this.bounds

    if (this.attached) {
      this.detachActiveView(false)
    }
    this.activeTabId = tabId

    if (!this.browserSessionStateRestoring) {
      this.restoredLogicalActiveTabId = null
    }

    if (wasAttached && this.ownerWindow && this.bounds) {
      this.ensureChildView(this.ownerWindow, entry.view)
      entry.view.setBounds(this.bounds)
      this.applyFrameRate(entry, true)

      try {
        entry.view.webContents.focus()
      } catch {
        // View may have crashed between checks.
      }

      this.attached = true
    }

    this.persistBrowserSessionState()
    this.emitState()

    return this.state()
  }

  async navigate(value: string): Promise<WorkstationBrowserState> {
    this.ensure()
    const wc = this.getActiveWebContents()

    if (!wc) {
      throw new Error('Hermes Browser has no active tab.')
    }
    await wc.loadURL(normalizeWorkstationBrowserTarget(value))
    this.emitState()

    return this.state()
  }

  back(): WorkstationBrowserState {
    const wc = this.getActiveWebContents()

    if (wc?.navigationHistory.canGoBack()) {
      wc.navigationHistory.goBack()
    }

    return this.state()
  }

  forward(): WorkstationBrowserState {
    const wc = this.getActiveWebContents()

    if (wc?.navigationHistory.canGoForward()) {
      wc.navigationHistory.goForward()
    }

    return this.state()
  }

  reload(): WorkstationBrowserState {
    this.getActiveWebContents()?.reload()

    return this.state()
  }

  stop(): WorkstationBrowserState {
    this.getActiveWebContents()?.stop()

    return this.state()
  }

  focus(): WorkstationBrowserState {
    this.getActiveWebContents()?.focus()

    return this.state()
  }

  attach(
    window: BrowserWindow,
    rawBounds: WorkstationBrowserBounds,
    host: 'hub' | 'chat' | string = 'hub',
    preferredTaskId?: string | null
  ): WorkstationBrowserState {
    this.ensure()
    this.preferredTaskId = preferredTaskId || null
    if (preferredTaskId) {
      const tabId = this.taskTabs.get(preferredTaskId)
      if (tabId && this.entries.has(tabId)) {
        const candidate = this.entries.get(tabId)
        if (candidate && !candidate.crashed && !candidate.view.webContents.isDestroyed()) {
          if (this.activeTabId !== tabId) {
            this.activateTab(tabId)
          }
        }
      }
    }
    const entry = this.activeEntry()

    if (!entry) {
      return this.state()
    }
    const bounds = this.validBounds(window, rawBounds)

    if (!bounds) {
      return this.state()
    }

    if (this.ownerWindow && this.ownerWindow !== window && this.attached) {
      this.detachActiveView(false)
    }
    this.ownerWindow = window
    this.bindViewportGeometry(window)
    this.bounds = bounds
    this.viewportHost = host
    this.ensureChildView(window, entry.view)
    entry.view.setBounds(bounds)
    this.applyFrameRate(entry, true)

    try {
      entry.view.webContents.focus()
    } catch {
      // View may have crashed between checks.
    }

    this.attached = true
    this.emitState()

    return this.state()
  }

  setBounds(
    window: BrowserWindow,
    rawBounds: WorkstationBrowserBounds,
    expectedHost?: string
  ): WorkstationBrowserState {
    const bounds = this.validBounds(window, rawBounds)

    if (!bounds) {
      return this.state()
    }

    // Chat and Browser Hub share the same BrowserWindow, so the sender window
    // alone cannot identify which host's ResizeObserver produced this update.
    // Ignore geometry from a stale/non-owner host; otherwise an unmounted or
    // background pane can move the one live native view over its own rectangle.
    if (
      this.attached &&
      this.ownerWindow === window &&
      expectedHost !== undefined &&
      expectedHost !== this.viewportHost
    ) {
      return this.state()
    }

    this.bounds = bounds

    if (this.attached && this.ownerWindow === window) {
      const entry = this.activeEntry()

      if (entry) {
        entry.view.setBounds(bounds)
      }
    }

    return this.state()
  }

  detach(window?: BrowserWindow | null): WorkstationBrowserState {
    if (window && this.ownerWindow && window !== this.ownerWindow) {
      return this.state()
    }
    this.detachActiveView(true)
    this.viewportHost = null
    this.preferredTaskId = null
    this.emitState()

    return this.state()
  }

  setVisible(visible: boolean): WorkstationBrowserState {
    this.viewVisible = visible
    const entry = this.activeEntry()

    if (!visible) {
      if (entry) {
        this.removeChildView(entry)
      }
    } else {
      if (entry && this.ownerWindow && this.bounds && this.attached) {
        this.ensureChildView(this.ownerWindow, entry.view)
        entry.view.setBounds(this.bounds)
      }
    }

    return this.state()
  }

  clearError(): WorkstationBrowserState {
    if (this.lastError) {
      this.lastError = null
      this.emitState()
    }

    return this.state()
  }

  transferViewport(
    window: BrowserWindow,
    targetHost: 'hub' | 'chat' | string,
    rawBounds: WorkstationBrowserBounds
  ): WorkstationBrowserState {
    if (this.attached) {
      this.detachActiveView(false)
    }

    return this.attach(window, rawBounds, targetHost)
  }

  getTaskJournal(taskId: string): any[] {
    const cleanId = String(taskId || '')
      .trim()
      .replace(/[^a-zA-Z0-9_-]/g, '_')

    if (!cleanId) {
      return []
    }
    const hermesHome = process.env.HERMES_HOME || path.join(os.homedir(), '.hermes')
    const journalPath = path.join(hermesHome, 'workstation', 'journals', `${cleanId}.jsonl`)

    if (!fs.existsSync(journalPath)) {
      return []
    }

    try {
      const lines = fs.readFileSync(journalPath, 'utf-8').split('\n')
      const events: any[] = []
      let prevDt: number | null = null

      for (const line of lines) {
        if (!line.trim()) {
          continue
        }

        try {
          const ev = JSON.parse(line)
          const ts = ev.timestamp ? new Date(ev.timestamp).getTime() : 0

          if (prevDt !== null && ts > 0) {
            ev.elapsed_seconds = Math.max(0, Math.round(((ts - prevDt) / 1000) * 100) / 100)
          } else {
            ev.elapsed_seconds = 0
          }

          if (ts > 0) {
            prevDt = ts
          }
          events.push(ev)
        } catch {
          // ignore malformed line
        }
      }

      return events
    } catch {
      return []
    }
  }

  async pause(): Promise<WorkstationBrowserState> {
    if (this.paused) {
      return this.state()
    }
    this.paused = true
    // Do not destroy tabs or auth state. Hidden Chromium keeps its process and
    // profile; pausing is an agent-control gate, not a logout/reset operation.
    this.emitState()

    return this.state()
  }

  async resume(): Promise<WorkstationBrowserState> {
    if (!this.paused) {
      return this.state()
    }
    this.paused = false
    this.emitState()

    return this.state()
  }

  takeControl(): WorkstationBrowserState {
    this.controlOwner = 'human'
    this.emitState()

    return this.state()
  }

  releaseControl(): WorkstationBrowserState {
    this.controlOwner = 'agent'
    this.emitState()

    return this.state()
  }

  async cleanupCache(force = false): Promise<WorkstationBrowserState> {
    this.ensureSession()
    const size = await this.browserSession!.getCacheSize()
    this.cacheBytes = size
    const busy = Array.from(this.entries.values()).some(entry => entry.loading)

    if ((force || size > cacheLimitBytes()) && !busy) {
      // Cache only: cookies, localStorage, IndexedDB and login/session state survive.
      await this.browserSession!.clearCache()
      this.cacheBytes = await this.browserSession!.getCacheSize()
    }

    this.emitState()

    return this.state()
  }

  async startControlServer(): Promise<void> {
    if (this.control) {
      return
    }

    // Keep the loopback controller available from app startup, but defer
    // creating a detached WebContentsView until the browser UI or a browser_*
    // action actually needs one. An unhosted view is intentionally valid for
    // background work, yet Electron exposes it as a page target without an
    // owning BrowserWindow; eager creation makes packaged Electron inspectors
    // wait forever for that target to initialize and spends Chromium resources
    // before the Workstation Browser is used.
    this.ensureSession()
    this.ensureBrowserSessionStateRestored()
    const token = crypto.randomBytes(32).toString('base64url')
    const controlPath = workstationBrowserControlPath()

    const server = http.createServer(async (req, res) => {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1')

      if (!authorized(req, token)) {
        sendJson(res, 401, { success: false, error: 'unauthorized' })

        return
      }

      if (url.pathname === '/health' && req.method === 'GET') {
        sendJson(res, 200, { success: true, runtime: 'electron-chromium', state: this.state() })

        return
      }

      if (url.pathname === '/resources' && req.method === 'GET') {
        sendJson(res, 200, { success: true, ...this.resources() })

        return
      }

      if (url.pathname === '/events' && req.method === 'GET') {
        const taskId = url.searchParams.get('task_id')
        const rawLimit = Number(url.searchParams.get('limit') ?? MAX_EVENTS_ENDPOINT)
        sendJson(res, 200, { success: true, ...this.events(taskId, rawLimit) })

        return
      }

      if (url.pathname !== '/v1/action' || req.method !== 'POST') {
        sendJson(res, 404, { success: false, error: 'not_found' })

        return
      }

      try {
        const raw = await readBody(req)
        const request = JSON.parse(raw || '{}') as BrowserControlRequest
        const result = await this.executeControlRequest(request)

        if (this.lastError) {
          this.lastError = null
          this.emitState()
        }

        sendJson(res, 200, { success: true, result })
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        this.recordError(error)
        const structured = normalizeWorkstationControllerError(error)
        const status = structured.error_code === 'USER_CONTROL_ACTIVE' || structured.error_code === 'NO_BOUND_TAB' || structured.error_code === 'STALE_REF' ? 409 : 400
        // Keep `error` for older controller clients while new clients receive
        // a stable code and a recovery action rather than retrying blindly.
        sendJson(res, status, { success: false, error: message, ...structured })
      }
    })

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', () => {
        server.off('error', reject)
        resolve()
      })
    })

    const address = server.address()

    if (!address || typeof address === 'string') {
      server.close()
      throw new Error('Hermes Browser controller could not bind a loopback TCP port.')
    }

    const control: ControlHandle = {
      server,
      url: `http://127.0.0.1:${address.port}`,
      token,
      controlPath
    }

    this.control = control
    atomicWritePrivateJson(controlPath, {
      version: CONTROL_FILE_VERSION,
      pid: process.pid,
      url: control.url,
      token,
      runtime: 'electron-chromium',
      profile_path: workstationBrowserProfilePath(),
      created_at: new Date().toISOString()
    })
    this.emitState()
  }

  async stopControlServer(): Promise<void> {
    const control = this.control
    this.control = null

    if (!control) {
      return
    }
    removeOwnedControlFile(control.controlPath, control.token)
    await new Promise<void>(resolve => control.server.close(() => resolve()))
    this.emitState()
  }

  async destroy(): Promise<void> {
    if (this.cacheTimer) {
      clearInterval(this.cacheTimer)
    }
    this.cacheTimer = null
    await this.stopControlServer()
    // Persist the structural projection before Electron begins destroying
    // process-local WebContents objects. Destruction events during shutdown
    // must not erase the logical restart state we just committed.
    this.persistBrowserSessionState()
    this.browserSessionPersistenceSuppressed = true
    this.detachActiveView(false)
    this.unbindViewportGeometry()

    for (const entry of this.entries.values()) {
      this.removeChildView(entry)

      try {
        entry.view.webContents.close()
      } catch {
        // Best effort during app shutdown.
      }
    }

    // App shutdown destroys Chromium process objects, but BrowserSessionState
    // and BrowserTask metadata remain on disk. Ordinary tabs are recreated from
    // sanitized metadata; task pages remain lazy under BrowserTaskLifecycle.
    this.entries.clear()
    this.taskTabs.clear()
    this.activeTabId = null
    this.viewportHost = null
  }

  private async executeControlRequest(request: BrowserControlRequest): Promise<Record<string, unknown>> {
    const action = typeof request.action === 'string' ? request.action : ''
    const args =
      request.arguments && typeof request.arguments === 'object' ? (request.arguments as Record<string, unknown>) : {}
    const taskId = typeof request.task_id === 'string' && request.task_id.trim() ? request.task_id.trim() : 'default'
    const sessionHost = controllerSessionIdentity(request.session_id)
    const kanbanCardId = controllerBoundedIdentity(request.kanban_card_id ?? request.card_id, 'kanban card identity')
    const runId = controllerBoundedIdentity(request.run_id, 'run identity')

    if (!action.startsWith('browser_')) {
      throw new Error('unsupported_action')
    }

    // Extension operations deliberately remain on the authenticated
    // loopback controller, but they do not need (and must not fabricate) a
    // BrowserTask page just to load or verify a Chromium capability.
    if (action === 'browser_extension_load') {
      return this.loadExtensionForController(String(args.extension_id ?? ''), String(args.path ?? ''))
    }
    if (action === 'browser_extension_verify') {
      return this.verifyExtensionForController(String(args.extension_id ?? ''))
    }
    if (action === 'browser_extension_remove') {
      return this.removeExtensionForController(String(args.extension_id ?? ''))
    }

    const mutating = new Set([
      'browser_navigate',
      'browser_click',
      'browser_type',
      'browser_scroll',
      'browser_back',
      'browser_press',
      'browser_extension_open_options'
    ])

    if (mutating.has(action)) {
      this.assertAgentControl()
    }

    if (sessionHost || kanbanCardId || runId) {
      this.bindControllerSessionIdentity(taskId, sessionHost, kanbanCardId, runId)
    }

    if (action === 'browser_navigate') {
      const entry = this.entryForTask(taskId, true, sessionHost, kanbanCardId, runId)!
      const url = normalizeWorkstationBrowserTarget(String(args.url ?? ''))
      await entry.view.webContents.loadURL(url)
      if (!this.activeTabId || this.activeTabId === entry.id || (this.preferredTaskId && this.preferredTaskId === taskId)) {
        this.activateTab(entry.id)
      }

      for (const [id, candidate] of this.entries.entries()) {
        if (id !== entry.id && !candidate.ownerTaskId && (candidate.safeUrl === 'about:blank' || !candidate.safeUrl)) {
          this.closeTab(id)
        }
      }

      try {
        for (const win of BrowserWindow.getAllWindows()) {
          if (!win.isDestroyed() && win.webContents) {
            win.webContents.send('hermes:workstation-browser:open-chat-preview', { url, taskId, tabId: entry.id })
          }
        }
      } catch {
        // Notification is best-effort.
      }

      return this.snapshotForEntry(entry, false)
    }

    let entry = this.entryForTask(taskId, false, sessionHost, kanbanCardId, runId)

    if (!entry) {
      if (this.pendingTabForTask(taskId)) {
        entry = this.entryForTask(taskId, true, sessionHost, kanbanCardId, runId)
      }

      if (!entry && this.activeTabId) {
        const active = this.entries.get(this.activeTabId)

        if (active && !active.ownerTaskId && !active.view.webContents.isDestroyed() && !active.crashed) {
          this.taskTabs.set(taskId, active.id)
          active.ownerTaskId = taskId
          entry = active
        }
      }

      if (!entry) {
        entry = this.entryForTask(taskId, true, sessionHost, kanbanCardId, runId)
      }
    }

    if (!entry) {
      throw new Error('no_bound_browser_tab: call browser_navigate first')
    }

    if (!this.activeTabId) {
      this.activateTab(entry.id)
    }

    if (action === 'browser_extension_open_options') {
      const extensionId = String(args.extension_id ?? '').trim().toLowerCase()
      const optionsPath = String(args.options_path ?? 'options.html').replace(/^[/\\]+/, '')
      if (!/^[a-p]{32}$/.test(extensionId) || !optionsPath || optionsPath.includes('..')) {
        throw new Error('invalid_extension_options_request')
      }
      const verified = this.verifyExtensionForController(extensionId)
      if (!verified.loaded) {
        throw new Error('extension_not_loaded')
      }
      const optionsUrl = `chrome-extension://${extensionId}/${optionsPath}`
      await entry.view.webContents.loadURL(optionsUrl)
      return { ...verified, options_url: optionsUrl }
    }

    switch (action) {
      case 'browser_snapshot':
        return this.snapshotForEntry(entry, Boolean(args.full))

      case 'browser_click':
        await this.clickRef(entry, String(args.ref ?? ''))
        await delay(220)

        return this.snapshotForEntry(entry, false)

      case 'browser_type':
        await this.typeRef(entry, String(args.ref ?? ''), String(args.text ?? ''))
        await delay(160)

        return this.snapshotForEntry(entry, false)

      case 'browser_scroll':
        await this.scrollEntry(entry, String(args.direction ?? 'down'))
        await delay(140)

        return this.snapshotForEntry(entry, false)

      case 'browser_back':
        if (entry.view.webContents.navigationHistory.canGoBack()) {
          entry.view.webContents.navigationHistory.goBack()
        }
        await delay(220)

        return this.snapshotForEntry(entry, false)

      case 'browser_press':
        await this.pressKey(entry, String(args.key ?? ''))
        await delay(120)

        return this.snapshotForEntry(entry, false)

      case 'browser_get_images':
        return this.imagesForEntry(entry)

      case 'browser_console':
        return this.consoleForEntry(entry, args)

      case 'browser_vision':
        return this.screenshotForEntry(entry)

      default:
        throw new Error(`unsupported_action:${action}`)
    }
  }

  private assertAgentControl(): void {
    if (this.paused) {
      throw new Error('Hermes Browser is paused. Resume it before agent actions continue.')
    }

    if (this.controlOwner === 'human') {
      throw new Error('Hermes Browser is under human control. Release Control before agent actions continue.')
    }
  }

  private bindControllerSessionIdentity(
    taskId: string,
    sessionHost?: string | null,
    kanbanCardId?: string | null,
    runId?: string | null
  ): void {
    this.ensureBrowserSessionStateRestored()
    const lifecycle = this.taskLifecycle()

    if (!lifecycle.task(taskId)) {
      return
    }
    this.withBrowserSessionProjectionSuppressed(() => {
      if (sessionHost) {
        lifecycle.bindSessionHost(taskId, sessionHost)
      }

      if (kanbanCardId) {
        lifecycle.bindKanbanCard(taskId, kanbanCardId)
      }

      if (runId) {
        lifecycle.bindRun(taskId, runId)
      }
    })
    this.persistBrowserSessionState()
  }

  private taskLifecycle(): BrowserTaskLifecycle<BrowserEntry, BrowserTaskShowContext> {
    if (this.browserTasks) {
      return this.browserTasks
    }

    this.browserTasks = new BrowserTaskLifecycle(
      {
        ensurePage: taskId => {
          const entry = this.rawEntryForTask(taskId, true)

          if (!entry) {
            throw new Error(`BrowserTask page could not be created: ${taskId}`)
          }

          return entry
        },
        pageForTask: taskId => this.rawEntryForTask(taskId, false),
        pageIsAlive: entry => !entry.crashed && !entry.view.webContents.isDestroyed(),
        showPage: (_taskId, entry, context) => {
          if (entry.id !== this.activeTabId) {
            this.activateTab(entry.id)
          }
          this.attach(context.window, context.bounds, context.host ?? 'hub')
        },
        hidePage: (_taskId, entry) => {
          this.removeChildView(entry)

          if (entry.id === this.activeTabId) {
            this.attached = false
          }
          this.applyFrameRate(entry, false)
          this.emitState()
        },
        parkPage: (_taskId, entry) => {
          if (entry.id === this.activeTabId && this.attached) {
            this.removeChildView(entry)
            this.attached = false
          }

          this.parkEntry(entry)
          this.emitState()
        },
        destroyPage: (_taskId, entry) => {
          this.closeTab(entry.id)
        }
      },
      this.sessionStatePersistence().browserTaskPersistence()
    )

    return this.browserTasks
  }

  private sessionStatePersistence(): BrowserSessionStateFilePersistence {
    if (this.browserSessionState) {
      return this.browserSessionState
    }
    this.browserSessionState = new BrowserSessionStateFilePersistence(
      workstationBrowserSessionStatePath(),
      workstationBrowserTaskStatePath()
    )

    return this.browserSessionState
  }

  private ensureBrowserTasksRestored(): void {
    if (this.browserTasksRestored) {
      return
    }
    this.taskLifecycle().restore()
    this.browserTasksRestored = true
  }

  private ensureBrowserSessionStateRestored(): void {
    if (this.browserSessionStateRestored || this.browserSessionStateRestoring) {
      return
    }
    this.browserSessionStateRestoring = true
    this.browserSessionPersistenceSuppressed = true

    try {
      const snapshot = this.sessionStatePersistence().load()
      this.ensureBrowserTasksRestored()

      if (snapshot) {
        this.restoreSessionTabs(snapshot)
      }
      this.browserSessionStateRestored = true
    } finally {
      this.browserSessionStateRestoring = false
      this.browserSessionPersistenceSuppressed = false
    }

    this.reconcileRestoredEntryOrder()

    if (this.pendingSessionTabs.size === 0) {
      this.restoredTabOrder = []
    }
    this.persistBrowserSessionState()
  }

  private restoreSessionTabs(snapshot: BrowserSessionStateSnapshot): void {
    this.restoredTabOrder = snapshot.tabs.map(tab => tab.id)
    this.restoredLogicalActiveTabId = snapshot.activeTabId

    for (const saved of snapshot.tabs) {
      const restored: BrowserSessionTab = {
        ...saved,
        recoveryState: 'restored',
        recoveryReason: 'process-restart'
      }

      if (saved.browserTaskId) {
        this.pendingSessionTabs.set(saved.id, restored)

        continue
      }

      this.createTabEntry(saved.safeUrl ?? 'about:blank', false, null, restored)
    }

    if (snapshot.activeTabId && this.entries.has(snapshot.activeTabId)) {
      this.activateTab(snapshot.activeTabId)
      this.restoredLogicalActiveTabId = null
    }

    this.reconcileRestoredEntryOrder()
  }

  private entryForTask(
    taskId: string,
    create: boolean,
    sessionHost: string | null = null,
    kanbanCardId: string | null = null,
    runId: string | null = null
  ): BrowserEntry | null {
    this.ensureBrowserSessionStateRestored()
    const lifecycle = this.taskLifecycle()

    if (create) {
      this.withBrowserSessionProjectionSuppressed(() =>
        lifecycle.createTask({ taskId, sessionHost, kanbanCardId, runId })
      )
    } else if (!lifecycle.task(taskId)) {
      const legacyEntry = this.rawEntryForTask(taskId, false)

      if (!legacyEntry) {
        return null
      }
      this.withBrowserSessionProjectionSuppressed(() =>
        lifecycle.createTask({ taskId, sessionHost, kanbanCardId, runId })
      )
    }

    const entry = this.rawEntryForTask(taskId, create)
    const visible = entry?.id === this.activeTabId && this.attached

    if (entry && !visible) {
      this.withBrowserSessionProjectionSuppressed(() => lifecycle.parkTask(taskId))
    }
    this.persistBrowserSessionState()

    return entry
  }

  private rawEntryForTask(taskId: string, create: boolean): BrowserEntry | null {
    const mapped = this.taskTabs.get(taskId)

    if (mapped) {
      const entry = this.entries.get(mapped)

      if (entry && !entry.crashed && !entry.view.webContents.isDestroyed()) {
        if (entry.id !== this.activeTabId || !this.attached) {
          this.parkEntry(entry)
        }

        return entry
      }

      if (entry) {
        this.discardEntry(entry)
      } else {
        this.taskTabs.delete(taskId)
      }

      if (!create) {
        this.emitState()
      }
    }

    if (!create) {
      return null
    }
    let restored = this.pendingTabForTask(taskId)

    if (restored?.recoveryState === 'stale') {
      const staleId = restored.id
      const replacementId = crypto.randomUUID()
      this.pendingSessionTabs.delete(staleId)
      this.restoredTabOrder = this.restoredTabOrder.map(id => (id === staleId ? replacementId : id))
      restored = {
        ...restored,
        id: replacementId,
        recoveryState: 'live',
        recoveryReason: 'page-gone'
      }
    }

    this.createTabEntry(restored?.safeUrl ?? 'about:blank', !this.activeTabId, taskId, restored)
    const id = this.taskTabs.get(taskId)
    const entry = id ? (this.entries.get(id) ?? null) : null

    if (entry) {
      this.parkEntry(entry)
    }

    return entry
  }

  private async snapshotForEntry(entry: BrowserEntry, full: boolean): Promise<Record<string, unknown>> {
    const wc = entry.view.webContents

    if (wc.isDestroyed()) {
      throw new Error('browser_tab_destroyed')
    }

    const inv = (await wc.executeJavaScript(
      inventoryScript(full ? FULL_TEXT_CHARS : COMPACT_TEXT_CHARS, full ? FULL_ELEMENTS : COMPACT_ELEMENTS),
      true
    )) as PageInventory

    return {
      success: true,
      runtime: 'electron-chromium',
      task_id: entry.ownerTaskId,
      tab_id: entry.id,
      url: inv.url,
      title: inv.title,
      snapshot: formatInventory(inv, full),
      truncated: inv.truncated,
      total_text_chars: inv.totalTextChars,
      element_count: inv.elements.length
    }
  }

  private async resolvePoint(entry: BrowserEntry, ref: string, focus: boolean): Promise<{ x: number; y: number }> {
    if (!ref) {
      throw new Error('ref_required')
    }

    const result = (await entry.view.webContents.executeJavaScript(pointScript(ref, focus), true)) as {
      success?: boolean
      error?: string
      x?: number
      y?: number
    }

    if (!result?.success || !Number.isFinite(result.x) || !Number.isFinite(result.y)) {
      throw new Error(result?.error || 'element_unavailable')
    }

    return { x: Number(result.x), y: Number(result.y) }
  }

  private async ensureDebugger(wc: WebContents): Promise<void> {
    if (wc.debugger.isAttached()) {
      return
    }

    try {
      wc.debugger.attach('1.3')
    } catch (error) {
      if (!wc.debugger.isAttached()) {
        throw error
      }
    }
  }

  private async cdp(wc: WebContents, method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    await this.ensureDebugger(wc)

    return wc.debugger.sendCommand(method, params)
  }

  private async cdpClick(wc: WebContents, x: number, y: number): Promise<void> {
    await this.cdp(wc, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x, y })
    await this.cdp(wc, 'Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 })
    await this.cdp(wc, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 })
  }

  private async clickRef(entry: BrowserEntry, ref: string): Promise<void> {
    const wc = entry.view.webContents
    const point = await this.resolvePoint(entry, ref, true)
    await this.cdpClick(wc, point.x, point.y)
  }

  private async typeRef(entry: BrowserEntry, ref: string, text: string): Promise<void> {
    const wc = entry.view.webContents
    const point = await this.resolvePoint(entry, ref, true)
    await this.cdpClick(wc, point.x, point.y)
    const modifiers = process.platform === 'darwin' ? 4 : 2 // Meta=4, Ctrl=2 in CDP Input domain.
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'rawKeyDown', key: 'a', code: 'KeyA', modifiers })
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'keyUp', key: 'a', code: 'KeyA', modifiers })
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'rawKeyDown', key: 'Backspace', code: 'Backspace' })
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'keyUp', key: 'Backspace', code: 'Backspace' })
    await this.cdp(wc, 'Input.insertText', { text })
  }

  private async scrollEntry(entry: BrowserEntry, direction: string): Promise<void> {
    const wc = entry.view.webContents
    const sign = direction.toLowerCase() === 'up' ? -1 : 1
    const viewport = (await wc.executeJavaScript(
      '({ width: window.innerWidth, height: window.innerHeight })',
      true
    )) as { width?: number; height?: number }
    const width = Math.max(2, Number(viewport.width) || 1280)
    const height = Math.max(2, Number(viewport.height) || 720)
    await this.cdp(wc, 'Input.dispatchMouseEvent', {
      type: 'mouseWheel',
      x: Math.round(width / 2),
      y: Math.round(height / 2),
      deltaX: 0,
      deltaY: sign * Math.max(400, Math.round(height * 0.75))
    })
  }

  private async pressKey(entry: BrowserEntry, key: string): Promise<void> {
    if (!key) {
      throw new Error('key_required')
    }
    const wc = entry.view.webContents
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'rawKeyDown', key })
    await this.cdp(wc, 'Input.dispatchKeyEvent', { type: 'keyUp', key })
  }

  private async imagesForEntry(entry: BrowserEntry): Promise<Record<string, unknown>> {
    const images = (await entry.view.webContents.executeJavaScript(
      `(function () {
      return Array.from(document.images).slice(0, 250).map(function (img) {
        return { src: img.currentSrc || img.src || '', alt: img.alt || '', width: img.naturalWidth || 0, height: img.naturalHeight || 0 };
      }).filter(function (img) { return !!img.src; });
    })()`,
      true
    )) as Array<Record<string, unknown>>

    return { success: true, runtime: 'electron-chromium', task_id: entry.ownerTaskId, images, count: images.length }
  }

  private async consoleForEntry(entry: BrowserEntry, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const expression = typeof args.expression === 'string' ? args.expression : ''

    if (!expression) {
      return {
        success: true,
        runtime: 'electron-chromium',
        task_id: entry.ownerTaskId,
        messages: [],
        note: 'Historical console capture is not enabled in Workstation V1 foundation; expression evaluation is available.'
      }
    }

    const result = await entry.view.webContents.executeJavaScript(expression, true)

    return { success: true, runtime: 'electron-chromium', task_id: entry.ownerTaskId, result }
  }

  private async screenshotForEntry(entry: BrowserEntry): Promise<Record<string, unknown>> {
    fs.mkdirSync(screenshotDirectory(), { recursive: true })
    const image = await entry.view.webContents.capturePage()
    const filePath = path.join(screenshotDirectory(), `browser-${Date.now()}-${entry.id.slice(0, 8)}.png`)
    fs.writeFileSync(filePath, image.toPNG())

    return {
      success: true,
      runtime: 'electron-chromium',
      task_id: entry.ownerTaskId,
      screenshot_path: filePath,
      note: 'Screenshot captured locally. Use the normal Hermes vision pipeline when visual interpretation is required.'
    }
  }

  private ensureSession(): void {
    if (this.browserSession) {
      return
    }
    const profilePath = workstationBrowserProfilePath()
    fs.mkdirSync(profilePath, { recursive: true })
    this.browserSession = session.fromPath(profilePath, { cache: true })

    if (typeof this.browserSession.setUserAgent === 'function') {
      this.browserSession.setUserAgent(getStandardChromeUserAgent())
    }

    if (this.browserSession.webRequest?.onBeforeSendHeaders) {
      this.browserSession.webRequest.onBeforeSendHeaders((details, callback) => {
        const headers = { ...details.requestHeaders }
        headers['User-Agent'] = getStandardChromeUserAgent()
        callback({ requestHeaders: headers })
      })
    }

    this.browserSession.on?.('will-download', (_event: unknown, item: any) => {
      const filename = typeof item.getFilename === 'function' ? item.getFilename() : 'download'

      const downloadInfo: WorkstationDownloadItem = {
        id: `${Date.now()}-${filename}`,
        filename,
        savePath: typeof item.getSavePath === 'function' ? item.getSavePath() : '',
        totalBytes: typeof item.getTotalBytes === 'function' ? item.getTotalBytes() : 0,
        receivedBytes: 0,
        state: 'progressing',
        url: typeof item.getURL === 'function' ? item.getURL() : ''
      }

      this.downloads.unshift(downloadInfo)

      if (this.downloads.length > 20) {
        this.downloads.pop()
      }
      this.emitState()

      item.on?.('updated', (_evt: unknown, state: string) => {
        downloadInfo.receivedBytes =
          typeof item.getReceivedBytes === 'function' ? item.getReceivedBytes() : downloadInfo.receivedBytes
        downloadInfo.savePath = typeof item.getSavePath === 'function' ? item.getSavePath() : downloadInfo.savePath

        if (state === 'interrupted') {
          downloadInfo.state = 'interrupted'
        }

        this.emitState()
      })

      item.once?.('done', (_evt: unknown, state: string) => {
        downloadInfo.state = state === 'completed' ? 'completed' : 'cancelled'
        downloadInfo.savePath = typeof item.getSavePath === 'function' ? item.getSavePath() : downloadInfo.savePath
        downloadInfo.receivedBytes =
          typeof item.getReceivedBytes === 'function' ? item.getReceivedBytes() : downloadInfo.receivedBytes
        this.emitState()
      })
    })

    void this.loadInstalledExtensions()

    this.cacheTimer = setInterval(() => {
      void this.cleanupCache(false).catch(error => this.recordError(error))
    }, CACHE_CHECK_INTERVAL_MS)
    this.cacheTimer.unref?.()
    setTimeout(() => void this.cleanupCache(false).catch(error => this.recordError(error)), 5_000).unref?.()
  }

  private extensionRoot(): string {
    const hermesHome = process.env.HERMES_HOME || path.join(os.homedir(), '.hermes')
    return path.resolve(hermesHome, 'workstation', 'extensions')
  }

  private extensionInfo(extensionId: string): any | null {
    const all = (this.browserSession as any)?.getAllExtensions?.()
    return all && typeof all === 'object' ? all[extensionId] ?? null : null
  }

  private isLoadedExtensionUrl(url: string): boolean {
    try {
      const parsed = new URL(url)
      return parsed.protocol === 'chrome-extension:' && Boolean(this.extensionInfo(parsed.hostname))
    } catch {
      return false
    }
  }

  private async loadExtensionForController(extensionId: string, extensionPath: string): Promise<Record<string, unknown>> {
    const id = extensionId.trim().toLowerCase()
    if (!/^[a-p]{32}$/.test(id)) {
      throw new Error('invalid_extension_id')
    }
    if (!this.browserSession?.loadExtension) {
      throw new Error('extension_loading_unavailable')
    }
    const root = this.extensionRoot()
    const expectedPath = path.resolve(root, id)
    if (path.resolve(extensionPath) !== expectedPath || !expectedPath.startsWith(`${root}${path.sep}`)) {
      throw new Error('extension_path_outside_workstation_store')
    }
    if (!fs.existsSync(path.join(expectedPath, 'manifest.json'))) {
      throw new Error('extension_manifest_missing')
    }
    const extension = await this.browserSession.loadExtension(expectedPath, { allowFileAccess: true })
    const loadedId = String((extension as any)?.id ?? '')
    if (loadedId && loadedId !== id) {
      ;(this.browserSession as any).removeExtension?.(loadedId)
      throw new Error('extension_id_mismatch')
    }
    return this.verifyExtensionForController(id)
  }

  private verifyExtensionForController(extensionId: string): Record<string, unknown> {
    const id = extensionId.trim().toLowerCase()
    if (!/^[a-p]{32}$/.test(id)) {
      throw new Error('invalid_extension_id')
    }
    const extension = this.extensionInfo(id)
    return {
      extension_id: id,
      loaded: Boolean(extension),
      name: String(extension?.name ?? ''),
      version: String(extension?.version ?? '')
    }
  }

  private removeExtensionForController(extensionId: string): Record<string, unknown> {
    const id = extensionId.trim().toLowerCase()
    if (!/^[a-p]{32}$/.test(id)) {
      throw new Error('invalid_extension_id')
    }
    const loaded = Boolean(this.extensionInfo(id))
    if (loaded) {
      ;(this.browserSession as any)?.removeExtension?.(id)
    }
    return { extension_id: id, removed: loaded, loaded: false }
  }

  private async loadInstalledExtensions(): Promise<void> {
    if (!this.browserSession?.loadExtension) {
      return
    }
    const extensionsDir = this.extensionRoot()

    if (!fs.existsSync(extensionsDir)) {
      return
    }

    try {
      const entries = fs.readdirSync(extensionsDir, { withFileTypes: true })

      for (const entry of entries) {
        if (entry.isDirectory()) {
          const extPath = path.join(extensionsDir, entry.name)
          const manifestPath = path.join(extPath, 'manifest.json')

          if (fs.existsSync(manifestPath)) {
            this.loadExtensionForController(entry.name, extPath).catch(err => {
              console.warn(`[workstation-browser] Failed to load extension ${entry.name}:`, err)
            })
          }
        }
      }
    } catch (error) {
      console.warn('[workstation-browser] Failed to enumerate extensions:', error)
    }
  }

  private wireEntry(entry: BrowserEntry): void {
    const wc = entry.view.webContents
    wc.setWindowOpenHandler(details => {
      if (permittedTopLevelUrl(details.url) || this.isLoadedExtensionUrl(details.url)) {
        const shouldActivate = this.activeTabId === entry.id
        // createTab is idempotent for ownerTaskId, so a task-owned popup is
        // redirected into the same live page instead of creating a second owner.
        this.createTab(details.url, shouldActivate, entry.ownerTaskId)
      }

      return { action: 'deny' }
    })

    const guardTopLevelNavigation = (event: { preventDefault: () => void }, url: string): void => {
      if (!permittedTopLevelUrl(url) && !this.isLoadedExtensionUrl(url)) {
        event.preventDefault()
        this.lastError = `Blocked unsafe top-level navigation: ${url}`
        this.emitState()
      }
    }

    wc.on('will-navigate', guardTopLevelNavigation)
    wc.on('will-redirect', guardTopLevelNavigation)
    wc.on('did-start-loading', () => {
      entry.loading = true
      this.emitState()
    })
    wc.on('did-stop-loading', () => {
      entry.loading = false
      this.emitState()
    })

    const refreshStructuralMetadata = (): void => {
      this.updateEntrySafeMetadata(entry, wc.getURL(), wc.getTitle())
      this.persistBrowserSessionState()
      this.emitState()
    }

    wc.on('did-navigate', refreshStructuralMetadata)
    wc.on('did-navigate-in-page', refreshStructuralMetadata)
    wc.on('page-title-updated', refreshStructuralMetadata)
    wc.on('render-process-gone', (_event, details) => {
      this.updateEntrySafeMetadata(entry, wc.getURL(), wc.getTitle())
      entry.crashed = true
      entry.recoveryState = 'stale'
      entry.recoveryReason = 'page-gone'
      this.lastError = `Browser renderer exited: ${details.reason}`
      this.persistBrowserSessionState()
      this.emitState()
    })
    wc.on('destroyed', () => {
      if (this.entries.get(entry.id) === entry) {
        this.rememberPendingSessionTab(entry, 'stale', 'page-gone')
        this.entries.delete(entry.id)

        if (entry.ownerTaskId && this.taskTabs.get(entry.ownerTaskId) === entry.id) {
          this.taskTabs.delete(entry.ownerTaskId)
        }

        if (this.activeTabId === entry.id) {
          this.activeTabId = null
        }
        this.persistBrowserSessionState()
        this.emitState()
      }
    })
  }

  private activeEntry(): BrowserEntry | null {
    return this.activeTabId ? (this.entries.get(this.activeTabId) ?? null) : null
  }

  private discardEntry(entry: BrowserEntry): void {
    const wasActive = this.activeTabId === entry.id

    if (entry.ownerTaskId && !this.pendingSessionTabs.has(entry.id)) {
      this.rememberPendingSessionTab(entry, 'stale', 'page-gone')
    }

    if (wasActive && this.attached) {
      this.detachActiveView(false)
    }
    this.removeChildView(entry)

    if (entry.ownerTaskId && this.taskTabs.get(entry.ownerTaskId) === entry.id) {
      this.taskTabs.delete(entry.ownerTaskId)
    }

    this.entries.delete(entry.id)

    if (!this.pendingSessionTabs.has(entry.id)) {
      this.restoredTabOrder = this.restoredTabOrder.filter(id => id !== entry.id)
    }

    if (wasActive) {
      this.activeTabId = null
    }

    if (!entry.view.webContents.isDestroyed()) {
      entry.view.webContents.close()
    }
  }

  private tabState(entry: BrowserEntry): WorkstationBrowserTabState {
    const wc = entry.view.webContents

    if (wc.isDestroyed()) {
      return {
        id: entry.id,
        title: 'Crashed tab',
        url: '',
        active: entry.id === this.activeTabId,
        loading: false,
        canGoBack: false,
        canGoForward: false,
        crashed: true,
        ownerTaskId: entry.ownerTaskId
      }
    }

    const history = historyState(wc)

    return {
      id: entry.id,
      title: wc.getTitle() || entry.safeTitle || 'New Tab',
      url: wc.getURL() || entry.safeUrl || 'about:blank',
      active: entry.id === this.activeTabId,
      loading: entry.loading,
      canGoBack: history.canGoBack,
      canGoForward: history.canGoForward,
      crashed: entry.crashed,
      ownerTaskId: entry.ownerTaskId
    }
  }

  private updateEntrySafeMetadata(entry: BrowserEntry, rawUrl: string, rawTitle: string): void {
    const safeUrl = safeRestorableUrlMetadata(rawUrl)
    const safeTitle = safeTitleMetadata(rawTitle)
    const urlWasSanitized = Boolean(rawUrl) && safeUrl !== rawUrl
    entry.safeUrl = safeUrl
    entry.safeTitle = safeTitle
    entry.recoveryState = entry.crashed ? 'stale' : 'live'
    entry.recoveryReason = urlWasSanitized ? 'unsafe-metadata' : null
  }

  private sessionTabFromEntry(entry: BrowserEntry): BrowserSessionTab {
    return {
      id: entry.id,
      browserTaskId: entry.ownerTaskId,
      safeUrl: entry.safeUrl,
      safeTitle: entry.safeTitle,
      recoveryPolicy: entry.ownerTaskId
        ? 'browser-task-lazy'
        : entry.safeUrl
          ? 'restore-safe-url'
          : 'restore-about-blank',
      recoveryState: entry.recoveryState,
      recoveryReason: entry.recoveryReason
    }
  }

  private rememberPendingSessionTab(
    entry: BrowserEntry,
    recoveryState: BrowserSessionTabRecoveryState,
    recoveryReason: BrowserSessionTabRecoveryReason
  ): void {
    const pending = {
      ...this.sessionTabFromEntry(entry),
      recoveryState,
      recoveryReason
    }

    this.pendingSessionTabs.set(entry.id, pending)

    if (!this.restoredTabOrder.includes(entry.id)) {
      this.restoredTabOrder.push(entry.id)
    }
  }

  private pendingTabForTask(taskId: string): BrowserSessionTab | null {
    for (const tab of this.pendingSessionTabs.values()) {
      if (tab.browserTaskId === taskId) {
        return tab
      }
    }

    return null
  }

  private removePendingTaskTab(taskId: string): void {
    for (const [id, tab] of this.pendingSessionTabs) {
      if (tab.browserTaskId !== taskId) {
        continue
      }
      this.pendingSessionTabs.delete(id)
      this.restoredTabOrder = this.restoredTabOrder.filter(candidate => candidate !== id)

      if (this.restoredLogicalActiveTabId === id) {
        this.restoredLogicalActiveTabId = null
      }
    }

    this.reconcileRestoredEntryOrder()
  }

  private reconcileRestoredEntryOrder(): void {
    if (this.restoredTabOrder.length === 0) {
      return
    }
    const reordered = new Map<string, BrowserEntry>()

    for (const id of this.restoredTabOrder) {
      const entry = this.entries.get(id)

      if (entry) {
        reordered.set(id, entry)
      }
    }

    for (const [id, entry] of this.entries) {
      if (!reordered.has(id)) {
        reordered.set(id, entry)
      }
    }

    this.entries = reordered

    if (this.pendingSessionTabs.size === 0 && !this.browserSessionStateRestoring) {
      this.restoredTabOrder = []
    }
  }

  private sessionTabsSnapshot(): BrowserSessionTab[] {
    if (this.pendingSessionTabs.size === 0) {
      return [...this.entries.values()].map(entry => this.sessionTabFromEntry(entry))
    }

    const tabs: BrowserSessionTab[] = []
    const included = new Set<string>()

    for (const id of this.restoredTabOrder) {
      const entry = this.entries.get(id)
      const tab = entry ? this.sessionTabFromEntry(entry) : this.pendingSessionTabs.get(id)

      if (!tab || included.has(id)) {
        continue
      }
      tabs.push({ ...tab })
      included.add(id)
    }

    for (const entry of this.entries.values()) {
      if (included.has(entry.id)) {
        continue
      }
      tabs.push(this.sessionTabFromEntry(entry))
      included.add(entry.id)
    }

    for (const [id, tab] of this.pendingSessionTabs) {
      if (included.has(id)) {
        continue
      }
      tabs.push({ ...tab })
    }

    return tabs
  }

  private persistBrowserSessionState(): void {
    if (this.browserSessionPersistenceSuppressed || !this.browserSessionStateRestored || !this.browserSessionState) {
      return
    }

    try {
      const tabs = this.sessionTabsSnapshot()

      const logicalActiveTabId =
        this.restoredLogicalActiveTabId && tabs.some(tab => tab.id === this.restoredLogicalActiveTabId)
          ? this.restoredLogicalActiveTabId
          : null

      const activeTabId =
        logicalActiveTabId ??
        (this.activeTabId && tabs.some(tab => tab.id === this.activeTabId) ? this.activeTabId : null)

      this.browserSessionState.saveSession(tabs, activeTabId)
    } catch (error) {
      this.lastError = `BrowserSessionState persistence failed: ${error instanceof Error ? error.message : String(error)}`
    }
  }

  private withBrowserSessionProjectionSuppressed<Result>(operation: () => Result): Result {
    const previous = this.browserSessionPersistenceSuppressed
    this.browserSessionPersistenceSuppressed = true

    try {
      return operation()
    } finally {
      this.browserSessionPersistenceSuppressed = previous
    }
  }

  private bindViewportGeometry(window: BrowserWindow): void {
    if (this.viewportGeometryWindow === window) {
      return
    }

    const emitter = window as unknown as {
      on?: (event: string, listener: () => void) => void
    }

    if (typeof emitter.on !== 'function') {
      return
    }

    this.unbindViewportGeometry()
    const listener = (): void => this.reconcileViewportGeometry()
    emitter.on('resize', listener)
    emitter.on('maximize', listener)
    emitter.on('unmaximize', listener)
    emitter.on('restore', listener)
    this.viewportGeometryWindow = window
    this.viewportGeometryListener = listener
  }

  private unbindViewportGeometry(): void {
    if (!this.viewportGeometryWindow || !this.viewportGeometryListener) {
      this.viewportGeometryWindow = null
      this.viewportGeometryListener = null

      return
    }

    const emitter = this.viewportGeometryWindow as unknown as {
      off?: (event: string, listener: () => void) => void
    }

    if (typeof emitter.off === 'function') {
      emitter.off('resize', this.viewportGeometryListener)
      emitter.off('maximize', this.viewportGeometryListener)
      emitter.off('unmaximize', this.viewportGeometryListener)
      emitter.off('restore', this.viewportGeometryListener)
    }
    this.viewportGeometryWindow = null
    this.viewportGeometryListener = null
  }

  private reconcileViewportGeometry(): void {
    if (
      !this.attached ||
      !this.ownerWindow ||
      this.ownerWindow.isDestroyed() ||
      (typeof this.ownerWindow.isMinimized === 'function' && this.ownerWindow.isMinimized()) ||
      !this.bounds
    ) {
      return
    }

    const entry = this.activeEntry()
    const bounds = this.validBounds(this.ownerWindow, this.bounds)

    if (!entry || !bounds) {
      return
    }

    this.bounds = bounds

    try {
      entry.view.setBounds(bounds)
    } catch {
      // The native view may be in teardown while the BrowserWindow emits its
      // final geometry event; the next attach will reconcile it again.
    }
  }

  private ensureChildView(window: BrowserWindow, view: WebContentsView): void {
    if (window.contentView.children.includes(view)) {
      return
    }
    window.contentView.addChildView(view)
  }

  private removeChildView(entry: BrowserEntry): void {
    const window = this.ownerWindow

    if (!window || window.isDestroyed()) {
      return
    }

    if (!window.contentView.children.includes(entry.view)) {
      return
    }

    try {
      window.contentView.removeChildView(entry.view)
    } catch {
      // View may already be detached/destroyed.
    }
  }

  private applyFrameRate(entry: BrowserEntry, visible: boolean): void {
    try {
      entry.view.webContents.setFrameRate(visible ? DEFAULT_VISIBLE_FRAME_RATE : backgroundFrameRate())
    } catch {
      // Frame-rate throttling is an optimization; control remains functional.
    }
  }

  private parkEntry(entry: BrowserEntry): void {
    // Keeping task-owned WebContentsView attached at the edge of the compositor
    // avoids background-rendering/screenshot stalls while exposing only a 1px
    // sliver. This mirrors the proven parking pattern in browser-use/desktop.
    if (!entry.ownerTaskId) {
      this.applyFrameRate(entry, false)

      return
    }

    const window = this.ownerWindow ?? BrowserWindow.getAllWindows().find(candidate => !candidate.isDestroyed()) ?? null

    if (!window || window.isDestroyed()) {
      this.applyFrameRate(entry, false)

      return
    }

    this.ownerWindow = window
    this.ensureChildView(window, entry.view)
    const content = window.getContentBounds()
    const width = Math.max(1, this.bounds?.width ?? DEFAULT_BROWSER_WIDTH)
    const height = Math.max(1, this.bounds?.height ?? DEFAULT_BROWSER_HEIGHT)
    entry.view.setBounds({
      x: Math.max(0, content.width - 1),
      y: Math.max(0, content.height - 1),
      width,
      height
    })
    this.applyFrameRate(entry, false)
  }

  private detachActiveView(park: boolean): void {
    if (!this.attached || !this.ownerWindow || this.ownerWindow.isDestroyed()) {
      this.attached = false

      return
    }

    const entry = this.activeEntry()

    if (entry) {
      this.removeChildView(entry)

      if (park) {
        this.parkEntry(entry)
      }
    }

    this.attached = false
  }

  private validBounds(window: BrowserWindow | null, bounds: WorkstationBrowserBounds): WorkstationBrowserBounds | null {
    const finite = [bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite)

    if (!finite || bounds.width < 1 || bounds.height < 1) {
      return null
    }

    const zoom =
      window && !window.isDestroyed() && window.webContents && typeof window.webContents.zoomFactor === 'number'
        ? window.webContents.zoomFactor
        : 1

    const content = window && !window.isDestroyed() ? window.getContentBounds() : null
    const x = Math.max(0, Math.round(bounds.x * zoom))
    const y = Math.max(0, Math.round(bounds.y * zoom))
    const contentWidth = content ? Math.max(1, Math.round(content.width)) : null
    const contentHeight = content ? Math.max(1, Math.round(content.height)) : null
    const boundedX = contentWidth === null ? x : Math.min(x, contentWidth - 1)
    const boundedY = contentHeight === null ? y : Math.min(y, contentHeight - 1)
    const width = Math.max(1, Math.round(bounds.width * zoom))
    const height = Math.max(1, Math.round(bounds.height * zoom))

    return {
      x: boundedX,
      y: boundedY,
      width: contentWidth === null ? width : Math.min(width, contentWidth - boundedX),
      height: contentHeight === null ? height : Math.min(height, contentHeight - boundedY)
    }
  }

  private async refreshCacheSize(): Promise<void> {
    if (!this.browserSession) {
      return
    }

    try {
      this.cacheBytes = await this.browserSession.getCacheSize()
      this.emitState()
    } catch {
      // Metrics are non-critical.
    }
  }

  private recordError(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error)
    this.lastError = message
    this.emitState()

    if (
      message === 'stale_or_unknown_ref' ||
      message === 'element_not_visible' ||
      message === 'element_unavailable' ||
      message === 'ref_required'
    ) {
      setTimeout(() => {
        if (this.lastError === message) {
          this.lastError = null
          this.emitState()
        }
      }, 7_000).unref?.()
    }
  }

  private emitState(): void {
    const state = this.state()

    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send('hermes:workstation-browser:state', state)
      }
    }
  }
}

let runtime: WorkstationBrowserRuntime | null = null

export function getWorkstationBrowserRuntime(): WorkstationBrowserRuntime {
  if (!runtime) {
    runtime = new WorkstationBrowserRuntime()
  }

  return runtime
}

function senderWindow(event: Electron.IpcMainInvokeEvent): BrowserWindow {
  const window = BrowserWindow.fromWebContents(event.sender)

  if (!window) {
    throw new Error('Hermes Browser IPC sender is not a BrowserWindow.')
  }

  return window
}

function registerIpc(): void {
  ipcMain.handle('hermes:workstation-browser:status', () => getWorkstationBrowserRuntime().ensure())
  ipcMain.handle('hermes:workstation-browser:ensure', () => getWorkstationBrowserRuntime().ensure())
  ipcMain.handle('hermes:workstation-browser:resources', () => getWorkstationBrowserRuntime().resources())
  ipcMain.handle('hermes:workstation-browser:events', (_event, taskId, limit) =>
    getWorkstationBrowserRuntime().events(
      typeof taskId === 'string' ? taskId : null,
      typeof limit === 'number' ? limit : MAX_EVENTS_ENDPOINT
    )
  )
  ipcMain.handle('hermes:workstation-browser:new-tab', (_event, target) =>
    getWorkstationBrowserRuntime().createTab(String(target ?? 'about:blank'), true)
  )
  ipcMain.handle('hermes:workstation-browser:activate-tab', (_event, tabId) =>
    getWorkstationBrowserRuntime().activateTab(String(tabId ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:close-tab', (_event, tabId) =>
    getWorkstationBrowserRuntime().closeTab(String(tabId ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:navigate', async (_event, target) =>
    getWorkstationBrowserRuntime().navigate(String(target ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:back', () => getWorkstationBrowserRuntime().back())
  ipcMain.handle('hermes:workstation-browser:forward', () => getWorkstationBrowserRuntime().forward())
  ipcMain.handle('hermes:workstation-browser:reload', () => getWorkstationBrowserRuntime().reload())
  ipcMain.handle('hermes:workstation-browser:stop', () => getWorkstationBrowserRuntime().stop())
  ipcMain.handle('hermes:workstation-browser:focus', () => getWorkstationBrowserRuntime().focus())
  ipcMain.handle('hermes:workstation-browser:attach', (event, bounds, host, preferredTaskId) =>
    getWorkstationBrowserRuntime().attach(
      senderWindow(event),
      bounds as WorkstationBrowserBounds,
      typeof host === 'string' ? host : 'hub',
      typeof preferredTaskId === 'string' ? preferredTaskId : null
    )
  )
  ipcMain.handle('hermes:workstation-browser:set-bounds', (event, bounds, expectedHost) =>
    getWorkstationBrowserRuntime().setBounds(
      senderWindow(event),
      bounds as WorkstationBrowserBounds,
      typeof expectedHost === 'string' ? expectedHost : undefined
    )
  )
  ipcMain.handle('hermes:workstation-browser:detach', event =>
    getWorkstationBrowserRuntime().detach(senderWindow(event))
  )
  ipcMain.handle('hermes:workstation-browser:set-visible', (_event, visible) =>
    getWorkstationBrowserRuntime().setVisible(Boolean(visible))
  )
  ipcMain.handle('hermes:workstation-browser:clear-error', () => getWorkstationBrowserRuntime().clearError())
  ipcMain.handle('hermes:workstation-browser:transfer-viewport', (event, targetHost, bounds) =>
    getWorkstationBrowserRuntime().transferViewport(
      senderWindow(event),
      typeof targetHost === 'string' ? targetHost : 'hub',
      bounds as WorkstationBrowserBounds
    )
  )
  ipcMain.handle('hermes:workstation-browser:list-tasks', () => getWorkstationBrowserRuntime().listTasks())
  ipcMain.handle('hermes:workstation-browser:show-task', (event, taskId, bounds, host) =>
    getWorkstationBrowserRuntime().showTask(
      String(taskId ?? ''),
      senderWindow(event),
      bounds as WorkstationBrowserBounds,
      typeof host === 'string' ? host : 'hub'
    )
  )
  ipcMain.handle('hermes:workstation-browser:hide-task', (_event, taskId) =>
    getWorkstationBrowserRuntime().hideTask(String(taskId ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:park-task', (_event, taskId) =>
    getWorkstationBrowserRuntime().parkTask(String(taskId ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:destroy-task', (_event, taskId) =>
    getWorkstationBrowserRuntime().destroyTask(String(taskId ?? ''))
  )
  ipcMain.handle('hermes:workstation-browser:clear-parked-tasks', () =>
    getWorkstationBrowserRuntime().clearParkedTasks()
  )
  ipcMain.handle('hermes:workstation-browser:pause', () => getWorkstationBrowserRuntime().pause())
  ipcMain.handle('hermes:workstation-browser:resume', () => getWorkstationBrowserRuntime().resume())
  ipcMain.handle('hermes:workstation-browser:take-control', () => getWorkstationBrowserRuntime().takeControl())
  ipcMain.handle('hermes:workstation-browser:release-control', () => getWorkstationBrowserRuntime().releaseControl())
  ipcMain.handle('hermes:workstation-browser:cleanup-cache', (_event, force) =>
    getWorkstationBrowserRuntime().cleanupCache(Boolean(force))
  )
  ipcMain.handle('hermes:workstation-browser:task-journal', (_event, taskId) =>
    getWorkstationBrowserRuntime().getTaskJournal(String(taskId ?? ''))
  )
}

registerIpc()
void app
  .whenReady()
  .then(() => getWorkstationBrowserRuntime().startControlServer())
  .catch(error => {
    console.error('[workstation-browser] failed to start controller', error)
  })
app.on('before-quit', () => {
  void runtime?.destroy()
})
