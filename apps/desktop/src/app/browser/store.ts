import { atom, type WritableAtom } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'

export const BROWSER_PANE_ID = 'browser'
export const BROWSER_QC_PANE_ID = 'browser-qc'
export const BROWSER_QC_DIMENSIONS = [
  'composition',
  'color',
  'typography',
  'spacing',
  'contrast',
  'clipping',
  'referenceMatch'
] as const
export const BROWSER_QC_STATUSES = ['pass', 'fail', 'unchecked'] as const

export type BrowserQcDimension = (typeof BROWSER_QC_DIMENSIONS)[number]
export type BrowserQcStatus = (typeof BROWSER_QC_STATUSES)[number]

export interface BrowserQcDimensionState {
  status: BrowserQcStatus
  note: string
  evidence: string
}

export type BrowserQc = Record<BrowserQcDimension, BrowserQcDimensionState>

export interface BrowserTab {
  id: string
  url: string
  title: string
  pinned: boolean
  qc: BrowserQc
}

export interface BrowserState {
  activeTabId: null | string
  qcOpen: boolean
  tabs: BrowserTab[]
}

export interface BrowserTabInput {
  title?: string
  url?: string
  pinned?: boolean
}

export interface BrowserQcUpdateInput {
  evidence?: string
  note?: string
  status?: BrowserQcStatus
}

export interface BrowserCapture {
  captureId: string
  createdAt: number
  dataUrl: string
  height: number
  tabId: null | string
  width: number
}

export const BROWSER_TAB_LIMIT_ERROR_CODE = 'BROWSER_TAB_LIMIT'

export class BrowserTabLimitError extends Error {
  readonly code = BROWSER_TAB_LIMIT_ERROR_CODE

  constructor() {
    super('All browser tabs are pinned.')
    this.name = 'BrowserTabLimitError'
  }
}
export const BROWSER_UNSUPPORTED_URL_ERROR_CODE = 'BROWSER_UNSUPPORTED_URL'

export class BrowserUnsupportedUrlError extends Error {
  readonly code = BROWSER_UNSUPPORTED_URL_ERROR_CODE

  constructor() {
    super('Browser URL is unsupported.')
    this.name = 'BrowserUnsupportedUrlError'
  }
}

export const BROWSER_MAX_TABS = 8
export const BROWSER_URL_MAX_LENGTH = 2_048
export const BROWSER_TITLE_MAX_LENGTH = 256
export const BROWSER_QC_NOTE_MAX_LENGTH = 1_024
export const BROWSER_QC_EVIDENCE_MAX_LENGTH = 2_048
export const BROWSER_RUNTIME_IMAGE_MAX_LENGTH = 20 * 1024 * 1024

const BROWSER_STORAGE_PREFIX = 'hermes.desktop.browser.v1'
const EMPTY_BROWSER_STATE: BrowserState = { activeTabId: null, qcOpen: false, tabs: [] }
let nextTabId = 0

const clampText = (value: unknown, maxLength: number) =>
  typeof value === 'string' ? value.trim().slice(0, maxLength) : ''

const clampQcText = (value: unknown, maxLength: number) => {
  const text = clampText(value, maxLength)

  return /(?:data|blob):/i.test(text) ? '' : text
}

const clampRuntimeQcText = (value: unknown, maxLength: number) => {
  const text = typeof value === 'string' ? value.slice(0, maxLength) : ''

  return /(?:data|blob):/i.test(text) ? '' : text
}

const isBrowserQcStatus = (value: unknown): value is BrowserQcStatus =>
  typeof value === 'string' && (BROWSER_QC_STATUSES as readonly string[]).includes(value)

const isBrowserQcDimension = (value: string): value is BrowserQcDimension =>
  (BROWSER_QC_DIMENSIONS as readonly string[]).includes(value)

const createEmptyQc = (): BrowserQc =>
  Object.fromEntries(
    BROWSER_QC_DIMENSIONS.map(dimension => [dimension, { status: 'unchecked', note: '', evidence: '' }])
  ) as BrowserQc

const isTransientImageDataUrl = (url: string) => /^data:image\/(?:png|jpe?g|webp|gif|avif|bmp)(?:;[^,]*)?,/i.test(url)

/** Drops unsafe and non-navigable values before they can enter persisted state. */
export const sanitizeBrowserUrl = (value: unknown) => {
  const url = clampText(value, BROWSER_URL_MAX_LENGTH)

  if (!url || /^(?:data|blob):/i.test(url)) {
    return ''
  }

  try {
    const protocol = new URL(url).protocol

    return protocol === 'http:' || protocol === 'https:' || url === 'about:blank' ? url : ''
  } catch {
    return ''
  }
}

/** Retains an explicit runtime navigation source while bounding its memory use. */
export const normalizeBrowserRuntimeUrl = (value: unknown) => {
  const url = typeof value === 'string' ? value.trim() : ''

  if (isTransientImageDataUrl(url)) {
    return url.slice(0, BROWSER_RUNTIME_IMAGE_MAX_LENGTH)
  }

  return sanitizeBrowserUrl(url)
}

const hasUnsupportedExplicitBrowserUrl = (input: BrowserTabInput | undefined) => {
  if (!input || !Object.prototype.hasOwnProperty.call(input, 'url')) {
    return false
  }

  const value = input.url

  return (typeof value !== 'string' || value.trim() !== '') && !normalizeBrowserRuntimeUrl(value)
}

const sanitizeQc = (value: unknown): BrowserQc => {
  const source = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
  const qc = createEmptyQc()

  for (const [dimension, entry] of Object.entries(source)) {
    if (!isBrowserQcDimension(dimension) || !entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue
    }

    const candidate = entry as Record<string, unknown>
    qc[dimension] = {
      status: isBrowserQcStatus(candidate.status) ? candidate.status : 'unchecked',
      note: clampQcText(candidate.note, BROWSER_QC_NOTE_MAX_LENGTH),
      evidence: clampQcText(candidate.evidence, BROWSER_QC_EVIDENCE_MAX_LENGTH)
    }
  }

  return qc
}

/** Sanitizes untrusted stored state and removes non-persistable capture data. */
export const sanitizeBrowserState = (value: unknown): BrowserState => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { ...EMPTY_BROWSER_STATE, tabs: [] }
  }

  const source = value as Record<string, unknown>
  const seenIds = new Set<string>()
  const tabs: BrowserTab[] = []
  const rawTabs = Array.isArray(source.tabs) ? source.tabs : []

  for (const candidate of rawTabs) {
    if (tabs.length === BROWSER_MAX_TABS) {
      break
    }

    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
      continue
    }

    const rawTab = candidate as Record<string, unknown>
    const id = clampText(rawTab.id, 128)

    if (!id || seenIds.has(id)) {
      continue
    }

    seenIds.add(id)
    tabs.push({
      id,
      url: sanitizeBrowserUrl(rawTab.url),
      title: clampQcText(rawTab.title, BROWSER_TITLE_MAX_LENGTH),
      pinned: rawTab.pinned === true,
      qc: sanitizeQc(rawTab.qc)
    })
  }

  const activeTabId =
    typeof source.activeTabId === 'string' && tabs.some(tab => tab.id === source.activeTabId)
      ? source.activeTabId
      : (tabs[0]?.id ?? null)

  return { activeTabId, qcOpen: source.qcOpen === true, tabs }
}

const browserStateCodec: Codec<BrowserState> = {
  decode: raw => sanitizeBrowserState(JSON.parse(raw)),
  encode: value => JSON.stringify(sanitizeBrowserState(value))
}

const BROWSER_WINDOW_SCOPE_SESSION_KEY = 'hermes.desktop.browser.windowScope'

const browserWindowScopeToken = () => {
  if (typeof sessionStorage === 'undefined') {
    return 'window'
  }

  const existing = sessionStorage.getItem(BROWSER_WINDOW_SCOPE_SESSION_KEY)

  if (existing) {
    return existing
  }

  const token = createTabId()
  sessionStorage.setItem(BROWSER_WINDOW_SCOPE_SESSION_KEY, token)

  return token
}

export const browserWindowScope = (
  locationLike: Pick<Location, 'hash' | 'search'> | undefined = typeof window === 'undefined'
    ? undefined
    : window.location,
  windowToken = browserWindowScopeToken()
) => {
  if (!locationLike) {
    return 'primary'
  }

  const params = new URLSearchParams(locationLike.search)

  if (params.get('win') !== 'secondary') {
    return 'primary'
  }

  const route = locationLike.hash.replace(/^#\/?/, '').trim()

  if (route) {
    return `secondary.session.${encodeURIComponent(route)}`
  }

  return params.get('new') === '1'
    ? `secondary.new.${encodeURIComponent(windowToken)}`
    : `secondary.window.${encodeURIComponent(windowToken)}`
}

export const browserStorageKey = (scope: string, name: 'open' | 'state') =>
  `${BROWSER_STORAGE_PREFIX}.${encodeURIComponent(scope)}.${name}`

const createTabId = () => {
  nextTabId += 1

  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `browser-tab-${Date.now()}-${nextTabId}`
}

const createBrowserTab = (input: BrowserTabInput = {}): BrowserTab => ({
  id: createTabId(),
  url: normalizeBrowserRuntimeUrl(input.url),
  title: clampText(input.title, BROWSER_TITLE_MAX_LENGTH),
  pinned: input.pinned === true,
  qc: createEmptyQc()
})

export interface BrowserStore {
  $browserCapture: WritableAtom<BrowserCapture | null>
  $browserQcRevealRequest: WritableAtom<number>
  $browserRevealRequest: WritableAtom<number>
  $browserOpen: WritableAtom<boolean>
  $browserState: WritableAtom<BrowserState>
  addBrowserTab: (input?: BrowserTabInput) => null | string
  clearBrowserCapture: () => void
  closeBrowserTab: (tabId: string) => void
  openBrowserQc: (input?: BrowserTabInput) => null | string
  openBrowserSurface: (input?: BrowserTabInput) => null | string
  setBrowserActiveTab: (tabId: string) => void
  setBrowserCapture: (capture: BrowserCapture | null) => void
  setBrowserOpen: (open: boolean) => void
  setBrowserQcOpen: (open: boolean) => void
  toggleBrowserTabPin: (tabId: string) => void
  updateBrowserQc: (tabId: string, dimension: BrowserQcDimension, update: BrowserQcUpdateInput) => void
  updateBrowserTab: (tabId: string, update: BrowserTabInput) => void
}

/** Creates one isolated presentation store per stable renderer-window scope. */
export const createBrowserStore = (scope = browserWindowScope()): BrowserStore => {
  const $browserOpen = persistentAtom(browserStorageKey(scope, 'open'), false, {
    decode: raw => raw === 'true',
    encode: value => String(value)
  })

  const $browserState = persistentAtom(browserStorageKey(scope, 'state'), EMPTY_BROWSER_STATE, browserStateCodec)
  // Capture previews may contain image bytes and intentionally never enter persistent state.
  const $browserCapture = atom<BrowserCapture | null>(null)
  // Explicit user/SDK open requests advance this transient counter so pane chrome
  // can front the Browser surface even when it is already open.
  const $browserRevealRequest = atom(0)
  const $browserQcRevealRequest = atom(0)

  const addBrowserTab = (input: BrowserTabInput = {}) => {
    const state = $browserState.get()
    const url = normalizeBrowserRuntimeUrl(input.url)
    const existing = url ? state.tabs.find(tab => tab.url === url) : undefined

    if (existing) {
      if (state.activeTabId !== existing.id) {
        $browserState.set({ ...state, activeTabId: existing.id })
      }

      return existing.id
    }

    const tab = createBrowserTab(input)

    if (state.tabs.length < BROWSER_MAX_TABS) {
      $browserState.set({ ...state, activeTabId: tab.id, tabs: [...state.tabs, tab] })

      return tab.id
    }

    const oldestUnpinnedIndex = state.tabs.findIndex(tab => !tab.pinned)

    if (oldestUnpinnedIndex < 0) {
      throw new BrowserTabLimitError()
    }

    $browserState.set({
      ...state,
      activeTabId: tab.id,
      tabs: [...state.tabs.filter((_, index) => index !== oldestUnpinnedIndex), tab]
    })

    return tab.id
  }

  const setBrowserOpen = (open: boolean) => $browserOpen.set(open)
  const requestBrowserReveal = () => $browserRevealRequest.set($browserRevealRequest.get() + 1)
  const requestBrowserQcReveal = () => $browserQcRevealRequest.set($browserQcRevealRequest.get() + 1)

  const setBrowserQcOpen = (open: boolean) => {
    const state = $browserState.get()

    if (state.qcOpen !== open) {
      $browserState.set({ ...state, qcOpen: open })
    }

    if (open) {
      requestBrowserQcReveal()
    }
  }

  const assertSupportedExplicitBrowserUrl = (input: BrowserTabInput | undefined) => {
    if (hasUnsupportedExplicitBrowserUrl(input)) {
      throw new BrowserUnsupportedUrlError()
    }
  }

  const openBrowserSurface = (input?: BrowserTabInput) => {
    assertSupportedExplicitBrowserUrl(input)
    setBrowserOpen(true)
    requestBrowserReveal()
    const state = $browserState.get()

    if (input || state.tabs.length === 0) {
      return addBrowserTab(input)
    }

    return state.activeTabId
  }

  const openBrowserQc = (input?: BrowserTabInput) => {
    const tabId = openBrowserSurface(input)
    setBrowserQcOpen(true)

    return tabId
  }

  const closeBrowserTab = (tabId: string) => {
    const state = $browserState.get()
    const index = state.tabs.findIndex(tab => tab.id === tabId)

    if (index < 0) {
      return
    }

    const tabs = state.tabs.filter(tab => tab.id !== tabId)

    const activeTabId =
      state.activeTabId === tabId ? (tabs[index]?.id ?? tabs[index - 1]?.id ?? null) : state.activeTabId

    $browserState.set({ ...state, activeTabId, tabs })
  }

  const setBrowserActiveTab = (tabId: string) => {
    const state = $browserState.get()

    if (state.activeTabId !== tabId && state.tabs.some(tab => tab.id === tabId)) {
      $browserState.set({ ...state, activeTabId: tabId })
    }
  }

  const updateBrowserTab = (tabId: string, update: BrowserTabInput) => {
    const state = $browserState.get()
    const tab = state.tabs.find(candidate => candidate.id === tabId)

    if (!tab) {
      return
    }

    const next: BrowserTab = {
      ...tab,
      ...(Object.prototype.hasOwnProperty.call(update, 'url') ? { url: normalizeBrowserRuntimeUrl(update.url) } : {}),
      ...(Object.prototype.hasOwnProperty.call(update, 'title')
        ? { title: clampText(update.title, BROWSER_TITLE_MAX_LENGTH) }
        : {}),
      ...(typeof update.pinned === 'boolean' ? { pinned: update.pinned } : {})
    }

    $browserState.set({ ...state, tabs: state.tabs.map(candidate => (candidate.id === tabId ? next : candidate)) })
  }

  const toggleBrowserTabPin = (tabId: string) => {
    const tab = $browserState.get().tabs.find(candidate => candidate.id === tabId)

    if (tab) {
      updateBrowserTab(tabId, { pinned: !tab.pinned })
    }
  }

  const updateBrowserQc = (tabId: string, dimension: BrowserQcDimension, update: BrowserQcUpdateInput) => {
    const state = $browserState.get()
    const tab = state.tabs.find(candidate => candidate.id === tabId)

    if (!tab) {
      return
    }

    const current = tab.qc[dimension]

    const qc = {
      ...tab.qc,
      [dimension]: {
        status: isBrowserQcStatus(update.status) ? update.status : current.status,
        note: Object.prototype.hasOwnProperty.call(update, 'note')
          ? clampRuntimeQcText(update.note, BROWSER_QC_NOTE_MAX_LENGTH)
          : current.note,
        evidence: Object.prototype.hasOwnProperty.call(update, 'evidence')
          ? clampRuntimeQcText(update.evidence, BROWSER_QC_EVIDENCE_MAX_LENGTH)
          : current.evidence
      }
    }

    $browserState.set({
      ...state,
      tabs: state.tabs.map(candidate => (candidate.id === tabId ? { ...tab, qc } : candidate))
    })
  }

  return {
    $browserCapture,
    $browserQcRevealRequest,
    $browserRevealRequest,
    $browserOpen,
    $browserState,
    addBrowserTab,
    clearBrowserCapture: () => $browserCapture.set(null),
    closeBrowserTab,
    openBrowserQc,
    openBrowserSurface,
    setBrowserActiveTab,
    setBrowserCapture: capture => $browserCapture.set(capture),
    setBrowserOpen,
    setBrowserQcOpen,
    toggleBrowserTabPin,
    updateBrowserQc,
    updateBrowserTab
  }
}

const browserStore = createBrowserStore()

export const $browserOpen = browserStore.$browserOpen
export const $browserState = browserStore.$browserState
export const $browserCapture = browserStore.$browserCapture
/** @internal Core pane controller signal; intentionally omitted from the public SDK. */
export const $browserRevealRequest = browserStore.$browserRevealRequest
/** @internal Core QC pane controller signal; intentionally omitted from the public SDK. */
export const $browserQcRevealRequest = browserStore.$browserQcRevealRequest
export const setBrowserOpen = browserStore.setBrowserOpen
export const openBrowserSurface = browserStore.openBrowserSurface
export const openBrowserQc = browserStore.openBrowserQc
export const addBrowserTab = browserStore.addBrowserTab
export const closeBrowserTab = browserStore.closeBrowserTab
export const setBrowserActiveTab = browserStore.setBrowserActiveTab
export const updateBrowserTab = browserStore.updateBrowserTab
export const toggleBrowserTabPin = browserStore.toggleBrowserTabPin
export const updateBrowserQc = browserStore.updateBrowserQc
export const setBrowserQcOpen = browserStore.setBrowserQcOpen
export const setBrowserCapture = browserStore.setBrowserCapture
export const clearBrowserCapture = browserStore.clearBrowserCapture
