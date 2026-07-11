import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'

import type { SetTitlebarToolGroup, TitlebarTool } from '@/app/shell/titlebar-controls'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { Bug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $browserCurrentState,
  $browserDriveCommand,
  $browserSessionId,
  type BrowserDomActionPayload,
  normalizeBrowserUrl,
  setBrowserSessionState
} from '@/store/browser'

import { compactUrl } from './preview-console'
import { PreviewEmptyState } from './preview-file'

const AGENT_ACTION_LABELS = {
  act: 'Act',
  goBack: 'Back',
  goForward: 'Forward',
  navigate: 'Navigate',
  open: 'Open',
  reload: 'Reload',
  snapshot: 'Snapshot'
}

type BrowserWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  closeDevTools?: () => void
  executeJavaScript?: (code: string, userGesture?: boolean) => Promise<unknown>
  getTitle?: () => string
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  isDevToolsOpened?: () => boolean
  loadURL?: (url: string) => void
  openDevTools?: () => void
  reload?: () => void
  stop?: () => void
}

interface BrowserSnapshotElement {
  ariaLabel?: string
  hermesRef?: string
  href?: string
  id?: string
  index: number
  name?: string
  placeholder?: string
  role?: string
  tag: string
  text?: string
  type?: string
  value?: string
  visible?: boolean
}

interface BrowserSnapshotHeading {
  level: number
  text: string
}

interface BrowserSnapshotTable {
  caption?: string
  headers: string[]
  rows: string[][]
}

interface BrowserSnapshot {
  capturedAt: number
  elements: BrowserSnapshotElement[]
  headings: BrowserSnapshotHeading[]
  ok: boolean
  sessionId: string
  tables: BrowserSnapshotTable[]
  text: string
  title: string
  url: string
  error?: string
}

interface BrowserActionTarget {
  ariaLabel?: string
  hermesRef?: string
  id?: string
  index?: number
  name?: string
  role?: string
  tag?: string
  text?: string
  value?: string
}

interface BrowserActionResult {
  action: BrowserDomActionPayload['kind']
  capturedAt: number
  ok: boolean
  sessionId: string
  target?: BrowserActionTarget
  title?: string
  url?: string
  error?: string
}

interface BrowserPaneProps {
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

interface BrowserLoadErrorState {
  code?: number
  description: string
  url: string
}

const TITLEBAR_GROUP_ID = 'browser'
const BROWSER_PARTITION = 'persist:hermes-browser'

function browserPartition(): string {
  return BROWSER_PARTITION
}

export const BROWSER_SNAPSHOT_SCRIPT = String.raw`(() => {
  const MAX_TEXT = 200000
  const MAX_ELEMENTS = 500
  const MAX_TABLES = 30
  const MAX_ROWS = 80
  const MAX_CELL_TEXT = 500

  const clean = value => String(value || '').replace(/\s+/g, ' ').trim()
  const clip = (value, max = 4000) => clean(value).slice(0, max)
  const snapshotId = globalThis.crypto?.randomUUID?.() || (Date.now().toString(36) + Math.random().toString(36).slice(2))
  const isVisible = element => {
    const style = window.getComputedStyle(element)
    const rect = element.getBoundingClientRect()

    return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0
  }
  const elementText = element => {
    if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
      return element.placeholder || element.getAttribute('aria-label') || ''
    }

    return element.innerText || element.textContent || element.getAttribute('aria-label') || ''
  }

  const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'))
    .filter(isVisible)
    .slice(0, 100)
    .map(element => ({
      level: Number(element.tagName.slice(1)),
      text: clip(element.innerText || element.textContent || '', 1000)
    }))
    .filter(item => item.text)

  const semanticElements = Array.from(
    document.querySelectorAll('a,button,input,textarea,select,[role],[contenteditable="true"],summary,label,[tabindex],[data-e2e],[data-testid]')
  )
  const pointerElements = Array.from(document.querySelectorAll('div,span'))
    .filter(element => window.getComputedStyle(element).cursor === 'pointer' && Boolean(clean(elementText(element))))
  const elements = Array.from(new Set([...semanticElements, ...pointerElements]))
    .slice(0, MAX_ELEMENTS)
    .map((element, index) => {
      const hermesRef = snapshotId + ':' + index
      element.setAttribute('data-hermes-browser-ref', hermesRef)

      return {
        ariaLabel: clip(element.getAttribute('aria-label') || '', 1000) || undefined,
        hermesRef,
        href: element instanceof HTMLAnchorElement ? element.href || undefined : undefined,
        id: element.id || undefined,
        index,
        name: element.getAttribute('name') || undefined,
        placeholder: element.getAttribute('placeholder') || undefined,
        role: element.getAttribute('role') || undefined,
        tag: element.tagName.toLowerCase(),
        text: clip(elementText(element), 2000) || undefined,
        type: element.getAttribute('type') || undefined,
        visible: isVisible(element)
      }
    })
    .filter(item => item.visible || item.text || item.ariaLabel || item.placeholder)

  const tables = Array.from(document.querySelectorAll('table'))
    .filter(isVisible)
    .slice(0, MAX_TABLES)
    .map(table => {
      const headers = Array.from(table.querySelectorAll('th'))
        .slice(0, 40)
        .map(cell => clip(cell.innerText || cell.textContent || '', MAX_CELL_TEXT))
      const rows = Array.from(table.querySelectorAll('tr'))
        .slice(0, MAX_ROWS)
        .map(row =>
          Array.from(row.querySelectorAll('th,td'))
            .slice(0, 40)
            .map(cell => clip(cell.innerText || cell.textContent || '', MAX_CELL_TEXT))
        )
        .filter(row => row.some(Boolean))

      return {
        caption: clip(table.querySelector('caption')?.innerText || '', 1000) || undefined,
        headers,
        rows
      }
    })

  return {
    capturedAt: Date.now(),
    elements,
    headings,
    ok: true,
    tables,
    text: (document.body?.innerText || document.documentElement?.innerText || '').slice(0, MAX_TEXT),
    title: document.title || '',
    url: location.href
  }
})()`

export function browserActionScript(action: BrowserDomActionPayload): string {
  return `(async () => {
  const action = ${JSON.stringify(action)}
  const TARGET_SELECTOR = '[data-hermes-browser-ref],a,button,input,textarea,select,[role],[contenteditable="true"],summary,label,[tabindex],[data-e2e],[data-testid]'
  const clean = value => String(value || '').replace(/\\s+/g, ' ').trim()
  const clip = (value, max = 2000) => clean(value).slice(0, max)
  const elementText = element => {
    if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
      return element.placeholder || element.getAttribute('aria-label') || ''
    }

    return element.innerText || element.textContent || element.getAttribute('aria-label') || ''
  }
  const targetInfo = (element, index) => ({
    ariaLabel: element.getAttribute('aria-label') || undefined,
    hermesRef: element.getAttribute('data-hermes-browser-ref') || undefined,
    href: element instanceof HTMLAnchorElement ? element.href || undefined : undefined,
    id: element.id || undefined,
    index,
    name: element.getAttribute('name') || undefined,
    placeholder: element.getAttribute('placeholder') || undefined,
    role: element.getAttribute('role') || undefined,
    tag: element.tagName?.toLowerCase?.() || undefined,
    text: clip(elementText(element)),
    type: element.getAttribute('type') || undefined
  })
  const resolveTarget = () => {
    if (action.target && typeof action.target === 'object') {
      const fingerprint = Object.entries(action.target).filter(([, value]) => typeof value === 'string' && value)
      if (!fingerprint.length) return { element: null, error: 'Target fingerprint is empty', index: -1 }
      const elements = Array.from(document.querySelectorAll(TARGET_SELECTOR))
      const matches = elements
        .map((element, index) => ({ element, index, info: targetInfo(element, index) }))
        .filter(candidate => fingerprint.every(([key, value]) => candidate.info[key] === value))
      if (matches.length === 0) return { element: null, error: 'Target fingerprint was not found on the current page', index: -1 }
      if (matches.length > 1) return { element: null, error: 'Target fingerprint is ambiguous on the current page', index: -1 }
      return matches[0]
    }

    if (typeof action.selector === 'string' && action.selector.trim()) {
      const element = document.querySelector(action.selector)
      const index = Array.from(document.querySelectorAll(TARGET_SELECTOR)).indexOf(element)
      return { element, index }
    }

    if (Number.isInteger(action.index)) {
      const elements = Array.from(document.querySelectorAll(TARGET_SELECTOR))
      return { element: elements[action.index], index: action.index }
    }

    return { element: document.activeElement, index: -1 }
  }
  const emitInput = element => {
    element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText' }))
    element.dispatchEvent(new Event('change', { bubbles: true }))
  }
  const setTextValue = (element, value) => {
    if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
      const proto = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
      const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set
      setter ? setter.call(element, value) : (element.value = value)
      emitInput(element)
      return true
    }

    if (element instanceof HTMLElement && element.isContentEditable) {
      element.textContent = value
      emitInput(element)
      return true
    }

    return false
  }
  const uploadLocalFile = async value => {
    const fileInput = document.querySelector('input[type="file"]')
    if (!(fileInput instanceof HTMLInputElement)) return { error: 'Page has no file input' }
    const source = String(value || '').trim()
    const path = String(action.value || '').trim()
    if (!source.startsWith('data:') || !path.startsWith('/')) return { error: 'Local upload payload is invalid' }
    const response = await fetch(source)
    if (!response.ok) return { error: 'Unable to decode local file' }
    const blob = await response.blob()
    const mime = blob.type || 'application/octet-stream'
    const name = path.split('/').filter(Boolean).pop() || 'upload'
    const file = new File([blob], name, { type: mime })
    const transfer = new DataTransfer()
    transfer.items.add(file)
    fileInput.files = transfer.files
    emitInput(fileInput)
    return {}
  }
  const keyEvent = (type, key) => new KeyboardEvent(type, { bubbles: true, cancelable: true, key })
  const finish = (ok, target, extra = {}) => ({
    action: action.kind,
    capturedAt: Date.now(),
    ok,
    target,
    title: document.title || '',
    url: location.href,
    ...extra
  })

  if (typeof action.expectedUrl === 'string' && action.expectedUrl && location.href !== action.expectedUrl) {
    return finish(false, undefined, { error: 'Page URL changed after the snapshot; take a new browser_snapshot before acting' })
  }

  const { element, error, index } = resolveTarget()
  if (error) return finish(false, undefined, { error })
  if (!element) return finish(false, undefined, { error: 'Target element not found' })

  element.scrollIntoView?.({ block: 'center', inline: 'center' })
  element.focus?.()
  const target = targetInfo(element, index)

  if (action.kind === 'click') {
    element.click?.()
    return finish(true, target)
  }

  if (action.kind === 'type' || action.kind === 'setValue') {
    const next = String(action.text ?? action.value ?? '')
    if (element instanceof HTMLSelectElement) {
      const option = Array.from(element.options).find(item => item.value === next || clean(item.textContent) === clean(next))
      if (!option) return finish(false, target, { error: 'Select option was not found' })
      element.value = option.value
      emitInput(element)
      return finish(true, targetInfo(element, index))
    }
    if (!setTextValue(element, next)) {
      const label = clean(elementText(element)).toLowerCase()
      if (element instanceof HTMLButtonElement && (label.includes('上传') || label.includes('upload'))) {
        const uploaded = await uploadLocalFile(next)
        if (uploaded.error) return finish(false, target, { error: uploaded.error })
        return finish(true, target)
      }
      return finish(false, target, { error: 'Target is not editable' })
    }
    return finish(true, targetInfo(element, index))
  }

  if (action.kind === 'select') {
    if (!(element instanceof HTMLSelectElement)) return finish(false, target, { error: 'Target is not a select element' })
    element.value = String(action.value ?? '')
    emitInput(element)
    return finish(true, targetInfo(element, index))
  }

  if (action.kind === 'press') {
    const key = String(action.key || 'Enter')
    element.dispatchEvent(keyEvent('keydown', key))
    element.dispatchEvent(keyEvent('keyup', key))
    if (key === 'Enter' && element instanceof HTMLInputElement) element.form?.requestSubmit?.()
    return finish(true, target)
  }

  if (action.kind === 'scroll') {
    const amount = Number.isFinite(action.amount) ? Number(action.amount) : 600
    const direction = action.direction || 'down'
    const delta = {
      down: [0, amount],
      left: [-amount, 0],
      right: [amount, 0],
      up: [0, -amount]
    }[direction] || [0, amount]
    const scrollTarget = element === document.body || element === document.documentElement ? window : element
    if (scrollTarget === window) window.scrollBy({ left: delta[0], top: delta[1], behavior: 'instant' })
    else scrollTarget.scrollBy?.({ left: delta[0], top: delta[1], behavior: 'instant' })
    return finish(true, target)
  }

  return finish(false, target, { error: 'Unsupported browser action' })
})()`
}

function BrowserLoadError({ error, onRetry }: { error: BrowserLoadErrorState; onRetry: () => void }) {
  return (
    <PreviewEmptyState
      body={
        <>
          <div className="font-mono text-[0.6875rem] text-muted-foreground/90">{compactUrl(error.url)}</div>
          <div className="mt-1 text-[0.6875rem] text-muted-foreground/70">
            {error.description}
            {error.code ? ` (${error.code})` : ''}
          </div>
        </>
      }
      primaryAction={{ label: 'Try again', onClick: onRetry }}
      title="Page failed to load"
    />
  )
}

function navigateWebview(webview: BrowserWebview | null, url: string) {
  if (!webview) {
    return
  }

  if (webview.loadURL) {
    webview.loadURL(url)
  } else {
    webview.setAttribute('src', url)
  }
}

function emptyBrowserSnapshot(sessionId: string, error: string): BrowserSnapshot {
  return {
    capturedAt: Date.now(),
    elements: [],
    error,
    headings: [],
    ok: false,
    sessionId,
    tables: [],
    text: '',
    title: '',
    url: ''
  }
}

function normalizeBrowserSnapshot(raw: unknown, sessionId: string): BrowserSnapshot {
  if (!raw || typeof raw !== 'object') {
    return emptyBrowserSnapshot(sessionId, 'Snapshot returned no data')
  }

  const record = raw as Partial<BrowserSnapshot>

  return {
    capturedAt: typeof record.capturedAt === 'number' ? record.capturedAt : Date.now(),
    elements: Array.isArray(record.elements) ? record.elements : [],
    headings: Array.isArray(record.headings) ? record.headings : [],
    ok: record.ok !== false,
    sessionId,
    tables: Array.isArray(record.tables) ? record.tables : [],
    text: typeof record.text === 'string' ? record.text : '',
    title: typeof record.title === 'string' ? record.title : '',
    url: typeof record.url === 'string' ? record.url : '',
    ...(typeof record.error === 'string' ? { error: record.error } : {})
  }
}

function emptyBrowserActionResult(action: BrowserDomActionPayload, sessionId: string, error: string): BrowserActionResult {
  return {
    action: action.kind,
    capturedAt: Date.now(),
    error,
    ok: false,
    sessionId
  }
}

function normalizeBrowserActionResult(
  raw: unknown,
  action: BrowserDomActionPayload,
  sessionId: string
): BrowserActionResult {
  if (!raw || typeof raw !== 'object') {
    return emptyBrowserActionResult(action, sessionId, 'Action returned no data')
  }

  const record = raw as Partial<BrowserActionResult>

  return {
    action: action.kind,
    capturedAt: typeof record.capturedAt === 'number' ? record.capturedAt : Date.now(),
    ok: record.ok === true,
    sessionId,
    ...(record.target && typeof record.target === 'object' ? { target: record.target } : {}),
    ...(typeof record.title === 'string' ? { title: record.title } : {}),
    ...(typeof record.url === 'string' ? { url: record.url } : {}),
    ...(typeof record.error === 'string' ? { error: record.error } : {})
  }
}

export function BrowserPane({ setTitlebarToolGroup }: BrowserPaneProps) {
  const sessionId = useStore($browserSessionId)
  const browserState = useStore($browserCurrentState)
  const driveCommand = useStore($browserDriveCommand)
  const webviewRef = useRef<BrowserWebview | null>(null)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const lastHandledDriveCommandIdRef = useRef(0)
  const lastRequestedUrlRef = useRef(browserState.url)
  const [addressValue, setAddressValue] = useState(browserState.url)
  const [canGoBack, setCanGoBack] = useState(false)
  const [canGoForward, setCanGoForward] = useState(false)
  const [currentUrl, setCurrentUrl] = useState(browserState.url)
  const [devtoolsOpen, setDevtoolsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<BrowserLoadErrorState | null>(null)

  const publishBrowserState = useCallback(
    (patch: {
      canGoBack?: boolean
      canGoForward?: boolean
      loading?: boolean
      title?: string
      url?: string
    }) => {
      void window.hermesDesktop?.browser?.updateState?.({ ...patch, sessionId })
    },
    [sessionId]
  )

  const updateNavigationState = useCallback(() => {
    const webview = webviewRef.current
    const nextCanGoBack = Boolean(webview?.canGoBack?.())
    const nextCanGoForward = Boolean(webview?.canGoForward?.())

    setCanGoBack(nextCanGoBack)
    setCanGoForward(nextCanGoForward)
    publishBrowserState({ canGoBack: nextCanGoBack, canGoForward: nextCanGoForward })

    return { canGoBack: nextCanGoBack, canGoForward: nextCanGoForward }
  }, [publishBrowserState])

  const navigateTo = useCallback(
    (rawUrl: string) => {
      const url = normalizeBrowserUrl(rawUrl)

      lastRequestedUrlRef.current = url
      setAddressValue(url)
      setCurrentUrl(url)
      setLoadError(null)
      setBrowserSessionState(sessionId, { url })
      publishBrowserState({ loading: true, url })
      navigateWebview(webviewRef.current, url)
    },
    [publishBrowserState, sessionId]
  )

  const reload = useCallback(() => {
    setLoadError(null)
    publishBrowserState({ loading: true })
    webviewRef.current?.reload?.()
  }, [publishBrowserState])

  const completeSnapshotRequest = useCallback(
    async (requestId?: string) => {
      const webview = webviewRef.current
      let snapshot: BrowserSnapshot

      try {
        if (!webview?.executeJavaScript) {
          snapshot = emptyBrowserSnapshot(sessionId, 'Browser webview snapshot API is unavailable')
        } else {
          snapshot = normalizeBrowserSnapshot(await webview.executeJavaScript(BROWSER_SNAPSHOT_SCRIPT, false), sessionId)
        }
      } catch (error) {
        snapshot = emptyBrowserSnapshot(sessionId, error instanceof Error ? error.message : String(error))
      }

      void window.hermesDesktop?.browser?.completeSnapshot?.({ requestId, sessionId, snapshot })
    },
    [sessionId]
  )

  const completeActionRequest = useCallback(
    async (domAction: BrowserDomActionPayload | undefined, requestId?: string) => {
      const action = domAction?.kind ? domAction : ({ kind: 'click' } satisfies BrowserDomActionPayload)
      const webview = webviewRef.current
      let result: BrowserActionResult

      try {
        if (!domAction?.kind) {
          result = emptyBrowserActionResult(action, sessionId, 'Browser action payload is missing')
        } else if (!webview?.executeJavaScript) {
          result = emptyBrowserActionResult(action, sessionId, 'Browser webview action API is unavailable')
        } else {
          let executableAction = action
          const uploadLabel = String(action.target?.text || action.target?.ariaLabel || '').toLowerCase()
          const localPath = String(action.text || '')

          if (
            action.kind === 'type' &&
            action.target?.tag === 'button' &&
            (uploadLabel.includes('上传') || uploadLabel.includes('upload')) &&
            localPath.startsWith('/')
          ) {
            const dataUrl = await window.hermesDesktop?.readFileDataUrl?.(localPath)

            if (!dataUrl) {
              throw new Error('Unable to read local upload file')
            }

            executableAction = { ...action, text: dataUrl, value: localPath }
          }

          result = normalizeBrowserActionResult(
            await webview.executeJavaScript(browserActionScript(executableAction), true),
            action,
            sessionId
          )
        }
      } catch (error) {
        result = emptyBrowserActionResult(action, sessionId, error instanceof Error ? error.message : String(error))
      }

      void window.hermesDesktop?.browser?.completeAction?.({ requestId, result, sessionId })
    },
    [sessionId]
  )

  const toggleDevTools = useCallback(() => {
    const webview = webviewRef.current

    if (!webview?.openDevTools) {
      return
    }

    if (webview.isDevToolsOpened?.()) {
      webview.closeDevTools?.()
      setDevtoolsOpen(false)

      return
    }

    webview.openDevTools()
    setDevtoolsOpen(true)
  }, [])

  useEffect(() => {
    if (!setTitlebarToolGroup) {
      return
    }

    const tools: TitlebarTool[] = [
      {
        active: devtoolsOpen,
        icon: <Bug />,
        id: `${TITLEBAR_GROUP_ID}-devtools`,
        label: devtoolsOpen ? 'Hide browser DevTools' : 'Open browser DevTools',
        onSelect: toggleDevTools
      }
    ]

    setTitlebarToolGroup(TITLEBAR_GROUP_ID, tools)

    return () => setTitlebarToolGroup(TITLEBAR_GROUP_ID, [])
  }, [devtoolsOpen, setTitlebarToolGroup, toggleDevTools])

  useEffect(() => {
    setAddressValue(browserState.url)
    setCurrentUrl(browserState.url)
    setLoadError(null)

    const webviewUrl = webviewRef.current?.getURL?.()

    if (webviewUrl !== browserState.url && lastRequestedUrlRef.current !== browserState.url) {
      lastRequestedUrlRef.current = browserState.url
      navigateWebview(webviewRef.current, browserState.url)
    }
  }, [browserState.url, sessionId])

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    const initialUrl = $browserCurrentState.get().url
    const webview = document.createElement('webview') as BrowserWebview
    webview.className = 'flex h-full w-full flex-1 bg-white dark:bg-black'
    webview.setAttribute('partition', browserPartition())
    webview.setAttribute('src', initialUrl)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }
      const url = normalizeBrowserUrl(detail.url || webview.getURL?.() || initialUrl)

      lastRequestedUrlRef.current = url
      setLoadError(null)
      setCurrentUrl(url)
      setAddressValue(url)
      setBrowserSessionState(sessionId, { title: webview.getTitle?.() || undefined, url })
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, title: webview.getTitle?.() || undefined, url })
    }

    const onTitle = (event: Event) => {
      const detail = event as Event & { title?: string }
      const title = detail.title || webview.getTitle?.() || ''

      if (title) {
        setBrowserSessionState(sessionId, { title })
        publishBrowserState({ title })
      }
    }

    const onFail = (event: Event) => {
      const detail = event as Event & {
        errorCode?: number
        errorDescription?: string
        validatedURL?: string
      }

      if (detail.errorCode === -3) {
        return
      }

      const url = detail.validatedURL || webview.getURL?.() || lastRequestedUrlRef.current

      setLoadError({
        code: detail.errorCode,
        description: detail.errorDescription || 'The embedded browser could not load this page.',
        url
      })
      setLoading(false)
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, url })
    }

    const onStart = () => {
      setLoading(true)
      setLoadError(null)
      publishBrowserState({ loading: true, url: webview.getURL?.() || lastRequestedUrlRef.current })
    }

    const onStop = () => {
      setLoading(false)
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, title: webview.getTitle?.() || undefined, url: webview.getURL?.() })
    }

    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    webview.addEventListener('page-title-updated', onTitle)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.removeEventListener('page-title-updated', onTitle)
      webview.remove()
      webviewRef.current = null
    }
  }, [publishBrowserState, sessionId, updateNavigationState])

  // Listen for drive commands from the store. open/navigate are primarily
  // handled by the persisted URL state; reload/back/forward must touch the
  // live <webview> instance directly.
  useEffect(() => {
    if (
      !driveCommand ||
      driveCommand.sessionId !== sessionId ||
      driveCommand.id === lastHandledDriveCommandIdRef.current
    ) {
      return
    }

    lastHandledDriveCommandIdRef.current = driveCommand.id

    if (driveCommand.action === 'snapshot') {
      void completeSnapshotRequest(driveCommand.requestId)

      return
    }

    if (driveCommand.action === 'act') {
      void completeActionRequest(driveCommand.domAction, driveCommand.requestId)

      return
    }

    const webview = webviewRef.current

    if (!webview) {
      return
    }

    if (driveCommand.action === 'reload') {
      reload()
    } else if (driveCommand.action === 'goBack') {
      webview.goBack?.()
    } else if (driveCommand.action === 'goForward') {
      webview.goForward?.()
    }
  }, [completeActionRequest, completeSnapshotRequest, driveCommand, reload, sessionId])

  const activeDriveCommand = driveCommand?.sessionId === sessionId ? driveCommand : null
  const agentActionLabel = activeDriveCommand ? AGENT_ACTION_LABELS[activeDriveCommand.action] : null

  const submitAddress = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    navigateTo(addressValue)
  }

  const openExternal = () => {
    if (currentUrl && currentUrl !== 'about:blank') {
      void window.hermesDesktop?.openExternal(currentUrl)
    }
  }

  return (
    <aside className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-(--ui-editor-surface-background) text-(--ui-text-tertiary)">
      <div className="flex h-(--titlebar-height) shrink-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) px-2 [-webkit-app-region:no-drag]">
        <Tip label="Back">
          <button
            aria-label="Back"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!canGoBack}
            onClick={() => webviewRef.current?.goBack?.()}
            type="button"
          >
            <Codicon name="arrow-left" size="1rem" />
          </button>
        </Tip>
        <Tip label="Forward">
          <button
            aria-label="Forward"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!canGoForward}
            onClick={() => webviewRef.current?.goForward?.()}
            type="button"
          >
            <Codicon name="arrow-right" size="1rem" />
          </button>
        </Tip>
        <Tip label={loading ? 'Stop loading' : 'Reload'}>
          <button
            aria-label={loading ? 'Stop loading' : 'Reload'}
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={() => (loading ? webviewRef.current?.stop?.() : reload())}
            type="button"
          >
            <Codicon name={loading ? 'debug-stop' : 'refresh'} size="1rem" />
          </button>
        </Tip>
        <form className="min-w-0 flex-1" onSubmit={submitAddress}>
          <input
            aria-label="Browser address"
            className={cn(
              'h-8 w-full rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) px-2.5 font-mono text-xs text-foreground outline-none transition-colors',
              'focus:border-(--ui-stroke-primary) focus:ring-2 focus:ring-sidebar-ring/35'
            )}
            onChange={event => setAddressValue(event.target.value)}
            spellCheck={false}
            value={addressValue}
          />
        </form>
        {agentActionLabel && (
          <div
            aria-label="Browser agent status"
            className="hidden shrink-0 items-center rounded-full border border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) px-2 py-0.5 text-[0.625rem] font-medium text-(--ui-text-secondary) sm:flex"
          >
            Agent: {agentActionLabel}
          </div>
        )}
        <Tip label="Open in system browser">
          <button
            aria-label="Open in system browser"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!currentUrl || currentUrl === 'about:blank'}
            onClick={openExternal}
            type="button"
          >
            <Codicon name="link-external" size="1rem" />
          </button>
        </Tip>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden bg-white dark:bg-black">
        <div className={cn('absolute inset-0 flex', loadError && 'pointer-events-none opacity-0')} ref={hostRef} />
        {loadError && <BrowserLoadError error={loadError} onRetry={reload} />}
      </div>
    </aside>
  )
}
