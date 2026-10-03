/**
 * Cross-window bridge for the popped-out Browser.
 *
 * Each Electron window is its own renderer, so the webview registries that
 * drive_preview / read_preview use live only in the window that mounts the
 * pane. After pop-out that is `?win=browser`, while the chat window still owns
 * the active-session gate for agent tools (`server-requests.ts`
 * WINDOW_OWNED_REQUESTS). This channel lets the gated chat window ask the
 * pop-out to run the live act/read and return the result.
 *
 * Transport: the same main-process IPC relay the Comment Mode handoff uses
 * (`window.hermesDesktop.windowRelay`). Packaged builds load renderers over
 * `file://`, where BroadcastChannel origin semantics must not be trusted;
 * BroadcastChannel stays only as a dev/browser fallback (dev serves the
 * renderer over http, so same-origin channels work there).
 */

import type { PreviewActAction, PreviewActResult } from '@/lib/preview-act/act-in-page'
import { selectedPopoutTarget, validPopoutTarget } from '@/store/browser-workspaces'
import { previewTabIdsVisibleTo, previewTabsFor } from '@/store/preview'
import type { PreviewOwner } from '@/store/preview-ownership'
import { isBrowserWindow } from '@/store/windows'

import { BROWSER_REQUEST_HISTORY_LIMIT, BROWSER_REQUEST_TIMEOUT_MS, BrowserRequestHistory } from '../../../../electron/browser-request-history'
import type { BrowserConversation, BrowserRequestTarget } from '../../../../electron/browser-workspace-types'

import { actOnActivePreview } from './preview-act'
import { activePreviewNav } from './preview-nav'
import { type PreviewReadOptions, type PreviewReadResult, readActivePreview } from './preview-reader'
import { activePreviewScriptRunner } from './preview-script-runner'

const CHANNEL = 'hermes:preview-popout'

const ACT_TIMEOUT_MS = BROWSER_REQUEST_TIMEOUT_MS.act
const READ_TIMEOUT_MS = BROWSER_REQUEST_TIMEOUT_MS.read

type ActPayload = Omit<PreviewActAction, 'kind'> & { kind: string }

type BridgeRequest = { target: BrowserRequestTarget; requester: BrowserConversation; owner?: PreviewOwner; tabIds: string[]; deadline: number } & (
  { id: string; kind: 'act'; payload: ActPayload } | { id: string; kind: 'read'; payload: PreviewReadOptions }
)

type BridgeResponse = { target: BrowserRequestTarget } & (
  | { id: string; kind: 'act'; result: PreviewActResult }
  | { id: string; kind: 'read'; result: PreviewReadResult | null }
  | { id: string; kind: 'error'; error: string }
)

interface BridgeCancel {
  id: string
  kind: 'cancel'
  target: BrowserRequestTarget
}

let seq = 0

type RelayBus = {
  post: (message: unknown) => void
  subscribe: (listener: (data: unknown) => void) => () => void
}

let cachedBus: RelayBus | null = null

function getBus(): RelayBus | null {
  if (cachedBus) {
    return cachedBus
  }

  const desktopRelay = typeof window !== 'undefined' ? window.hermesDesktop?.windowRelay : undefined

  if (desktopRelay?.onMessage && desktopRelay.send) {
    cachedBus = {
      post: message => desktopRelay.send(message),
      subscribe: listener => desktopRelay.onMessage(payload => listener(payload))
    }

    return cachedBus
  }

  if (typeof BroadcastChannel === 'undefined') {
    return null
  }

  const channel = new BroadcastChannel(CHANNEL)

  cachedBus = {
    post: message => channel.postMessage(message),
    subscribe: listener => {
      const onMessage = (event: MessageEvent<unknown>) => listener(event.data)
      channel.addEventListener('message', onMessage)

      return () => channel.removeEventListener('message', onMessage)
    }
  }

  return cachedBus
}

/** True when this renderer has a live webview (or nav handle) for the active
 *  tab among those `owner` (omitted = the focused session) may see. */
export function hasLivePreviewSurface(owner?: PreviewOwner): boolean {
  return Boolean(activePreviewScriptRunner(owner) || activePreviewNav(owner))
}


function requestPreviewOwner(target: BrowserRequestTarget, requester?: BrowserConversation | null): PreviewOwner {
  const conversation = requester ?? target.owner.conversation!

  return {
    profile: target.owner.scope,
    runtimeId: conversation.kind === 'session' ? conversation.id : null,
    sessionId: conversation.kind === 'session' ? conversation.id : null
  }
}

function nextId(prefix: string, deadline: number): string {
  seq += 1

  return `${prefix}-${globalThis.crypto.randomUUID()}-${seq}-${deadline}`
}

function askPopout<T>(
  request: BridgeRequest,
  timeoutMs: number,
  pick: (response: BridgeResponse) => T | undefined,
  signal?: AbortSignal
): Promise<T | null> {
  const bus = getBus()

  if (!bus || signal?.aborted) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    let stop: (() => void) | undefined
    let settled = false

    const finish = (value: T | null) => {
      if (settled) {return}
      settled = true
      window.clearTimeout(timer)
      signal?.removeEventListener('abort', cancel)
      stop?.()
      resolve(value)
    }

    const cancel = () => {
      if (settled) {return}
      // Cancel the captured request, never re-resolve the currently selected tab.
      bus.post({ id: request.id, kind: 'cancel', target: request.target } satisfies BridgeCancel)
      finish(null)
    }

    const timer = window.setTimeout(cancel, timeoutMs)

    stop = bus.subscribe(data => {
      if (!data || typeof data !== 'object') {
        return
      }

      const response = data as Partial<BridgeResponse> & { id?: unknown }

      // Ignore our own request echo (same-window test buses / unusual hosts)
      // and unrelated traffic. Only a response carries `result` or `error`.
      if (response.id !== request.id || JSON.stringify(response.target) !== JSON.stringify(request.target)) {
        return
      }

      if (!('result' in response) && response.kind !== 'error') {
        return
      }

      if (response.kind === 'error') {
        finish(null)

        return
      }

      finish(pick(response as BridgeResponse) ?? null)
    })

    signal?.addEventListener('abort', cancel, { once: true })
    bus.post(request)
  })
}

/** Ask the browser pop-out to run drive_preview. Null when no pop-out answers. */
export async function requestPopoutPreviewAct(payload: ActPayload, requester?: BrowserConversation | null, signal?: AbortSignal, owner?: PreviewOwner): Promise<PreviewActResult | null> {
  const tabIds = owner === undefined ? previewTabsFor().map(tab => tab.id) : previewTabIdsVisibleTo(owner)
  const target = selectedPopoutTarget(requester, tabIds)

  if (!target) {
    return null
  }

  const deadline = Date.now() + ACT_TIMEOUT_MS

  return (
    (await askPopout({ id: nextId('act', deadline), kind: 'act', payload, target, owner: owner ?? requestPreviewOwner(target, requester), tabIds, requester: requester ?? target.owner.conversation!, deadline }, ACT_TIMEOUT_MS, response =>
      response.kind === 'act' ? response.result : undefined, signal
    )) ?? {
      success: false,
      error: 'The selected detached browser did not answer. Retry against that tab; the action was not redirected.'
    }
  )
}

/** Ask the browser pop-out to run read_preview. Null when no pop-out answers. */
export function requestPopoutPreviewRead(payload: PreviewReadOptions = {}, requester?: BrowserConversation | null, owner?: PreviewOwner): Promise<PreviewReadResult | null> {
  const tabIds = owner === undefined ? previewTabsFor().map(tab => tab.id) : previewTabIdsVisibleTo(owner)
  const target = selectedPopoutTarget(requester, tabIds)

  if (!target) {
    return Promise.resolve(null)
  }

  const deadline = Date.now() + READ_TIMEOUT_MS

  return askPopout({ id: nextId('read', deadline), kind: 'read', payload, target, owner: owner ?? requestPreviewOwner(target, requester), tabIds, requester: requester ?? target.owner.conversation!, deadline }, READ_TIMEOUT_MS, response =>
    response.kind === 'read' ? response.result : undefined
  )
}

let responderStop: (() => void) | null = null
let responderUsers = 0
// Survives responder remounts; unexpired IDs cannot regain execution authority.
const receivedRequests = new BrowserRequestHistory()

function releaseResponder(): () => void {
  let released = false

  return () => {
    if (released) {
      return
    }

    released = true

    if (--responderUsers === 0) {
      responderStop?.()
      responderStop = null
    }
  }
}

/**
 * Browser pop-out only: answer act/read requests from the chat window.
 * Safe to call more than once; installs a single listener.
 */
export function installPopoutPreviewResponder(): () => void {
  if (!isBrowserWindow()) {
    return () => {}
  }

  const bus = getBus()

  if (!bus) {
    return () => {}
  }

  if (responderStop) {
    responderUsers++

    return releaseResponder()
  }

  const running = new Map<string, { target: BrowserRequestTarget; controller: AbortController; timer: number }>()

  const stop = bus.subscribe(data => {
    if (!data || typeof data !== 'object') {
      return
    }

    const request = data as Partial<BridgeRequest | BridgeCancel>

    if (request.kind === 'cancel') {
      const pending = typeof request.id === 'string' ? running.get(request.id) : undefined

      // Main authenticates the opener against its pending record. The original
      // target may already be invalid; that must never prevent cancellation.
      if (pending && JSON.stringify(pending.target) === JSON.stringify(request.target)) {
        pending.controller.abort('interrupted')
      }

      return
    }

    if (
      typeof request.id !== 'string' ||
      (request.kind !== 'act' && request.kind !== 'read') ||
      !('payload' in request) ||
      !request.payload ||
      typeof request.payload !== 'object' ||
      typeof request.deadline !== 'number' || !Number.isFinite(request.deadline) ||
      !request.target ||
      !validPopoutTarget(request.target)
    ) {
      return
    }

    // Another session's request: stay silent (an answer — even an error —
    // would win the race against a pop-out that does show that session's tab).
    if (!Array.isArray(request.tabIds) || !request.tabIds.includes(request.target.tabId)) {
      return
    }

    const id = request.id
    const target = request.target

    if (running.has(id) || running.size >= BROWSER_REQUEST_HISTORY_LIMIT ||
      !receivedRequests.admit(id, request.kind, request.deadline)) {
      return
    }

    const remaining = request.deadline - Date.now()

    // Admission and execution can straddle the deadline; do not invoke a reader
    // (which has no AbortSignal) or an action after its budget expires.
    if (remaining <= 0) {return}
    const controller = new AbortController()
    const timer = window.setTimeout(() => controller.abort('timeout'), remaining)
    running.set(id, { target, controller, timer })

    void (async () => {
      try {
        if (request.kind === 'act') {
          const result = await actOnActivePreview(request.payload as ActPayload, controller.signal, request.owner, {
            tabId: target.tabId,
            valid: () => !controller.signal.aborted && Date.now() < request.deadline! && validPopoutTarget(target)
          })

          bus.post({ id, target, kind: 'act', result: controller.signal.aborted
            ? { success: false, error: 'The detached browser action was cancelled.' }
            : result } satisfies BridgeResponse)

          return
        }

        const result = await readActivePreview(request.payload as PreviewReadOptions, request.owner, target.tabId, request.tabIds)
        bus.post({
          id,
          target,
          kind: 'read',
          result: !controller.signal.aborted && validPopoutTarget(target) ? result : null
        } satisfies BridgeResponse)
      } catch (error) {
        bus.post({
          id,
          target,
          kind: 'error',
          error: error instanceof Error ? error.message : String(error)
        } satisfies BridgeResponse)
      } finally {
        window.clearTimeout(timer)
        running.delete(id)
      }
    })()
  })

  responderStop = () => {
    stop()

    for (const pending of running.values()) {
      pending.controller.abort('interrupted')
      window.clearTimeout(pending.timer)
    }

    running.clear()
  }

  responderUsers++

  return releaseResponder()
}
