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
import { previewTabIdsVisibleTo } from '@/store/preview'
import type { PreviewOwner } from '@/store/preview-ownership'
import { isBrowserWindow, windowBrowserTabId } from '@/store/windows'

import { actOnActivePreview } from './preview-act'
import { activePreviewNav } from './preview-nav'
import { type PreviewReadOptions, type PreviewReadResult, readActivePreview } from './preview-reader'
import { activePreviewScriptRunner } from './preview-script-runner'
import {
  closePreviewVaultBinding,
  evaluatePreviewVaultBinding,
  openPreviewVaultBinding,
  type PreviewVaultScope
} from './preview-vault'

const CHANNEL = 'hermes:preview-popout'

const ACT_TIMEOUT_MS = 20_000
const READ_TIMEOUT_MS = 8_000
const VAULT_TIMEOUT_MS = 20_000

type ActPayload = Omit<PreviewActAction, 'kind'> & { kind: string }

/** A vault binding op forwarded to the pop-out: the chat window keeps the
 *  session gate (it decides whether to forward), the pop-out owns the live
 *  document the binding pins. `scope` carries the request identity so the
 *  pop-out's own binding map answers only the matching request. */
type VaultPayload = {
  expression: string
  operation: string
  owner: PreviewOwner | undefined
  scope: PreviewVaultScope
  target: string
}

type VaultBridgeResult =
  | { decline: true }
  | { error: string; success: false }
  | { result: unknown; success: true }
  | { success: true; target: string }
  | { success: true }

/** `tabIds`: the tabs the requesting session may see, resolved in the chat
 *  window (it alone knows compression rotations and session tiles). A pop-out
 *  answers only when the tab it shows is among them; absent = unscoped. */
type BridgeRequest = { id: string; tabIds?: string[] } & (
  | { kind: 'act'; payload: ActPayload }
  | { kind: 'read'; payload: PreviewReadOptions }
  | { kind: 'vault'; payload: VaultPayload }
)

type BridgeResponse =
  | { id: string; kind: 'act'; result: PreviewActResult }
  | { id: string; kind: 'read'; result: PreviewReadResult | null }
  | { id: string; kind: 'vault'; result: VaultBridgeResult }
  | { id: string; kind: 'error'; error: string }

/** Cancel a forwarded request mid-flight (F5): the chat window owns the Stop/timeout
 *  gate, so when its signal aborts it tells the pop-out to abort the in-flight op
 *  before queued work (e.g. a password fill) reaches the page. */
type BridgeCancel = { cancel: true; id: string }

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

/** Scope a request to `owner`'s tabs (undefined = an unscoped request). */
const scopeFor = (owner: PreviewOwner | undefined): { tabIds?: string[] } =>
  owner === undefined ? {} : { tabIds: previewTabIdsVisibleTo(owner) }

function nextId(prefix: string): string {
  seq += 1

  return `${prefix}-${Date.now()}-${seq}`
}

function askPopout<T>(
  request: BridgeRequest,
  timeoutMs: number,
  pick: (response: BridgeResponse) => T | undefined,
  signal?: AbortSignal
): Promise<T | null> {
  const bus = getBus()

  if (!bus) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    let stop: (() => void) | undefined
    let onAbort: (() => void) | undefined

    const finish = (value: T | null) => {
      if (onAbort) {
        signal?.removeEventListener('abort', onAbort)
      }

      resolve(value)
    }

    const timer = window.setTimeout(() => {
      stop?.()
      finish(null)
    }, timeoutMs)

    // The chat window's Stop/timeout aborted: tell the pop-out to abort its own
    // in-flight controller so a queued fill never reaches the page (F5), then
    // stop waiting for a reply.
    if (signal) {
      onAbort = () => {
        window.clearTimeout(timer)
        bus.post({ cancel: true, id: request.id } satisfies BridgeCancel)
        stop?.()
        finish(null)
      }

      if (signal.aborted) {
        onAbort()

        return
      }

      signal.addEventListener('abort', onAbort, { once: true })
    }

    stop = bus.subscribe(data => {
      if (!data || typeof data !== 'object') {
        return
      }

      const response = data as Partial<BridgeResponse> & { id?: unknown }

      // Ignore our own request echo (same-window test buses / unusual hosts)
      // and unrelated traffic. Only a response carries `result` or `error`.
      if (response.id !== request.id) {
        return
      }

      if (!('result' in response) && response.kind !== 'error') {
        return
      }

      window.clearTimeout(timer)
      stop?.()

      if (response.kind === 'error') {
        finish(null)

        return
      }

      finish(pick(response as BridgeResponse) ?? null)
    })

    bus.post(request)
  })
}

/** Ask the browser pop-out to run drive_preview for `owner` (the requesting
 *  session's stored id). Null when no pop-out showing one of its tabs answers. */
export function requestPopoutPreviewAct(payload: ActPayload, owner?: PreviewOwner): Promise<PreviewActResult | null> {
  return askPopout({ id: nextId('act'), kind: 'act', payload, ...scopeFor(owner) }, ACT_TIMEOUT_MS, response =>
    response.kind === 'act' ? response.result : undefined
  )
}

/** Ask the browser pop-out to run read_preview for `owner`. Null when no
 *  pop-out showing one of its tabs answers. */
export function requestPopoutPreviewRead(
  payload: PreviewReadOptions = {},
  owner?: PreviewOwner
): Promise<PreviewReadResult | null> {
  return askPopout({ id: nextId('read'), kind: 'read', payload, ...scopeFor(owner) }, READ_TIMEOUT_MS, response =>
    response.kind === 'read' ? response.result : undefined
  )
}

/** Ask the browser pop-out to run a vault binding op (open/evaluate/close)
 *  against the live document it hosts. Null when no pop-out showing one of
 *  `owner`'s tabs answers — the caller then runs the op locally (#150283
 *  server-requests pop-out forwarding). The chat window keeps the session gate;
 *  this only reaches the document after that gate passed. */
export function requestPopoutPreviewVault(
  payload: VaultPayload,
  owner?: PreviewOwner,
  signal?: AbortSignal
): Promise<VaultBridgeResult | null> {
  return askPopout(
    { id: nextId('vault'), kind: 'vault', payload, ...scopeFor(owner) },
    VAULT_TIMEOUT_MS,
    response => (response.kind === 'vault' ? response.result : undefined),
    signal
  )
}

/**
 * Run a vault binding op against THIS renderer's live document — the pop-out's
 * local binding map. `open` mints a session-less user binding for the active
 * tab (the chat window already passed the session gate before forwarding);
 * `evaluate` runs the backend script while the pinned document is still live;
 * `close` releases it. The agent's session-active re-check lives in the chat
 * window, so here the binding's own document-still-pinned guard is authority.
 */
async function runPopoutVaultOperation(payload: VaultPayload, signal?: AbortSignal): Promise<VaultBridgeResult> {
  const { expression, operation, owner, scope, target } = payload

  if (operation === 'close') {
    return closePreviewVaultBinding({ ...scope, target })
  }

  if (operation === 'open') {
    // Pin the pop-out's active document for the agent request (same binding the
    // chat window would have minted locally, now minted where the page lives).
    return openPreviewVaultBinding(scope, owner ?? undefined)
  }

  if (operation === 'evaluate') {
    return evaluatePreviewVaultBinding({
      ...scope,
      expression,
      // The chat window gated the session before forwarding; the binding's own
      // document-pinned guard decides whether the page is still the right one.
      isSessionActive: () => true,
      // Carry the chat window's cancellation (F5): a Stop while this pop-out
      // waits to acquire guest activity must abort before the fill reaches the
      // page, not merely bump a revision after the await.
      signal,
      target
    })
  }

  return { error: 'The preview vault operation is invalid.', success: false }
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

  // In-flight forwarded ops the chat window can still cancel (F5). Keyed by
  // request id; a `cancel` message aborts the matching op before it fills.
  const inFlight = new Map<string, AbortController>()

  const stop = bus.subscribe(data => {
    if (!data || typeof data !== 'object') {
      return
    }

    const cancel = data as Partial<BridgeCancel>

    if (cancel.cancel === true && typeof cancel.id === 'string') {
      inFlight.get(cancel.id)?.abort()

      return
    }

    const request = data as Partial<BridgeRequest> & { id?: unknown; kind?: unknown }

    if (
      typeof request.id !== 'string' ||
      (request.kind !== 'act' && request.kind !== 'read' && request.kind !== 'vault') ||
      !('payload' in request)
    ) {
      return
    }

    // Another session's request: stay silent (an answer — even an error —
    // would win the race against a pop-out that does show that session's tab).
    if (Array.isArray(request.tabIds) && !request.tabIds.includes(windowBrowserTabId() ?? '')) {
      return
    }

    const id = request.id

    void (async () => {
      try {
        if (request.kind === 'act') {
          const result = await actOnActivePreview(request.payload as ActPayload)
          bus.post({ id, kind: 'act', result } satisfies BridgeResponse)

          return
        }

        if (request.kind === 'vault') {
          const controller = new AbortController()
          inFlight.set(id, controller)

          try {
            const result = await runPopoutVaultOperation(request.payload as VaultPayload, controller.signal)
            bus.post({ id, kind: 'vault', result } satisfies BridgeResponse)
          } finally {
            inFlight.delete(id)
          }

          return
        }

        const result = await readActivePreview(request.payload as PreviewReadOptions)
        bus.post({ id, kind: 'read', result } satisfies BridgeResponse)
      } catch (error) {
        inFlight.delete(id)
        bus.post({
          id,
          kind: 'error',
          error: error instanceof Error ? error.message : String(error)
        } satisfies BridgeResponse)
      }
    })()
  })

  return stop
}
