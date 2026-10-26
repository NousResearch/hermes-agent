import { randomUuid } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'
import { dismissNotification } from '@/store/notifications'

interface BrowserWindowOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  basePath: string
  privateSession: boolean
}

const POPUP_BLOCKED = 'Webapp window did not open. Allow popups for this site and try again.'

/** The blocked-window rejection. Its toast goes up under `notificationId` (runWindowOpen), so a
 *  window that says `ready` after the deadline can take it back down. */
class WindowBlockedError extends Error {
  readonly notificationId = `webapp-window-blocked:${randomUuid()}`

  constructor() {
    super(POPUP_BLOCKED)
  }
}

// The new window gets its own browsing context group: it does not share this tab's event loop,
// starts with empty sessionStorage and gets no Referer. The cost: window.open returns null whether
// or not the window opened (HTML "window open steps"), so each window says `ready` on a
// BroadcastChannel instead.
const WINDOW_FEATURES = 'noopener,noreferrer'

// A window that has not said `ready` by then is reported as blocked. In Chrome, Edge and WebKit a
// window that opened said it within 0.4 s (the launch-link handoff page) and about 1.3 s (the
// whole app on a gated host), even with the browser held to one CPU. A slower window still opens
// and takes the toast back down, and on a launch-link tab it still signs in.
const READY_DEADLINE_MS = 3_000

// How long a pop-out keeps listening past the deadline: for a window that starts late, or one the
// user opens from the browser's blocked-popup prompt.
const LATE_WINDOW_MS = 30_000

// A gated host's window is the app itself, which says `ready` at boot on the channel this
// parameter names (`announceWebappWindow`).
const HELLO_PARAM = 'hello'

const helloChannel = (id: string) => `hermes.webapp.hello:${id}`

// Not crypto.randomUUID: it exists only in secure contexts, and a gated host may be served over
// plain HTTP from another machine.
const helloId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('')

const asError = (error: unknown) => (error instanceof Error ? error : new Error(String(error)))

/** Open synchronously in the user gesture; never rely on an opener/storage copy. */
export function createBrowserWindowOpener({ api, basePath, privateSession }: BrowserWindowOptions) {
  return (target: URL): Promise<{ ok: true }> =>
    privateSession ? openLaunchLinkWindow(target, api, basePath) : openGatedWindow(target)
}

function openGatedWindow(target: URL): Promise<{ ok: true }> {
  const id = helloId()
  const channel = new BroadcastChannel(helloChannel(id))
  const href = new URL(target)
  href.searchParams.set(HELLO_PARAM, id)

  return new Promise((resolve, reject) => {
    const blocked = new WindowBlockedError()

    const close = () => {
      window.clearTimeout(deadline)
      window.clearTimeout(expiry)
      channel.close()
    }

    const deadline = window.setTimeout(() => reject(blocked), READY_DEADLINE_MS)
    const expiry = window.setTimeout(close, LATE_WINDOW_MS)

    // Before the deadline this resolves; past it the promise has rejected, and the window that
    // opened after all takes its toast down.
    channel.onmessage = event => {
      if (event.data?.type === 'ready') {
        close()
        resolve({ ok: true })
        dismissNotification(blocked.notificationId)
      }
    }

    try {
      window.open(href.href, '_blank', WINDOW_FEATURES)
    } catch (error) {
      close()
      reject(asError(error))
    }
  })
}

/**
 * The window opens on the handoff page, which says `ready` and signs in with a one-use ticket this
 * tab mints for it. Past the deadline the promise rejects as blocked, but the channel stays open
 * until the handoff expires: a window that starts late, or one the user opens from the browser's
 * blocked-popup prompt, still signs in.
 */
function openLaunchLinkWindow(
  target: URL,
  api: BrowserWindowOptions['api'],
  basePath: string
): Promise<{ ok: true }> {
  const id = randomUuid()
  const channel = new BroadcastChannel(`hermes.webapp.window:${id}`)
  const launch = new URL(`${basePath}/webapp/window`, window.location.origin)
  launch.hash = new URLSearchParams({ channel: id, query: target.search, route: target.hash.slice(1) }).toString()

  return new Promise((resolve, reject) => {
    const blocked = new WindowBlockedError()
    let minting = false
    let closed = false

    // Once the deadline has rejected the promise, a later done or error only closes the channel.
    const settle = (error?: Error) => {
      window.clearTimeout(deadline)

      if (error) {reject(error)} else {resolve({ ok: true })}
    }

    const close = (error?: Error) => {
      closed = true
      window.clearTimeout(expiry)
      channel.close()
      settle(error)
    }

    const deadline = window.setTimeout(() => settle(blocked), READY_DEADLINE_MS)
    const expiry = window.setTimeout(() => close(blocked), LATE_WINDOW_MS)

    channel.onmessage = event => {
      if (closed) {return}

      if (event.data?.type === 'ready' && !minting) {
        minting = true
        window.clearTimeout(deadline)
        dismissNotification(blocked.notificationId)
        void api<{ ticket: string }>({ method: 'POST', path: '/api/webapp/window-ticket' }).then(({ ticket }) => {
          if (closed) {return}
          channel.postMessage({ ticket })
        }).catch(error => {
          if (closed) {return}
          channel.postMessage({ error: 'Could not authorize this window. Try opening it again from the original tab.' })
          close(asError(error))
        })
      } else if (event.data?.type === 'done') {
        close()
      } else if (event.data?.type === 'error') {
        close(new Error('Could not authorize the Webapp window. Try opening it again.'))
      }
    }

    try {
      window.open(launch.href, '_blank', WINDOW_FEATURES)
    } catch (error) {
      close(asError(error))
    }
  })
}

/**
 * In a window a gated host opened: say `ready` on the channel its address names, then drop the
 * name from the address. The message carries nothing, so a same-origin page that guessed the
 * name could at most spare the opener its blocked-popup toast.
 */
export function announceWebappWindow(): void {
  const url = new URL(window.location.href)
  const id = url.searchParams.get(HELLO_PARAM)

  if (id === null) {return}
  url.searchParams.delete(HELLO_PARAM)
  window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`)

  if (!/^[a-f0-9]{32}$/.test(id)) {return}
  const channel = new BroadcastChannel(helloChannel(id))
  channel.postMessage({ type: 'ready' })
  channel.close()
}

/** The secondary-window URL showing one session, or null for an empty session id. */
export function sessionWindowTarget(
  currentHref: string,
  sessionId: string,
  opts?: { profile?: null | string; watch?: boolean }
): URL | null {
  const id = sessionId.trim()

  if (!id) {return null}
  const target = new URL(currentHref)
  target.searchParams.set('win', 'secondary')

  // Written out even for Default: without `?profile=` the window boots on
  // this origin's saved default, which may be another profile.
  const profile = opts?.profile?.trim()

  if (profile) {
    target.searchParams.set('profile', profile)
  }

  if (opts?.watch) {
    target.searchParams.set('watch', '1')
  } else {
    target.searchParams.delete('watch')
  }

  target.hash = `/${encodeURIComponent(id)}`

  return target
}
