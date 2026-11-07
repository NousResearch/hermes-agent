/**
 * The Webapp's half of `hermesDesktop.notify`: what Electron's notification IPC
 * does in main, done with the browser's Notification API.
 *
 * Notifications show through a service worker registration
 * (`public/notification-worker.js`, at the Webapp's base path): mobile browsers
 * throw from `new Notification()` and show only what a registration shows. The
 * worker routes a click back to a Webapp tab, which opens the session here. The
 * constructor stays for a browser that runs no worker (a desktop Firefox
 * private window), where it still works; where it throws too, notify answers
 * false rather than rejecting.
 *
 * Permission is asked only from a click in Settings (browsers ignore or
 * penalise prompts that no gesture started), so an event arriving before the
 * user opted in shows nothing here and the tab-title flash carries it alone.
 * The API needs a secure context: Tailscale serve (https) and localhost have
 * it, a plain-http LAN address does not.
 */

import { flagTabAttention } from '@/store/browser-tab-attention'

export type BrowserNotificationPermission = 'default' | 'denied' | 'granted' | 'insecure' | 'unsupported'

/** At the root of the Webapp's dist, so the worker's scope is the Webapp's base path. */
const WORKER_SCRIPT = 'notification-worker.js'
/** What the worker posts to the tab a click brings forward (public/notification-worker.js). */
const FOCUS_SESSION_MESSAGE = 'hermes:focus-session'

let basePath = ''
let workerRegistration: null | Promise<null | ServiceWorkerRegistration> = null

export function browserNotificationPermission(): BrowserNotificationPermission {
  if (typeof window === 'undefined') {
    return 'unsupported'
  }

  if (!window.isSecureContext) {
    return 'insecure'
  }

  return 'Notification' in window ? Notification.permission : 'unsupported'
}

/** Call from a click handler only. */
export async function requestBrowserNotificationPermission(): Promise<BrowserNotificationPermission> {
  if (browserNotificationPermission() === 'default') {
    await Notification.requestPermission()
  }

  const permission = browserNotificationPermission()

  // Register now, so the first event finds the worker active.
  if (permission === 'granted') {
    void notificationWorker()
  }

  return permission
}

type FocusSessionListener = (sessionId: string) => void

const focusSessionListeners = new Set<FocusSessionListener>()

/** Same contract as Electron's `onFocusSession`: a notification click names the session to open. */
export function onBrowserFocusSession(callback: FocusSessionListener): () => void {
  focusSessionListeners.add(callback)

  return () => focusSessionListeners.delete(callback)
}

function focusSession(sessionId: string): void {
  focusSessionListeners.forEach(listener => listener(sessionId))
}

/** The session a worker message asks this tab to open, or null for any other message. */
function focusSessionRequest(data: unknown): null | string {
  const message = (data ?? {}) as { sessionId?: unknown; type?: unknown }

  return message.type === FOCUS_SESSION_MESSAGE && typeof message.sessionId === 'string' && message.sessionId
    ? message.sessionId
    : null
}

/** *registration* once a worker of it is active: `showNotification` refuses a registration without one. */
async function whenActive(registration: ServiceWorkerRegistration): Promise<ServiceWorkerRegistration> {
  const pending = registration.installing ?? registration.waiting

  if (!registration.active && pending) {
    await new Promise<void>((resolve, reject) => {
      pending.addEventListener('statechange', () => {
        if (pending.state === 'activated') {
          resolve()
        } else if (pending.state === 'redundant') {
          reject(new Error('The notification worker did not install'))
        }
      })
    })
  }

  return registration
}

/** The notification worker's registration once it is active, or null where this browser runs no
 *  worker. A failed registration (a dropped connection) is tried again on the next event. */
function notificationWorker(): Promise<null | ServiceWorkerRegistration> {
  if (!('serviceWorker' in navigator)) {
    return Promise.resolve(null)
  }

  workerRegistration ??= navigator.serviceWorker
    .register(`${basePath}/${WORKER_SCRIPT}`, { scope: `${basePath}/` })
    .then(whenActive)
    .catch(() => {
      workerRegistration = null

      return null
    })

  return workerRegistration
}

/** Wire this tab to the notification worker at *base* (the Webapp's base path): a click it routes here
 *  opens its session, and a tab already allowed to notify registers it now, before any event. */
export function installBrowserNotifications(base: string): void {
  basePath = base

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', event => {
      const sessionId = focusSessionRequest(event.data)

      if (sessionId) {
        focusSession(sessionId)
      }
    })
  }

  if (browserNotificationPermission() === 'granted') {
    void notificationWorker()
  }
}

export interface BrowserNotifyPayload {
  body?: string
  focusSessionId?: string
  kind?: string
  sessionId?: string
  tag?: string
  title?: string
}

/** For a browser that runs no worker: its own notification, whose click this tab handles. */
function showWithConstructor(title: string, options: NotificationOptions, focusSessionId?: string): void {
  const notification = new Notification(title, options)

  notification.onclick = () => {
    // A click on our own notification is the user's request to come back.
    window.focus()
    notification.close()

    if (focusSessionId) {
      focusSession(focusSessionId)
    }
  }
}

/** Flash the tab, then show a browser notification when the user allowed it. True when the browser
 *  took the notification; never rejects. */
export async function showBrowserNotification(payload: BrowserNotifyPayload): Promise<boolean> {
  const title = payload.title || 'Hermes'

  flagTabAttention(title)

  if (browserNotificationPermission() !== 'granted') {
    return false
  }

  const focusSessionId = payload.focusSessionId || payload.sessionId

  const options: NotificationOptions = {
    body: payload.body,
    data: { focusSessionId },
    // One OS entry per session+kind: a replayed event replaces, never stacks.
    tag: `hermes:${payload.kind ?? ''}:${focusSessionId ?? payload.tag ?? ''}`
  }

  const registration = await notificationWorker()

  try {
    if (registration) {
      await registration.showNotification(title, options)
    } else {
      showWithConstructor(title, options, focusSessionId)
    }

    return true
  } catch {
    // A mobile browser that runs no worker throws from the constructor; a registration refuses
    // once the user revoked the permission.
    return false
  }
}
