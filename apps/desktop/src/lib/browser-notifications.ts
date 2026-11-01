/**
 * The Webapp's half of `hermesDesktop.notify`: what Electron's notification IPC
 * does in main, done with the browser's Notification API.
 *
 * Permission is asked only from a click in Settings (browsers ignore or
 * penalise prompts that no gesture started), so an event arriving before the
 * user opted in shows nothing here and the tab-title flash carries it alone.
 * The API needs a secure context: Tailscale serve (https) and localhost have
 * it, a plain-http LAN address does not.
 */

import { flagTabAttention } from '@/store/browser-tab-attention'

export type BrowserNotificationPermission = 'default' | 'denied' | 'granted' | 'insecure' | 'unsupported'

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

  return browserNotificationPermission()
}

type FocusSessionListener = (sessionId: string) => void

const focusSessionListeners = new Set<FocusSessionListener>()

/** Same contract as Electron's `onFocusSession`: a notification click names the session to open. */
export function onBrowserFocusSession(callback: FocusSessionListener): () => void {
  focusSessionListeners.add(callback)

  return () => focusSessionListeners.delete(callback)
}

export interface BrowserNotifyPayload {
  body?: string
  focusSessionId?: string
  kind?: string
  sessionId?: string
  tag?: string
  title?: string
}

/** Flash the tab, then show a browser notification when the user allowed it. */
export function showBrowserNotification(payload: BrowserNotifyPayload): boolean {
  const title = payload.title || 'Hermes'

  flagTabAttention(title)

  if (browserNotificationPermission() !== 'granted') {
    return false
  }

  const focusSessionId = payload.focusSessionId || payload.sessionId

  const notification = new Notification(title, {
    body: payload.body,
    // One OS entry per session+kind: a replayed event replaces, never stacks.
    tag: `hermes:${payload.kind ?? ''}:${focusSessionId ?? payload.tag ?? ''}`
  })

  notification.onclick = () => {
    // A click on our own notification is the user's request to come back.
    window.focus()
    notification.close()

    if (focusSessionId) {
      focusSessionListeners.forEach(listener => listener(focusSessionId))
    }
  }

  return true
}
