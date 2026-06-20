/**
 * native.ts — the handful of window.hermesDesktop methods that map cleanly onto
 * Capacitor plugins. Everything else is stubbed in stubs.ts.
 */

import { App } from '@capacitor/app'
import { Browser } from '@capacitor/browser'
import { Clipboard } from '@capacitor/clipboard'
import { Capacitor } from '@capacitor/core'
import { LocalNotifications } from '@capacitor/local-notifications'

import type { HermesNotification } from '@/global'

export async function writeClipboard(text: string): Promise<boolean> {
  try {
    await Clipboard.write({ string: text })
    return true
  } catch {
    return false
  }
}

export async function openExternal(url: string): Promise<void> {
  try {
    await Browser.open({ url })
  } catch {
    /* ignore — nothing actionable on the mobile side */
  }
}

let notifyId = 1
let notifPermissionAsked = false

export async function notify(payload: HermesNotification): Promise<boolean> {
  if (!Capacitor.isNativePlatform()) return false
  try {
    if (!notifPermissionAsked) {
      notifPermissionAsked = true
      const perm = await LocalNotifications.checkPermissions()
      if (perm.display !== 'granted') {
        await LocalNotifications.requestPermissions()
      }
    }
    await LocalNotifications.schedule({
      notifications: [
        {
          id: notifyId++,
          title: payload.title ?? 'Hermes',
          body: payload.body ?? '',
          silent: payload.silent,
          extra: { sessionId: payload.sessionId, kind: payload.kind },
        },
      ],
    })
    return true
  } catch {
    return false
  }
}

/** Wire Capacitor's app-resume into a callback so the vendored reconnect-on-wake
 *  logic fires unchanged. Returns an unsubscribe. */
export function onPowerResume(callback: () => void): () => void {
  const handle = App.addListener('resume', () => callback())
  return () => {
    void handle.then((h) => h.remove())
  }
}
