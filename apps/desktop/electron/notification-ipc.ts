import { BrowserWindow, ipcMain, Notification } from 'electron'

import type { HermesNotification } from '../src/global'

import { createEventDeduper } from './event-dedupe'
import { resolveNotificationAction } from './notification-actions'
import { createNotificationRegistry } from './notification-registry'

interface NotificationHost {
  getMainWindow: () => BrowserWindow | null
  focusWindow: (window: BrowserWindow) => void
}

export function registerNativeNotifications({ getMainWindow, focusWindow }: NotificationHost): void {
  const isDuplicateNotification = createEventDeduper()
  const notifications = createNotificationRegistry()

  ipcMain.handle('hermes:notify', (event, payload: HermesNotification) => {
    // The source renderer owns runtime bindings and plugin callbacks.
    const sourceWindow = BrowserWindow.fromWebContents(event.sender)
    const targetWindow = () => (sourceWindow && !sourceWindow.isDestroyed() ? sourceWindow : getMainWindow())

    if (!Notification.isSupported()) {
      return false
    }

    // Peer renderers share one OS notification for the same event.
    if (isDuplicateNotification(`${payload?.kind ?? ''}:${payload?.sessionId ?? payload?.tag ?? ''}`)) {
      return true
    }

    const actions = Array.isArray(payload?.actions) ? payload.actions : []
    const icon = typeof payload?.icon === 'string' && payload.icon.trim() ? payload.icon.trim() : undefined
    const notification = new Notification({
      title: payload?.title || 'Hermes',
      body: payload?.body || '',
      silent: Boolean(payload?.silent),
      ...(icon ? { icon } : {}),
      actions: actions.map(action => ({ type: 'button', text: String(action?.text || '') }))
    })

    notification.on('click', () => {
      const window = targetWindow()

      if (!window || window.isDestroyed()) {
        return
      }

      focusWindow(window)

      const focusSessionId = payload?.focusSessionId || payload?.sessionId

      if (focusSessionId) {
        window.webContents.send('hermes:focus-session', focusSessionId)
      }

      if (payload?.activate || payload?.notifyId) {
        window.webContents.send('hermes:notification-activate', {
          activate: payload?.activate,
          notifyId: window === sourceWindow ? payload?.notifyId : undefined,
          tag: payload?.tag
        })
      }
    })
    notification.on('action', (actionEvent, index) => {
      const window = targetWindow()

      if (!window || window.isDestroyed()) {
        return
      }

      const action = resolveNotificationAction(actions, actionEvent, index)

      if (!action?.id) {
        return
      }

      if (payload?.sessionId && !payload?.notifyId && !payload?.activate) {
        // Runtime actions belong to the source renderer, never the fallback.
        if (window !== sourceWindow) {
          return
        }

        window.webContents.send('hermes:notification-action', { sessionId: payload.sessionId, actionId: action.id })

        return
      }

      focusWindow(window)
      window.webContents.send('hermes:notification-activate', {
        actionId: action.id,
        activate: action.activate || payload?.activate,
        notifyId: window === sourceWindow ? payload?.notifyId : undefined,
        tag: payload?.tag
      })
    })
    notifications.retain(notification)
    notification.show()

    return true
  })
}
