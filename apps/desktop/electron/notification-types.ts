export interface HermesNotification {
  title?: string
  body?: string
  silent?: boolean
  kind?: string
  sessionId?: string
  /** Durable click target captured before runtime bindings can be recycled. */
  focusSessionId?: string
  /** Dedupe discriminator for session-less notifications (e.g. plugin id). */
  tag?: string
  /** Absolute icon path for Electron `Notification`. */
  icon?: string
  /** Resolved hash-router path opened on body click (plugin / deeplink-compatible). */
  activate?: string
  /** Renderer handle for onActivate / onAction callbacks. */
  notifyId?: string
  actions?: { id: string; text: string; activate?: string }[]
}
