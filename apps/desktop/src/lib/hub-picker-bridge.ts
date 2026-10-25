/**
 * Renderer half of the picker open-bridge (electron/hub-picker-bridge.ts).
 *
 * A link inside an embedded hub frame cannot reach the OS browser through
 * `window.open` (#135315): the main-process window-open handler has no way to
 * prove which frame asked, and widening it would re-open GHSA-9f4c-93c8-jc8g.
 * Instead main injects a shim into a frame it has identified by process/routing
 * id and verified as an exact hub origin; the shim posts the requested URL here.
 *
 * SECURITY: this parser is deliberately dumb — it only accepts the message
 * shape and the schemes the audited `hermes:openExternal` channel allows. The
 * trust decision belongs to the caller's `event.origin` check, which Chromium
 * sets and no frame can forge: only the hub origin is the hub origin. Call
 * this AFTER that check, never before it.
 */

/** The message type the injected shim posts (`HUB_PICKER_OPEN_MESSAGE`). */
export const HUB_PICKER_OPEN_MESSAGE = 'hermes-hub-open'

const OPENABLE_SCHEMES = new Set(['http:', 'https:', 'mailto:'])

/** The requested URL from a bridged picker message, or null when it is not one. */
export function bridgedHubOpenUrl(data: unknown): null | string {
  if (!data || typeof data !== 'object') {
    return null
  }

  const message = data as { type?: unknown; url?: unknown }

  if (message.type !== HUB_PICKER_OPEN_MESSAGE || typeof message.url !== 'string' || !message.url) {
    return null
  }

  try {
    return OPENABLE_SCHEMES.has(new URL(message.url).protocol) ? message.url : null
  } catch {
    return null
  }
}
