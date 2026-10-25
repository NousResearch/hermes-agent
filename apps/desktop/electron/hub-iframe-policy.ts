/**
 * Capability carve-outs for the embedded Skills Hub iframe ONLY.
 *
 * The Skills Hub picker (Bot Mode) embeds the public docs site
 * (`https://hermes-agent.nousresearch.com/docs/skills?embed=picker`, or the
 * GitHub Pages mirror when Vercel refuses the network). Three default denials
 * make that embed nearly unusable — window.open is killed by the CVE-2026-70608
 * handler, clipboard writes by the session permission handlers, and the frame
 * can drift off the picker URL into the un-embeddable docs site. Every carve-out
 * below is gated on the EXACT hub origin: a sandboxed artifact iframe (opaque
 * origin `null`) or any other guest frame never qualifies.
 */

/** The docs site served from Vercel. */
export const HERMES_HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
/** The equivalent GitHub Pages deployment of the same site. */
export const HERMES_HUB_FALLBACK_ORIGIN = 'https://nousresearch.github.io'

const HUB_ORIGINS = new Set([HERMES_HUB_ORIGIN, HERMES_HUB_FALLBACK_ORIGIN])

/**
 * Exact-origin membership — `origin` comes from the frame's own `origin`
 * property (URL parsing), never from a string we control. `new URL(...).origin`
 * is `null` for sandboxed frames and unique for `data:` URLs, so those can
 * never alias the hub.
 */
export function isHermesHubOrigin(origin: string | null | undefined): boolean {
  return typeof origin === 'string' && HUB_ORIGINS.has(origin)
}

/** The URL schemes a trusted hub frame may delegate to the OS browser. */
const HUB_EXTERNAL_SCHEMES = new Set(['http:', 'https:', 'mailto:'])

/**
 * The picker paths of the docs site. Both the embedded Skills Hub picker and
 * the plugin catalog live on one of these; every other path is the full,
 * un-embeddable docs site.
 */
const HUB_PICKER_PATHS = new Set([
  '/hermes-agent/docs/skills',
  '/docs/skills',
  '/hermes-agent/docs/plugins',
  '/docs/plugins'
])

/** Is this URL one of the hub's embeddable picker pages (exact origin AND path)? */
export function isHermesHubPickerUrl(url: string): boolean {
  try {
    const parsed = new URL(url)

    return HUB_ORIGINS.has(parsed.origin) && HUB_PICKER_PATHS.has(parsed.pathname)
  } catch {
    return false
  }
}

/**
 * May the desktop inject the picker open-bridge into a frame that just
 * committed `url` from `frameOrigin`?
 *
 * Both halves must hold. The origin comes off the LIVE frame object (URL
 * parsing of the real guest origin), never from a string we hold — so a
 * sandboxed artifact frame (opaque origin `null`) or any other guest can never
 * be bridged, and a frame that merely *claims* a picker URL in a message is
 * irrelevant here (this is a committed-navigation check).
 */
export function isBridgedPickerNavigation(url: string, frameOrigin: string | null | undefined): boolean {
  return isHermesHubOrigin(frameOrigin) && isHermesHubPickerUrl(url)
}

/**
 * May a trusted-origin window.open request be handed to the audited external
 * opener? http/https/mailto only — the same allowlist `openExternalUrl` in
 * external-open.ts enforces; anything else (file:, javascript:, custom
 * schemes) stays denied with no side effect.
 */
export function isHermesHubExternalUrl(url: string): boolean {
  try {
    return HUB_EXTERNAL_SCHEMES.has(new URL(url).protocol)
  } catch {
    return false
  }
}

/**
 * May the requesting frame use the Chromium clipboard-sanitized-write
 * permission? Only the exact hub origins, and ONLY the write direction —
 * clipboard reads from embedded web content stay denied everywhere.
 */
export function isHermesHubClipboardWrite(origin: string | null | undefined): boolean {
  return isHermesHubOrigin(origin)
}
