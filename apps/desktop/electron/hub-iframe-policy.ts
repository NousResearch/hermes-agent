/**
 * Capability carve-outs for the embedded Skills Hub iframe ONLY.
 *
 * The Skills Hub picker (Bot Mode) embeds the public docs site
 * (`https://github.com/seven0070/Rabbit-/tree/main/website/docs/skills?embed=picker`). Three default denials
 * make that embed nearly unusable — window.open is killed by the CVE-2026-70608
 * handler, clipboard writes by the session permission handlers, and the frame
 * can drift off the picker URL into the un-embeddable docs site. Every carve-out
 * below is gated on the EXACT hub origin: a sandboxed artifact iframe (opaque
 * origin `null`) or any other guest frame never qualifies.
 */

export const RABBIT_HUB_ORIGIN = 'https://github.com/seven0070/Rabbit-'

const HUB_ORIGINS = new Set([RABBIT_HUB_ORIGIN])

/**
 * Exact-origin membership — `origin` comes from the frame's own `origin`
 * property (URL parsing), never from a string we control. `new URL(...).origin`
 * is `null` for sandboxed frames and unique for `data:` URLs, so those can
 * never alias the hub.
 */
export function isRabbitHubOrigin(origin: string | null | undefined): boolean {
  return typeof origin === 'string' && HUB_ORIGINS.has(origin)
}

/** The URL schemes a trusted hub frame may delegate to the OS browser. */
const HUB_EXTERNAL_SCHEMES = new Set(['http:', 'https:', 'mailto:'])

/**
 * May a trusted-origin window.open request be handed to the audited external
 * opener? http/https/mailto only — the same allowlist `openExternalUrl` in
 * external-open.ts enforces; anything else (file:, javascript:, custom
 * schemes) stays denied with no side effect.
 */
export function isRabbitHubExternalUrl(url: string): boolean {
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
export function isRabbitHubClipboardWrite(origin: string | null | undefined): boolean {
  return isRabbitHubOrigin(origin)
}
