/**
 * Window-open policy for every BrowserWindow's webContents.
 *
 * Every external URL the desktop opens on purpose goes through the audited
 * `hermes:openExternal` IPC channel (`openExternalUrl` in main.ts: http/https/
 * mailto allowlist, guarded file:). The `window.open` / `target=_blank` path
 * that reaches `setWindowOpenHandler` is therefore only ever driven by content
 * we did NOT initiate — most dangerously untrusted HTML in sandboxed
 * `allow-scripts` iframes (artifact previews, inline preview directives).
 *
 * GHSA-9f4c-93c8-jc8g (CVE-2026-70608): a sandboxed iframe without
 * `allow-popups` and without a user gesture can still reach this handler via
 * the OpenURL navigation path. If the handler opens `details.url` as a side
 * effect, a malicious artifact forces the user's OS browser to an attacker URL.
 * There is no fixed Electron 40.x, so the defence lives here regardless of the
 * pin: deny every request and never open a URL from this handler.
 */

export interface WindowOpenRequestLike {
  url: string
}

export interface WindowOpenDecision {
  action: 'deny'
}

/**
 * `origin` only — a denied URL can carry query credentials, signed-URL tokens
 * or attacker-controlled text, none of which belongs in a persisted log.
 */
export function describeDeniedUrl(url: string): string {
  try {
    const parsed = new URL(url)

    return parsed.origin === 'null' ? parsed.protocol : parsed.origin
  } catch {
    return '<unparseable>'
  }
}

/**
 * Build a `setWindowOpenHandler` callback that denies unconditionally.
 * `onDenied` is logging-only and receives the sanitized origin; a throwing
 * observer must not be able to change the decision.
 */
export function createWindowOpenHandler(
  onDenied?: (origin: string) => void
): (details: WindowOpenRequestLike) => WindowOpenDecision {
  return details => {
    try {
      onDenied?.(describeDeniedUrl(details.url))
    } catch {
      // observer failure is not a reason to reconsider the decision
    }

    return { action: 'deny' }
  }
}

/**
 * Wire the always-deny window-open policy onto an auth-flow window
 * (OAuth gateway login, portal sign-in, silent portal renewal).
 *
 * These windows load REMOTE content we do not initiate — the OAuth redirect
 * chain passes through third-party IDP pages, and the portal page itself is
 * fetched over the network — so a `window.open` reaching their webContents
 * is content-driven, exactly the GHSA-9f4c-93c8-jc8g shape. The label
 * distinguishes the deny log lines by flow.
 */
export function wireAuthWindowOpenPolicy(
  win: { webContents: { setWindowOpenHandler: (handler: (details: WindowOpenRequestLike) => WindowOpenDecision) => void } },
  label: string,
  log?: (line: string) => void
): void {
  win.webContents.setWindowOpenHandler(
    createWindowOpenHandler(origin => log?.(`[window-open] ${label} denied: ${origin}`))
  )
}

/**
 * Session-level guard for the OAuth/portal partitions: cancel any download
 * those windows trigger.
 *
 * The auth windows exist solely to complete sign-in — every consumer is a
 * cookie read (`hasOauthSessionCookie`, `hasLivePortalSession`, ...) after
 * a navigation — so a real download is never a legitimate part of any flow.
 * Remote IDP/portal content (or a Content-Disposition: attachment response
 * on a redirect hop) can otherwise reach `will-download` with NO handler:
 * `installDownloadHandling` wires only `session.defaultSession`, so the raw
 * Chromium save dialog would open with the process cwd as the default
 * directory and an extensionless attacker-chosen filename. Same exposure
 * the link-title partition closes via `guardLinkTitleSession`.
 */
export function guardAuthSessionDownloads(
  partitionSession: { on: (event: string, handler: (event: unknown, item: { cancel: () => void }) => void) => void },
  label: string,
  log?: (line: string) => void
): void {
  try {
    partitionSession.on('will-download', (_event, item) => {
      log?.(`[auth-download] ${label} cancelled`)

      item.cancel()
    })
  } catch {
    // best-effort; worst case is a spurious download dialog
  }
}

/**
 * Deny EVERY permission request from the OAuth/portal auth partitions.
 *
 * `installMediaPermissions` wires a permission request/check handler ONLY on
 * `session.defaultSession` (a media-capture-only allowlist for voice
 * conversations). The auth partitions get Chromium's default behavior, in
 * which remote IDP/portal content in the auth windows can request and
 * receive notifications, geolocation, midi, clipboard-read, and more —
 * permission prompts from a sign-in flow the user never sanctioned.
 *
 * No permission is ever legitimate for completing sign-in (every consumer is
 * a cookie read after a navigation), so deny-all is the correct policy —
 * stricter than the default session's media-only allowlist.
 */
export function guardAuthSessionPermissions(
  partitionSession: {
    setPermissionRequestHandler: (
      handler: (webContents: unknown, permission: string, callback: (granted: boolean) => void, details: unknown) => void
    ) => void
    setPermissionCheckHandler?: (handler: (webContents: unknown, permission: string) => boolean) => void
  },
  label: string,
  log?: (line: string) => void
): void {
  const deny = (permission: string) => {
    log?.(`[auth-permission] ${label} denied: ${permission}`)
  }

  try {
    partitionSession.setPermissionRequestHandler((_webContents, permission, callback) => {
      deny(permission)

      callback(false)
    })

    // Synchronous check handler: Chromium consults it directly on some
    // platforms (e.g. getUserMedia on Windows); without it the check would
    // consult the request handler's default and grant.
    partitionSession.setPermissionCheckHandler?.((_webContents, permission) => {
      deny(permission)

      return false
    })
  } catch {
    // best-effort; degraded environments must not take sign-in down
  }
}
