/**
 * Visible renderer-load error page.
 *
 * The white-screen failure modes this exists for:
 *
 * 1. TORN BUNDLE after an update (#95575): `hermes update` replaces the app
 *    while its files are locked (antivirus, a still-running instance, an
 *    interrupted Windows replace), leaving index.html and its hashed chunks
 *    from DIFFERENT generations. The window loads, then dies on the first
 *    lazy import ("Failed to fetch dynamically imported module") — a white
 *    screen that no amount of restarting fixes.
 * 2. LOAD FAILURE: a missing/blocked index.html surfaces only as a bare
 *    ERR_FILE_NOT_FOUND window (see #39484) with a log line nobody sees.
 * 3. CRASH LOOP: the renderer dies (`render-process-gone`: GPU driver fault,
 *    OOM) faster than the bounded auto-reload can help. Once the reload
 *    budget is spent the window used to go dead with no explanation; now it
 *    shows the cause with "Restart Hermes" and "Restart with software
 *    rendering" (the latter persists `--disable-gpu`, see chromium-flags.ts).
 *    Pattern from the Claude desktop app's crash screen (v2.110.0 changelog).
 *
 * All three used to leave the user staring at a blank window with the only
 * explanation in `logs/desktop.log`. This module renders the failure INTO
 * the window — error code, what is missing, how to repair — with a Reload
 * button, so the white screen becomes a diagnosable, actionable surface.
 *
 * Pure + injectable so it is testable without booting Electron: the page is
 * a self-contained data: URL (no network, no file access), so `loadURL` can
 * never itself fail on a torn install.
 */

export interface RendererLoadErrorDetails {
  /** Chromium error code, e.g. -6 (ERR_FILE_NOT_FOUND) or its name. */
  errorCode?: number | string | undefined
  /** Human description of the failure, e.g. the renderer bundle is torn. */
  errorDescription?: string
  /** The URL that failed to load, when known. */
  url?: string
  /** Module files index.html declares but that are missing on disk. */
  missingAssets?: string[]
  /** Repair command hint, e.g. `hermes desktop --force-build`. */
  repairHint?: string
  /**
   * URL to navigate to when the user clicks Reload. On a data: page
   * `location.reload()` would just re-render the error page, so recovery
   * must target the real renderer URL. Omitted → the button reloads in
   * place (harmless: the caller's load-failure policy re-surfaces).
   */
  reloadUrl?: string
  /**
   * Page heading. Default names a load failure; the crash-loop caller passes
   * "Hermes' desktop UI keeps crashing" so the copy matches the event.
   */
  title?: string
  /**
   * Recovery buttons besides Reload. Each navigates to a `hermes-recovery:`
   * URL that `loadRendererLoadErrorPage` intercepts on `will-navigate` and
   * turns into the matching handler call in the main process — a data: page
   * has no preload bridge, and the intercept keeps the page dependency-free.
   */
  recovery?: RecoveryActions
}

/** Main-process handlers behind the recovery buttons. */
export interface RecoveryActions {
  /** "Restart Hermes": relaunch the whole app (a fresh GPU process and renderer). */
  restart?: () => void
  /**
   * "Restart with software rendering": persist `--disable-gpu` for every
   * future launch, then relaunch. Omitted when GPU acceleration is already
   * off (nothing left to try on that axis).
   */
  restartSoftwareRendering?: () => void
}

export type RecoveryActionName = keyof RecoveryActions

export const RECOVERY_URL_SCHEME = 'hermes-recovery:'

const RECOVERY_ACTIONS: Record<RecoveryActionName, { path: string; label: string }> = {
  restart: { path: 'restart', label: 'Restart Hermes' },
  restartSoftwareRendering: { path: 'restart-software-rendering', label: 'Restart with software rendering' }
}

/** The recovery action a `will-navigate` URL names, or null for any other URL. */
export function parseRecoveryAction(url: unknown): RecoveryActionName | null {
  if (typeof url !== 'string' || !url.startsWith(RECOVERY_URL_SCHEME)) {
    return null
  }

  const path = url.slice(RECOVERY_URL_SCHEME.length).replace(/^\/+/, '')

  for (const [name, spec] of Object.entries(RECOVERY_ACTIONS) as Array<[RecoveryActionName, { path: string }]>) {
    if (spec.path === path) {
      return name
    }
  }

  return null
}

/**
 * Plain-English cause for a `render-process-gone` reason, so the crash page
 * says "ran out of memory" rather than `reason=oom exitCode=-9`.
 */
export function describeRendererCrashCause(details: { reason?: string; exitCode?: number | string }): string {
  const reason = String(details.reason || '')
  const exit = details.exitCode === undefined || details.exitCode === null ? '' : ` (exit code ${details.exitCode})`

  switch (reason) {
    case 'oom':
      return `The desktop UI ran out of memory${exit}.`

    case 'crashed':
      return `The desktop UI process crashed${exit}. Repeated crashes on one machine usually point at the GPU driver.`

    case 'killed':
      return `The desktop UI process was killed by the system${exit}.`

    case 'launch-failed':
      return `The desktop UI process could not be started${exit}.`

    case 'integrity-failure':
      return `The desktop UI process failed a code-integrity check${exit}.`

    case 'abnormal-exit':
      return `The desktop UI process exited abnormally${exit}.`

    default:
      return `The desktop UI process stopped unexpectedly${reason ? ` (${reason})` : ''}${exit}.`
  }
}

/**
 * Escape a ``JSON.stringify`` result for embedding inside an inline
 * ``<script>`` element.  JSON does not escape ``<``, ``>``, ``&`` (nor
 * U+2028/U+2029), so a reloadUrl containing ``</script><script>…`` would
 * terminate the script block and let an attacker-controlled URL inject
 * markup/script into the error page.
 */
function escapeInlineScriptJson(value: string): string {
  return value
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

function reloadButtonJs(details: RendererLoadErrorDetails): string {
  const target = details.reloadUrl
    ? `location.replace(${escapeInlineScriptJson(JSON.stringify(details.reloadUrl))})`
    : 'location.reload()'

  return (
    '<button id="reload" type="button">Reload</button>\n' +
    `  <script>document.getElementById("reload").addEventListener("click", () => ${target})</script>`
  )
}

/**
 * Recovery buttons as plain links: a data: page cannot reach the main process
 * any other way, and `loadRendererLoadErrorPage` turns the navigation into the
 * handler call before Chromium ever tries to resolve the scheme.
 */
function recoveryButtonsHtml(recovery?: RecoveryActions): string {
  if (!recovery) {
    return ''
  }

  const buttons = (Object.keys(RECOVERY_ACTIONS) as RecoveryActionName[])
    .filter(name => typeof recovery[name] === 'function')
    .map(name => {
      const spec = RECOVERY_ACTIONS[name]

      return `<a class="button" href="${RECOVERY_URL_SCHEME}${spec.path}">${escapeHtml(spec.label)}</a>`
    })

  return buttons.length === 0 ? '' : `\n  ${buttons.join('\n  ')}`
}

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function missingAssetsList(missingAssets?: string[]): string {
  const assets = (missingAssets ?? []).slice(0, 5)

  if (assets.length === 0) {
    return ''
  }

  const items = assets.map(asset => `<li><code>${escapeHtml(asset)}</code></li>`).join('')

  return (
    `<p>The renderer bundle is missing ${missingAssets!.length} module file(s) ` +
    `(first ${assets.length} shown) — the last update replaced the app while ` +
    `its files were locked.</p><ul>${items}</ul>`
  )
}

/**
 * Build the self-contained error page. Deliberately dependency-free: no
 * stylesheets, no images, no fetch — a data: URL must render from a blank
 * origin with zero network access.
 */
export function buildRendererLoadErrorPage(details: RendererLoadErrorDetails = {}): string {
  const code =
    details.errorCode === undefined || details.errorCode === null ? '' : ` (${escapeHtml(details.errorCode)})`

  const title = escapeHtml(details.title || 'Hermes couldn\u2019t start the desktop UI')
  const description = escapeHtml(details.errorDescription || 'The desktop renderer failed to load.')
  const url = details.url ? `<p><code>${escapeHtml(details.url)}</code></p>` : ''
  const repair = details.repairHint ? `<p>Repair with: <code>hermes desktop --force-build</code></p>` : ''

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0b0e14;
    color: #e6e6e6;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main {
    max-width: 560px;
    padding: 32px;
    border: 1px solid #2b2f3a;
    border-radius: 12px;
    background: #11151d;
  }
  h1 { font-size: 18px; margin: 0 0 12px; }
  p { font-size: 14px; line-height: 1.5; margin: 8px 0; }
  code {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    font-size: 12px;
    background: #1a1f2a;
    padding: 2px 6px;
    border-radius: 4px;
    word-break: break-all;
  }
  ul { margin: 8px 0; padding-left: 20px; font-size: 13px; }
  button, a.button {
    display: inline-block;
    margin-right: 8px;
    text-decoration: none;
    margin-top: 16px;
    padding: 8px 18px;
    border: 0;
    border-radius: 6px;
    background: #4f7cff;
    color: #fff;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover, a.button:hover { background: #6b90ff; }
</style>
</head>
<body>
<main>
  <h1>${title}</h1>
  <p>${description}${code}</p>
  ${url}
  ${missingAssetsList(details.missingAssets)}
  ${repair}
  <p>If this keeps happening, check <code>logs/desktop.log</code> and try
  <code>hermes desktop --force-build</code>, then restart the app.</p>
  ${reloadButtonJs(details)}${recoveryButtonsHtml(details.recovery)}
</main>
</body>
</html>`
}

/** Minimal structural surface of BrowserWindow used here. */
export interface LoadErrorWindowLike {
  loadURL: (url: string) => Promise<unknown>
  /** Present on a real BrowserWindow; the recovery intercept needs it. */
  webContents?: {
    on: (event: 'will-navigate', listener: (event: { preventDefault: () => void }, url: string) => void) => unknown
    removeListener?: (event: 'will-navigate', listener: (...args: any[]) => void) => unknown
  }
}

const DATA_URL_PREFIX = 'data:text/html;charset=utf-8,'

/**
 * Load the visible error page into a window, replacing the white screen.
 * Always resolves — loadURL is the one call that could reject, and a
 * rejection must not be allowed to turn the recovery surface itself blank.
 */
export async function loadRendererLoadErrorPage(
  win: LoadErrorWindowLike,
  details: RendererLoadErrorDetails = {}
): Promise<void> {
  const url = `${DATA_URL_PREFIX}${encodeURIComponent(buildRendererLoadErrorPage(details))}`

  installRecoveryIntercept(win, details.recovery)

  try {
    await win.loadURL(url)
  } catch {
    // The white screen is strictly better than an unhandled rejection here;
    // the log line from the caller still tells the story.
  }
}

/**
 * Turn a recovery button's `hermes-recovery:` navigation into its handler.
 * Any other navigation (the Reload button's `location.replace`) passes
 * through untouched, and the listener removes itself once it has fired so a
 * window that recovers does not keep an intercept around.
 */
export function installRecoveryIntercept(win: LoadErrorWindowLike, recovery?: RecoveryActions): void {
  const contents = win.webContents

  if (!recovery || !contents || typeof contents.on !== 'function') {
    return
  }

  const listener = (event: { preventDefault: () => void }, url: string) => {
    const action = parseRecoveryAction(url)

    if (!action) {
      return
    }

    event.preventDefault()

    const handler = recovery[action]

    if (typeof handler === 'function') {
      contents.removeListener?.('will-navigate', listener)
      handler()
    }
  }

  contents.on('will-navigate', listener)
}
