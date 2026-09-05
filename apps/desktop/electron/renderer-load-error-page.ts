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
 *
 * Both used to leave the user staring at a blank window with the only
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

  const title = 'Hermes couldn\u2019t start the desktop UI'
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
  button {
    margin-top: 16px;
    padding: 8px 18px;
    border: 0;
    border-radius: 6px;
    background: #4f7cff;
    color: #fff;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #6b90ff; }
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
  ${reloadButtonJs(details)}
</main>
</body>
</html>`
}

/** Minimal structural surface of BrowserWindow used here. */
export interface LoadErrorWindowLike {
  loadURL: (url: string) => Promise<unknown>
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

  try {
    await win.loadURL(url)
  } catch {
    // The white screen is strictly better than an unhandled rejection here;
    // the log line from the caller still tells the story.
  }
}
