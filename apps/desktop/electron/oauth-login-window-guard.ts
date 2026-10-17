/**
 * Load-failure and deadline guard for the interactive OAuth login window.
 *
 * #80733: when the remote gateway accepts the TCP connection but never
 * answers (a hung `hermes serve` — zombie processes squatting on the port),
 * `openOauthLoginWindow`'s interactive window loads nothing and stays a
 * permanently blank white box. Only the hidden Cloud-recovery path had a
 * bounded deadline; the interactive path had neither a `did-fail-load`
 * handler nor a timeout, and the promise never settled.
 *
 * This module owns the decision logic, pure and injectable so it unit-tests
 * without booting Electron:
 *
 *  - `shouldSurfaceInteractiveLoginFailure` — which load failures belong to
 *    the user (fail the window into the visible error page) and which are
 *    normal OAuth navigation churn to ignore.
 *  - `buildOauthLoginTimeoutPage` — the self-contained error/retry page
 *    rendered INTO the login window when the deadline trips or the load
 *    fails, so the blank window becomes "gateway not responding, retry".
 */

/** Minimal structural surface of the login window used here. */
export interface LoginWindowLike {
  loadURL: (url: string) => Promise<unknown>
}

export interface OauthLoginTimeoutPageDetails {
  /** The gateway URL that didn't respond, e.g. http://box:9119. */
  gatewayUrl?: string
  /** True when the deadline tripped (gateway accepted the connection but never answered). */
  timedOut?: boolean
  /** Chromium error name/code when the /login load itself failed. */
  errorCode?: number | string | undefined
}

const DATA_URL_PREFIX = 'data:text/html;charset=utf-8,'

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/**
 * Escape a ``JSON.stringify`` result for embedding inside an inline
 * ``<script>`` element (JSON does not escape ``<``/``>``/``&``).
 */
function escapeInlineScriptJson(value: string): string {
  return value
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

/**
 * Whether a `did-fail-load`/loadURL error on the interactive login window
 * should surface to the user. OAuth sign-in is a navigation storm — provider
 * redirects, the callback bounce, user-initiated stops — and Electron reports
 * every aborted navigation as an error. Only real failures (unreachable
 * host, DNS, timeout, TLS, non-2xx landing page) render the error page.
 *
 * `isExpectedOauthNavigationAbort`-style ERR_ABORTED (-3) churn is filtered
 * by the caller; this predicate classifies everything else as user-visible.
 */
export function shouldSurfaceInteractiveLoginFailure(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return true
  }

  const code = 'code' in error ? Number((error as { code?: unknown }).code) : Number.NaN

  // -3 ERR_ABORTED: a navigation superseded this load (provider redirect,
  // callback bounce, window close). Not a gateway failure. Electron carries
  // it as `code` on did-fail-load errors but some paths only put it in the
  // message, so match both.
  if (code === -3 || (error instanceof Error && /\bERR_ABORTED\b|\(-3\)/.test(error.message))) {
    return false
  }

  return true
}

/**
 * Build the in-window error page for a login that cannot proceed. Dependency-
 * free (data: URL semantics): the gateway is by definition not answering, so
 * the page must render with zero network access. The Retry button navigates
 * back to the gateway's /login URL, re-running the interactive flow.
 */
export function buildOauthLoginTimeoutPage(details: OauthLoginTimeoutPageDetails = {}): string {
  const gateway = escapeHtml(details.gatewayUrl || 'the gateway')

  const code =
    details.errorCode === undefined || details.errorCode === null ? '' : ` (${escapeHtml(details.errorCode)})`

  const reason = details.timedOut
    ? 'connected but never responded — the gateway process is hung or overloaded. Restart it on the host (e.g. restart the systemd service) and retry.'
    : `could not be reached${code}. Check the URL, then retry.`

  const retryTarget = details.gatewayUrl
    ? `location.replace(${escapeInlineScriptJson(JSON.stringify(`${details.gatewayUrl.replace(/\/+$/, '')}/login`))})`
    : 'location.reload()'

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sign-in unavailable</title>
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
    max-width: 480px;
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
  <h1>Can’t reach the sign-in page</h1>
  <p><code>${gateway}</code> ${reason}</p>
  <p>This is a gateway problem, not a sign-out: your saved session is fine
  once the backend responds again. If it keeps happening, check
  <code>logs/desktop.log</code> and the backend’s health on the host.</p>
  <button id="retry" type="button">Retry</button>
  <script>document.getElementById("retry").addEventListener("click", () => ${retryTarget})</script>
</main>
</body>
</html>`
}

/**
 * Load the error page into the login window. Always resolves — a rejection
 * here must never leave the window blank again (that is the bug this fixes).
 */
export async function loadOauthLoginTimeoutPage(
  win: LoginWindowLike,
  details: OauthLoginTimeoutPageDetails = {}
): Promise<void> {
  const url = `${DATA_URL_PREFIX}${encodeURIComponent(buildOauthLoginTimeoutPage(details))}`

  try {
    await win.loadURL(url)
  } catch {
    // Strictly better than the blank window; the caller's log line still
    // tells the story.
  }
}
