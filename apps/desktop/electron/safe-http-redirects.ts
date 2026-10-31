/**
 * One bounded redirect ladder for automatic link-metadata fetches (titles and
 * favicons). Nothing here follows a `Location` by itself: the first URL and
 * every hop pass `admit` before any byte is sent, and the whole chain shares
 * one deadline instead of one per hop.
 */

export interface RedirectHop {
  redirectUrl?: string
  statusCode: number
}

export interface SafeRedirectResult<T> {
  finalUrl: string
  /** The destination policy (or the ladder's own limits) stopped the chain. */
  refused: boolean
  response: T | null
}

async function beforeDeadline<T>(promise: Promise<T>, deadline: number): Promise<T> {
  const remaining = deadline - Date.now()

  if (remaining <= 0) {
    throw new Error('Metadata fetch timed out')
  }

  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Metadata fetch timed out')), remaining)
      })
    ])
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}

/**
 * Follow at most `maxRedirects` hops. `admit` returns what the hop may use
 * (e.g. the vetted DNS answers to pin) or null to refuse it; `fetchHop`
 * performs exactly one request and never follows a redirect itself.
 */
export async function fetchWithSafeRedirects<T extends RedirectHop, A>(
  rawUrl: string,
  fetchHop: (url: string, remainingMs: number, admission: A) => Promise<T>,
  options: {
    admit: (url: string) => Promise<A | null>
    dispose?: (response: T) => Promise<void> | void
    maxRedirects: number
    timeoutMs: number
  }
): Promise<SafeRedirectResult<T>> {
  const deadline = Date.now() + Math.max(1, options.timeoutMs)
  let currentUrl = String(rawUrl || '').trim()

  for (let redirects = 0; ; redirects += 1) {
    let admission: A | null = null

    try {
      admission = await beforeDeadline(Promise.resolve(options.admit(currentUrl)), deadline)
    } catch {
      admission = null
    }

    if (admission === null) {
      return { finalUrl: currentUrl, refused: true, response: null }
    }

    let response: T

    try {
      response = await beforeDeadline(fetchHop(currentUrl, Math.max(1, deadline - Date.now()), admission), deadline)
    } catch {
      return { finalUrl: currentUrl, refused: false, response: null }
    }

    const location = response.redirectUrl?.trim() ?? ''

    if (response.statusCode < 300 || response.statusCode >= 400 || !location) {
      return { finalUrl: currentUrl, refused: false, response }
    }

    await options.dispose?.(response)

    // A chain that long is a loop or abuse; the next tier re-walking it would
    // be no safer, so it counts as a refusal.
    if (redirects >= options.maxRedirects) {
      return { finalUrl: currentUrl, refused: true, response: null }
    }

    try {
      currentUrl = new URL(location, currentUrl).toString()
    } catch {
      return { finalUrl: currentUrl, refused: true, response: null }
    }
  }
}
