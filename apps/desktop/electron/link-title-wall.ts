// The ladder between the curl tier and the hidden-Chromium tier of link-title
// resolution: curl (tier 1) → hidden BrowserWindow (tier 2).
//
// The title partition is cookieless (`session.fromPartition('hermes:link-titles',
// { cache: false })`), so a Google Workspace link answers tier 1 with Google's
// sign-in page — and tier 2 used to be reached *exactly* when tier 1 produced no
// usable title, i.e. for every sign-in wall. Loading that wall in the real
// hidden Chromium makes it ask the OS authenticator for a passkey: a native
// credential dialog on the user's desktop for what is only a link title.
//
// Measured shapes of the wall (curl, the app's own flags):
//   drive.google.com/drive/folders/<id>      -> accounts.google.com/v3/signin/identifier?…  (body: id="identifierId")
//   docs.google.com/document/d/<id>/edit     -> stays on docs.google.com, no <title>, body names accounts.google.com/ServiceLogin
//   script.google.com/a/<d>/macros/s/<id>/exec -> www.google.com/a/<d>/ServiceLogin?…  (no marker in the body at all)
// The last shape is why markup alone is not enough: the hosts whose cookieless
// answer can only ever be that wall are named too. A published Doc or Site still
// gets its title from the curl tier, which is what keeps these links fetchable.

const TITLE_MAX_CHARS = 240

// Strips known error/captcha titles (e.g. "GetYourGuide – Error", "Just a
// moment...") so they don't get cached as the resolved title. An error title is
// '' here, which is also the one case the renderer tier is still allowed to try.
const TITLE_ERROR_RE =
  /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

function usableTitle(value: string): string {
  return value && !TITLE_ERROR_RE.test(value) ? value : ''
}

/** Sign-in markup — the tell for a wall whose redirect curl already followed. */
const AUTH_WALL_BODY_RE =
  /accounts\.google\.com(?:\/|&#47;)(?:ServiceLogin|signin)|id="identifierId"|id="gaia_loginform"/i

/** Hosts whose only cookieless answer is a sign-in page. */
const RENDERER_WALL_HOSTS = new Set([
  'accounts.google.com',
  'console.cloud.google.com',
  'docs.google.com',
  'drive.google.com',
  'script.google.com',
  'sheets.google.com',
  'slides.google.com',
  'sites.google.com'
])

function hostOf(rawUrl: string): string {
  try {
    return new URL(rawUrl).hostname.toLowerCase()
  } catch {
    return ''
  }
}

/** Arrival URL or title of a sign-in page — the shapes that carry no markup id. */
const AUTH_WALL_URL_RE = /^accounts\.google\.com$|\/ServiceLogin\b|\/signin\b|\/o\/oauth2\//i
const AUTH_WALL_TITLE_RE = /\bsign[ -]?in\b/i

/**
 * Curl's proof that the page it landed on is a sign-in wall, decided on the
 * arrival URL → the title → the markup, in that order of reliability: an Apps
 * Script `/a/<domain>/ServiceLogin` wall carries none of the markup ids.
 */
export function isAuthWall(input: { body: string; effectiveUrl: string; title: string }): boolean {
  let arrival: URL | null = null

  try {
    arrival = input.effectiveUrl ? new URL(input.effectiveUrl) : null
  } catch {
    arrival = null
  }

  if (arrival && (AUTH_WALL_URL_RE.test(arrival.hostname) || AUTH_WALL_URL_RE.test(arrival.pathname))) {
    return true
  }

  if (AUTH_WALL_TITLE_RE.test(input.title || '')) {
    return true
  }

  return AUTH_WALL_BODY_RE.test(input.body || '')
}

/**
 * May this answer still escalate to the tier-2 hidden Chromium?
 *
 * `title` is tier 1's title after `usableTitle` — '' for both "nothing found"
 * and "found an error/captcha title". `authWall` is tier 1's proof that the page
 * it landed on is a sign-in wall.
 */
export function needsRendererFallback(input: { authWall: boolean; title: string; url: string }): boolean {
  // Tier 1 resolved a usable title — nothing left to escalate for.
  if (input.title) {
    return false
  }

  // A proven sign-in wall, or a host that can only answer with one: tier 2 would
  // load the same signed-out page in a real browser and raise the OS passkey
  // dialog. The link keeps its host/path label instead of a document title.
  if (input.authWall || RENDERER_WALL_HOSTS.has(hostOf(input.url))) {
    return false
  }

  return true
}

/**
 * Tier 1 → tier 2, with the sign-in wall never reaching the renderer. Both tiers
 * are injected so the ladder is provable without booting Electron: main.ts owns
 * the I/O, this owns the decision.
 */
export async function resolveLinkTitle(input: {
  curl: () => Promise<{ authWall: boolean; title: string }>
  renderer: () => Promise<string>
  url: string
}): Promise<string> {
  const tier1 = await input.curl().catch(() => ({ authWall: false, title: '' }))
  // A wall's own title ("Sign in - Google Accounts") is not the document's.
  const title = tier1.authWall ? '' : usableTitle((tier1.title || '').slice(0, TITLE_MAX_CHARS))

  if (!needsRendererFallback({ authWall: tier1.authWall, title, url: input.url })) {
    return title
  }

  const rendered = await input.renderer().catch(() => '')

  return usableTitle((rendered || '').slice(0, TITLE_MAX_CHARS))
}
