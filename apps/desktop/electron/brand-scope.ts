/** Parse the admin-portal login cookies into the brand the desktop chat should use. */

const APP_ID = /^[a-z0-9][a-z0-9-]{0,63}$/
const PORTAL_ORIGIN = 'https://admin.intelli-verse-x.ai'

/** The popup may show the login pages only. Anything else means sign-in is finished. */
export function isPortalLoginUrl(raw: string): boolean {
  let url: URL

  try {
    url = new URL(raw)
  } catch {
    return false
  }

  if (url.origin !== PORTAL_ORIGIN) {
    return false
  }

  const path = url.pathname

  if (
    path.startsWith('/_next') ||
    path.startsWith('/api/') ||
    path.startsWith('/videos/') ||
    path.startsWith('/favicon')
  ) {
    return true
  }

  return (
    path === '/' ||
    path.startsWith('/login') ||
    path.startsWith('/forgot-password') ||
    path.startsWith('/reset-password') ||
    path.startsWith('/sign-up') ||
    path.startsWith('/auth/')
  )
}

export type LoginNavigation = 'allow' | 'finish' | 'stay'

/**
 * The portal bounces `/login` to the dashboard when an old access token is
 * still stored. That bounce is not a finished sign-in: the OTP cookie lasts
 * 12 hours and the access token lasts days. Only a proven session may close
 * the window. Anything else stays on the login page.
 */
export function loginNavigationAction(url: string, signedIn: boolean): LoginNavigation {
  if (isPortalLoginUrl(url)) {
    return 'allow'
  }

  return signedIn ? 'finish' : 'stay'
}
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

export interface PortalCookie {
  name: string
  value: string
}

export interface BrandSession {
  signedIn: boolean
  email: string
  appIds: string[]
  activeAppId: string
  isSuper: boolean
  /** Paid Capabilities. Super admin is always unlocked. Unsigned local use is unlocked. */
  desktopUnlocked: boolean
}

export const signedOutBrand = (): BrandSession => ({
  signedIn: false,
  email: '',
  appIds: [],
  activeAppId: '',
  isSuper: false,
  desktopUnlocked: false
})

export function parsePinValue(raw: string, now = Date.now()): { appIds: string[] } | null {
  const dot = raw.lastIndexOf('.')

  if (dot <= 0 || dot >= raw.length - 1) {
    return null
  }

  const body = raw.slice(0, dot)
  const parts = body.split('|')
  const exp = Number(parts[parts.length - 1])

  if (!Number.isFinite(exp) || now > exp) {
    return null
  }

  const appIds: string[] = []

  for (const piece of (parts[0] ?? '').split('+')) {
    const appId = piece.trim().toLowerCase()

    if (APP_ID.test(appId) && !appIds.includes(appId)) {
      appIds.push(appId)
    }
  }

  return appIds.length ? { appIds } : null
}

export function parseOtpEmail(raw: string, now = Date.now()): string | null {
  const dot = raw.indexOf('.')

  if (dot <= 0 || dot >= raw.length - 1) {
    return null
  }

  const body = raw.slice(0, dot)

  let payload = ''

  try {
    payload = Buffer.from(body, 'base64url').toString('utf8')
  } catch {
    return null
  }

  const [email, expStr] = payload.split('|')
  const exp = Number(expStr)
  const normalized = (email ?? '').trim().toLowerCase()

  if (!EMAIL.test(normalized) || !Number.isFinite(exp) || now > exp) {
    return null
  }

  return normalized
}

/**
 * A portal login counts when the OTP session is unexpired and either a brand
 * pin or the super-admin marker is present. A missing pin is not a super admin.
 * A pin cookie that fails to parse signs the user out.
 */
export function brandSessionFromCookies(cookies: PortalCookie[], now = Date.now()): BrandSession {
  const otp = cookies.find(cookie => cookie.name === 'otp_verified')?.value ?? ''
  const email = parseOtpEmail(otp, now)

  if (!email) {
    return signedOutBrand()
  }

  const pinRaw = cookies.find(cookie => cookie.name === 'portal_pinned_app')?.value ?? ''

  if (pinRaw) {
    const pin = parsePinValue(pinRaw, now)

    if (!pin) {
      return signedOutBrand()
    }

    return {
      signedIn: true,
      email,
      appIds: pin.appIds,
      activeAppId: pin.appIds[0] ?? '',
      isSuper: false,
      desktopUnlocked: false
    }
  }

  const superEmail = parseOtpEmail(cookies.find(cookie => cookie.name === 'otp_super')?.value ?? '', now)

  if (!superEmail || superEmail !== email) {
    return signedOutBrand()
  }

  return {
    signedIn: true,
    email,
    appIds: [],
    activeAppId: '',
    isSuper: true,
    desktopUnlocked: true
  }
}

export function selectBrandApp(session: BrandSession, appId: string): BrandSession {
  const next = appId.trim().toLowerCase()

  if (!session.signedIn) {
    return session
  }

  if (!next) {
    return session.isSuper ? { ...session, activeAppId: '' } : session
  }

  if (!APP_ID.test(next)) {
    return session
  }

  if (!session.isSuper && !session.appIds.includes(next)) {
    return session
  }

  const appIds = session.isSuper && !session.appIds.includes(next) ? [...session.appIds, next] : session.appIds

  return { ...session, appIds, activeAppId: next }
}
