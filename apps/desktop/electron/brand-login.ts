import { BrowserWindow, ipcMain, session, app } from 'electron'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import { loadBrandConnectors, rememberInboxAccountId, saveBrandConnector } from './brand-connectors'
import {
  brandSessionFromCookies,
  isPortalLoginUrl,
  loginNavigationAction,
  selectBrandApp,
  signedOutBrand,
  type BrandSession,
  type PortalCookie
} from './brand-scope'

const PORTAL_ORIGIN = 'https://admin.intelli-verse-x.ai'
const PARTITION = 'persist:ivx-portal'
/** These outlive the 12-hour OTP cookie and make `/login` bounce to the dashboard. */
const STALE_AUTH_COOKIES = ['accessToken', 'idToken', 'refreshToken', 'adminAccessToken', 'adminRefreshToken']

let loginWindow: BrowserWindow | null = null
let loginInFlight: Promise<BrandSession> | null = null

function sessionFile(): string {
  return path.join(app.getPath('userData'), 'ivx-brand-session.json')
}

function readSaved(): BrandSession {
  try {
    const parsed = JSON.parse(readFileSync(sessionFile(), 'utf8')) as BrandSession

    if (parsed && parsed.signedIn && typeof parsed.email === 'string') {
      return {
        signedIn: true,
        email: parsed.email,
        appIds: Array.isArray(parsed.appIds) ? parsed.appIds : [],
        activeAppId: typeof parsed.activeAppId === 'string' ? parsed.activeAppId : '',
        isSuper: parsed.isSuper === true
      }
    }
  } catch {
    // Missing or unreadable session is signed out.
  }

  return signedOutBrand()
}

function writeSaved(next: BrandSession): void {
  if (!next.signedIn) {
    rmSync(sessionFile(), { force: true })
    return
  }

  mkdirSync(path.dirname(sessionFile()), { recursive: true })
  writeFileSync(sessionFile(), JSON.stringify(next), 'utf8')
}

async function readPortalCookies(): Promise<PortalCookie[]> {
  const cookies = await session.fromPartition(PARTITION).cookies.get({})

  return cookies
    .filter(cookie => cookie.domain.replace(/^\./, '').endsWith('intelli-verse-x.ai'))
    .map(cookie => {
      let value = cookie.value

      try {
        value = decodeURIComponent(cookie.value)
      } catch {
        value = cookie.value
      }

      return { name: cookie.name, value }
    })
}

const LOGIN_LOCK = `(() => {
  if (window.__ivxLoginLock) return
  window.__ivxLoginLock = true
  const allowed = (path) =>
    path === '/' ||
    path.startsWith('/login') ||
    path.startsWith('/forgot-password') ||
    path.startsWith('/reset-password') ||
    path.startsWith('/sign-up') ||
    path.startsWith('/auth/')
  const lock = () => {
    if (!allowed(location.pathname)) document.documentElement.style.display = 'none'
  }
  for (const name of ['pushState', 'replaceState']) {
    const orig = history[name]
    history[name] = function () {
      const result = orig.apply(this, arguments)
      lock()
      return result
    }
  }
  window.addEventListener('popstate', lock)
  lock()
})()`

/** Prefer a live portal cookie. Fall back to the saved choice of app id. */
async function currentBrand(): Promise<BrandSession> {
  const live = brandSessionFromCookies(await readPortalCookies())

  if (!live.signedIn) {
    return signedOutBrand()
  }

  const saved = readSaved()
  const active =
    saved.email === live.email && saved.activeAppId && (live.isSuper || live.appIds.includes(saved.activeAppId))
      ? saved.activeAppId
      : live.activeAppId
  const next = { ...live, activeAppId: active }

  writeSaved(next)
  return next
}

async function clearStaleAuthCookies(): Promise<void> {
  const jar = session.fromPartition(PARTITION)

  await Promise.all(STALE_AUTH_COOKIES.map(name => jar.cookies.remove(`${PORTAL_ORIGIN}/`, name)))
}

function openLogin(): Promise<BrandSession> {
  if (loginInFlight) {
    if (loginWindow && !loginWindow.isDestroyed()) {
      loginWindow.focus()
    }

    return loginInFlight
  }

  const pending = new Promise<BrandSession>(resolve => {
    const win = new BrowserWindow({
      width: 960,
      height: 760,
      title: 'Sign in',
      autoHideMenuBar: true,
      center: true,
      webPreferences: {
        partition: PARTITION,
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false
      }
    })

    loginWindow = win
    let settled = false
    const portalSession = session.fromPartition(PARTITION)

    let closing = false

    const finish = (next: BrandSession) => {
      if (settled) {
        return
      }

      settled = true
      loginWindow = null
      portalSession.cookies.removeListener('changed', onCookie)
      resolve(next)
    }

    const complete = async () => {
      if (settled || closing) {
        return
      }

      closing = true

      if (!win.isDestroyed()) {
        win.hide()
      }

      let next = await currentBrand()

      if (!next.signedIn) {
        await new Promise(resolveDelay => setTimeout(resolveDelay, 600))
        next = await currentBrand()
      }

      if (!win.isDestroyed()) {
        win.close()
      }

      finish(next.signedIn ? next : signedOutBrand())
    }

    const watch = async () => {
      if (settled || closing || win.isDestroyed()) {
        return
      }

      const next = await currentBrand()

      if (next.signedIn) {
        await complete()
      }
    }

    const holdLogin = (event: Electron.Event, url: string) => {
      if (loginNavigationAction(url, false) === 'allow') {
        return
      }

      event.preventDefault()
      void (async () => {
        const next = await currentBrand()

        if (loginNavigationAction(url, next.signedIn) === 'finish') {
          await complete()
          return
        }

        if (!win.isDestroyed() && !isPortalLoginUrl(win.webContents.getURL())) {
          await win.loadURL(`${PORTAL_ORIGIN}/login`)
        }
      })()
    }

    const onCookie = (_event: Electron.Event, cookie: Electron.Cookie, _cause: string, removed: boolean) => {
      if (removed || !['otp_verified', 'portal_pinned_app', 'otp_super'].includes(cookie.name)) {
        return
      }

      void watch()
    }

    portalSession.cookies.on('changed', onCookie)
    const timer = setInterval(() => void watch(), 400)

    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    win.webContents.on('will-navigate', holdLogin)
    win.webContents.on('will-redirect', holdLogin)
    win.webContents.on('did-finish-load', () => {
      if (!win.isDestroyed() && isPortalLoginUrl(win.webContents.getURL())) {
        void win.webContents.executeJavaScript(LOGIN_LOCK).catch(() => undefined)
      }

      void watch()
    })

    win.on('closed', () => {
      clearInterval(timer)
      portalSession.cookies.removeListener('changed', onCookie)
      loginWindow = null
      if (!settled) {
        void currentBrand().then(next => finish(next.signedIn ? next : signedOutBrand()))
      }
    })
    void (async () => {
      if (!(await currentBrand()).signedIn) {
        await clearStaleAuthCookies()
      }

      if (!win.isDestroyed()) {
        await win.loadURL(`${PORTAL_ORIGIN}/login`)
        win.focus()
      }
    })()
  })

  loginInFlight = pending.finally(() => {
    loginInFlight = null
  })

  return loginInFlight
}

export function registerBrandLoginIpc(): void {
  ipcMain.handle('hermes:brand:get', () => currentBrand())
  ipcMain.handle('hermes:brand:signIn', () => openLogin())
  ipcMain.handle('hermes:brand:signOut', async () => {
    await session.fromPartition(PARTITION).clearStorageData()
    writeSaved(signedOutBrand())
    rememberInboxAccountId('')
    return signedOutBrand()
  })
  ipcMain.handle('hermes:brand:select', async (_event, appId: unknown) => {
    const current = await currentBrand()
    const next = selectBrandApp(current, typeof appId === 'string' ? appId : '')

    writeSaved(next)
    return next
  })
  ipcMain.handle('hermes:brand:connectors', async () => loadBrandConnectors(await currentBrand()))
  ipcMain.handle('hermes:brand:saveConnector', async (_event, connectorId: unknown, credential: unknown) =>
    saveBrandConnector(await currentBrand(), connectorId, credential)
  )
}
