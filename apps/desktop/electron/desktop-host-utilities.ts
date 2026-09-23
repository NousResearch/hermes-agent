import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

interface DesktopMediaProtocolDeps {
  createMediaProtocolHandler: (options: any) => any
  protocol: { handle: (scheme: string, handler: any) => void }
  MEDIA_PROTOCOL: string
  ensureNativeAccessToken: (baseUrl: string) => Promise<any>
  fetchLocalMedia: (...args: any[]) => any
  electronNet: { fetch: (...args: any[]) => Promise<any> }
  getOauthSessionForUrl: (url: string) => { fetch: (...args: any[]) => Promise<any> } | null
  resolveReadableFileForIpc: (filePath: string, options: { purpose: string }) => Promise<{ resolvedPath: string }>
  backendDialClaims: { run: (scope: string, work: () => Promise<any>) => Promise<any> }
  backendScopeKey: (connectionId: string, profile: string) => string
  ensureRegistryBackend: (connectionId: string, profile: string) => Promise<any>
  ensureBackend: (profile: string) => Promise<any>
}

export function createDesktopMediaProtocolRuntime(deps: DesktopMediaProtocolDeps) {
  const {
    createMediaProtocolHandler, protocol, MEDIA_PROTOCOL, ensureNativeAccessToken, fetchLocalMedia,
    electronNet, getOauthSessionForUrl, resolveReadableFileForIpc, backendDialClaims,
    backendScopeKey, ensureRegistryBackend, ensureBackend
  } = deps

  function registerMediaProtocol() {
    const handler = createMediaProtocolHandler({
      ensureRemoteBearer: baseUrl => ensureNativeAccessToken(baseUrl),
      // Answer local files ourselves: Electron's file:// loader ignores Range and
      // returns the whole body as 200 without Accept-Ranges, which makes <video>
      // unseekable (seekable=[0,0]).
      fetchLocal: fetchLocalMedia,
      fetchRemote: (url, headers, method) =>
        electronNet.fetch(url, {
          bypassCustomProtocolHandlers: true,
          credentials: 'omit',
          headers,
          method
        }),
      fetchRemoteWithCookies: (url, headers, method) => {
        const oauthSession = getOauthSessionForUrl(url)

        if (!oauthSession) {
          throw new Error('OAuth session partition is unavailable.')
        }

        return oauthSession.fetch(url, {
          bypassCustomProtocolHandlers: true,
          credentials: 'include',
          headers,
          method
        })
      },
      resolveLocalFile: async filePath => {
        const { resolvedPath } = await resolveReadableFileForIpc(filePath, { purpose: 'Media stream' })

        return resolvedPath
      },
      // Claim-guarded (#90812): a media stream load can race a renderer's own
      // reconnect dial for the same (connectionId, profile) scope; coalescing
      // here avoids bootstrapping a second SSH tunnel / remote dashboard.
      resolveRemoteConnection: ({ connectionId, profile }) =>
        backendDialClaims.run(backendScopeKey(connectionId, profile), () =>
          connectionId ? ensureRegistryBackend(connectionId, profile) : ensureBackend(profile)
        )
    })

    protocol.handle(MEDIA_PROTOCOL, handler)
  }

  return { registerMediaProtocol }
}

interface WslFontDeps {
  isWsl: boolean
  fs: Pick<typeof fs, 'statSync' | 'readFileSync' | 'mkdirSync' | 'writeFileSync'>
  path: Pick<typeof path, 'join'>
  app: { getPath: (name: string) => string }
  spawn: (...args: any[]) => any
  rememberLog: (message: string) => void
}

export function ensureWslWindowsFonts(deps: WslFontDeps) {
  const { isWsl, fs, path, app, spawn, rememberLog } = deps

  if (!isWsl) {
    return
  }

  const fontsDir = ['/mnt/c/Windows/Fonts', '/mnt/c/windows/fonts'].find(candidate => {
    try {
      return fs.statSync(candidate).isDirectory()
    } catch {
      return false
    }
  })

  if (!fontsDir) {
    return
  }

  try {
    const confDir = path.join(app.getPath('home'), '.config', 'fontconfig', 'conf.d')
    const confPath = path.join(confDir, '99-hermes-wsl-windows-fonts.conf')
    let existing = ''

    try {
      existing = fs.readFileSync(confPath, 'utf8')
    } catch {
      existing = ''
    }

    if (existing.includes(fontsDir)) {
      return
    }

    fs.mkdirSync(confDir, { recursive: true })
    fs.writeFileSync(
      confPath,
      `<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n  <dir>${fontsDir}</dir>\n</fontconfig>\n`
    )
    rememberLog(`[fonts] wired WSL Windows fonts for renderer: ${fontsDir}`)

    const cache = spawn('fc-cache', ['-f', fontsDir], { detached: true, stdio: 'ignore' })
    cache.on('error', () => undefined)
    cache.unref()
  } catch (error) {
    rememberLog(`[fonts] WSL font setup skipped: ${error.message}`)
  }
}

export function makeDashboardReadyFile(userDataPath: string) {
  const dir = path.join(userDataPath, 'backend-ready')
  fs.mkdirSync(dir, { recursive: true })

  return path.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto.randomBytes(6).toString('hex')}.json`)
}

export function recentHermesLog(hermesLog: string[]) {
  return hermesLog.slice(-20).join('\n')
}

// Atomic file write: temp + rename (atomic on all platforms). Prevents
// partial writes on crash/power loss that corrupt JSON config files.
export function writeFileAtomic(targetPath: string, data: string, encoding?: BufferEncoding) {
  const tmp = targetPath + '.tmp'
  fs.writeFileSync(tmp, data, encoding)
  fs.renameSync(tmp, targetPath)
}
