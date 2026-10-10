#!/usr/bin/env node

import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync, renameSync, chmodSync } from 'node:fs'
import { createServer, request as httpRequest } from 'node:http'
import { request as httpsRequest } from 'node:https'
import { homedir } from 'node:os'
import { dirname, extname, join, normalize, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import net from 'node:net'
import tls from 'node:tls'
import { createHmac, randomBytes, scryptSync, timingSafeEqual, randomUUID } from 'node:crypto'
import { hostHeaderHostname, isAcceptedHost, LOOPBACK_HOSTS } from './host-validation.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(here, '..', '..', '..')
const distRoot = resolve(here, 'dist')
const desktopRoot = resolve(here, '..')
const uiFontsRoot = resolve(desktopRoot, 'node_modules/@nous-research/ui/dist/fonts')
const args = process.argv.slice(2)

function flag(name, fallback) {
  const index = args.indexOf(name)
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback
}

const host = flag('--host', process.env.HERMES_DESKTOP_WEB_HOST || '127.0.0.1')
const port = Number(flag('--port', process.env.HERMES_DESKTOP_WEB_PORT || '13043'))
const backendUrlOverride = validateUrl(flag('--backend-url', process.env.HERMES_DESKTOP_WEB_BACKEND_URL || ''))
const hermesHome = resolve(process.env.HERMES_HOME || join(homedir(), '.hermes'))
const stateRoot = join(hermesHome, 'desktop-web')
const registryFile = join(stateRoot, 'connections.json')
const maxBodyBytes = 16 * 1024 * 1024

const proxyPrefixes = ['/api/', '/api', '/auth/', '/auth', '/login', '/logout', '/oauth/', '/oauth']
const localRoutes = new Set([
  '/api/web-connections',
  '/api/web-connection-config',
  '/api/web-connection-config/save',
  '/api/web-connection-config/test',
  '/api/web-connection-session',
  '/api/web-connections/save',
  '/api/web-connections/remove',
  '/api/web-connections/primary',
  '/api/web-connections/launch-mode',
  '/api/web-connections/last-used',
  '/api/web-connections/test'
])

const mime = {
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
}

function json(response, status, value, headers = {}) {
  response.writeHead(status, {
    'cache-control': 'no-store',
    'content-type': 'application/json; charset=utf-8',
    ...headers
  })
  response.end(JSON.stringify(value))
}

function safeRegistry(registry) {
  return {
    version: 1,
    primary: registry.primary,
    connections: registry.connections.map(({ token, headers, ...connection }) => ({
      ...connection,
      tokenSet: Boolean(token),
      tokenPreview: null
    }))
  }
}

function fallbackRegistry() {
  return {
    version: 1,
    primary: '',
    connections: []
  }
}

function readRegistry() {
  try {
    const value = JSON.parse(readFileSync(registryFile, 'utf8'))
    if (!value || typeof value !== 'object' || !Array.isArray(value.connections)) return runtimeRegistry(fallbackRegistry())
    const connections = value.connections.filter(row => row && typeof row === 'object' && typeof row.id === 'string' && typeof row.label === 'string')
    const primary = typeof value.primary === 'string' && connections.some(row => row.id === value.primary)
      ? value.primary
      : (backendUrlOverride ? '' : (connections[0]?.id || ''))
    return runtimeRegistry({ version: 1, primary, connections })
  } catch {
    return runtimeRegistry(fallbackRegistry())
  }
}

function runtimeRegistry(registry) {
  if (!backendUrlOverride) return registry
  const local = {
    id: 'local',
    kind: 'local',
    label: 'This Desktop Web backend',
    url: backendUrlOverride.toString(),
    authMode: 'token'
  }
  const connections = registry.connections.filter(row => row.id !== 'local')
  const primary = connections.some(row => row.id === registry.primary) ? registry.primary : 'local'
  return { ...registry, primary, connections: [local, ...connections] }
}

function writeRegistry(registry) {
  mkdirSync(stateRoot, { recursive: true, mode: 0o700 })
  // The owned local connection is runtime state, not a user-configured row.
  // Never persist its private child URL or allow registry mutations to replace
  // or remove it. An empty persisted primary rehydrates to local at runtime.
  const stored = backendUrlOverride
    ? {
        ...registry,
        primary: registry.primary === 'local' ? '' : registry.primary,
        connections: registry.connections.filter(row => row.id !== 'local')
      }
    : registry
  const temporary = `${registryFile}.${process.pid}.${randomUUID()}.tmp`
  writeFileSync(temporary, `${JSON.stringify(stored, null, 2)}\n`, { mode: 0o600 })
  chmodSync(temporary, 0o600)
  renameSync(temporary, registryFile)
  chmodSync(registryFile, 0o600)
}

function validateUrl(value) {
  try {
    const url = new URL(String(value || '').trim())
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null
    url.hash = ''
    url.search = ''
    url.pathname = url.pathname.replace(/\/+$/, '')
    return url
  } catch {
    return null
  }
}

function selectedConnection(id) {
  const registry = readRegistry()
  const requested = String(id || '').trim()
  const selected = requested ? registry.connections.find(row => row.id === requested) : registry.connections.find(row => row.id === registry.primary)
  if (requested && !selected) throw new Error('Connection not found.')
  return selected || null
}

function connectionTarget(connection) {
  const url = validateUrl(connection?.url)
  if (!url) throw new Error('Desktop Web backend URL is not configured.')
  return url
}

function requestedConnection(requestUrl) {
  const parsed = new URL(requestUrl || '/', `http://${host}:${port}`)
  return { parsed, connection: selectedConnection(parsed.searchParams.get('connectionId')) }
}

function publicUrlFromConfig() {
  const env = validateUrl(process.env.HERMES_DESKTOP_WEB_PUBLIC_URL)
  if (env) return env
  return validateUrl(configValue('desktop_web', 'public_url'))
}

function configValue(section, key) {
  try {
    const lines = readFileSync(join(hermesHome, 'config.yaml'), 'utf8').split(/\r?\n/)
    let inSection = false
    let sectionIndent = -1
    let nestedIndent = -1
    let inNested = false
    for (const line of lines) {
      if (/^\S/.test(line) && !line.startsWith('desktop_web:')) {
        inSection = false
        inNested = false
      }
      const sectionMatch = line.match(/^(\s*)desktop_web:\s*(?:#.*)?$/)
      if (sectionMatch) {
        inSection = true
        sectionIndent = sectionMatch[1].length
        inNested = false
        continue
      }
      if (!inSection) continue
      const nestedMatch = line.match(/^(\s*)basic_auth:\s*(?:#.*)?$/)
      if (nestedMatch && nestedMatch[1].length > sectionIndent) {
        inNested = section === 'basic_auth'
        nestedIndent = nestedMatch[1].length
        continue
      }
      if (section === 'basic_auth' && inNested) {
        const valueMatch = line.match(/^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$/)
        if (valueMatch && valueMatch[1].length <= nestedIndent) {
          inNested = false
        } else if (valueMatch && valueMatch[1].length > nestedIndent && valueMatch[2] === key) {
          return parseConfigScalar(valueMatch[3])
        }
      } else if (section === 'desktop_web') {
        const valueMatch = line.match(/^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$/)
        if (valueMatch && valueMatch[1].length > sectionIndent && valueMatch[2] === key) {
          return parseConfigScalar(valueMatch[3])
        }
      }
    }
  } catch {
    // Configuration is optional; callers apply safe defaults.
  }
  return ''
}

function parseConfigScalar(value) {
  const raw = String(value || '').trim()
  if (!raw || raw.startsWith('#')) return ''
  if (raw.startsWith('"') && raw.endsWith('"')) {
    try { return JSON.parse(raw) } catch { return raw.slice(1, -1) }
  }
  if (raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1).replace(/''/g, "'")
  return raw.replace(/\s+#.*$/, '').trim()
}

function authConfig() {
  const env = name => String(process.env[name] || '').trim()
  const config = key => configValue('basic_auth', key)
  const username = env('HERMES_DESKTOP_WEB_BASIC_AUTH_USERNAME') || config('username')
  const envPassword = env('HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD')
  const passwordHash = envPassword ? '' : (env('HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD_HASH') || config('password_hash'))
  const password = envPassword || (!passwordHash ? config('password') : '')
  const secretText = env('HERMES_DESKTOP_WEB_BASIC_AUTH_SECRET') || config('secret')
  let ttl = Number(env('HERMES_DESKTOP_WEB_BASIC_AUTH_TTL_SECONDS') || config('session_ttl_seconds') || 43200)
  if (!Number.isFinite(ttl) || ttl < 60) ttl = 43200
  return { username, passwordHash, password, secretText, ttl: Math.floor(ttl) }
}

function decodeSecret(value) {
  if (!value) return randomBytes(32)
  try {
    const decoded = Buffer.from(value, 'base64')
    if (decoded.length >= 16) return decoded
  } catch {}
  try {
    const decoded = Buffer.from(value, 'hex')
    if (decoded.length >= 16) return decoded
  } catch {}
  const raw = Buffer.from(value, 'utf8')
  return raw.length >= 16 ? raw : null
}

function parseScryptHash(encoded) {
  try {
    const [scheme, nText, rText, pText, saltText, digestText] = String(encoded || '').split('$')
    const N = Number(nText), r = Number(rText), p = Number(pText)
    if (scheme !== 'scrypt' || !Number.isInteger(N) || !Number.isInteger(r) || !Number.isInteger(p) || N < 2 || N > 2 ** 20 || r < 1 || r > 32 || p < 1 || p > 16) return null
    const salt = Buffer.from(saltText, 'base64')
    const digest = Buffer.from(digestText, 'base64')
    if (!salt.length || !digest.length || digest.length > 128) return null
    return { N, r, p, salt, digest }
  } catch {
    return null
  }
}

function derivePasswordHash(password) {
  const salt = randomBytes(16)
  const digest = scryptSync(password, salt, 32, { N: 2 ** 14, r: 8, p: 1, maxmem: 32 * 1024 * 1024 })
  return { N: 2 ** 14, r: 8, p: 1, salt, digest }
}

const desktopAuthConfig = authConfig()
const desktopAuthSecret = decodeSecret(desktopAuthConfig.secretText)
const desktopPasswordHash = desktopAuthConfig.passwordHash
  ? parseScryptHash(desktopAuthConfig.passwordHash)
  : (desktopAuthConfig.password ? derivePasswordHash(desktopAuthConfig.password) : null)
const desktopAuthEnabled = Boolean(desktopAuthConfig.username && desktopPasswordHash && desktopAuthSecret)
const desktopSessionCookie = 'hermes_desktop_web_session'

function hostAllowed(request) {
  const publicUrl = publicUrlFromConfig()
  const trustedPublicHosts = publicUrl ? new Set([publicUrl.hostname.toLowerCase()]) : new Set()
  return isAcceptedHost(request.headers.host, host, trustedPublicHosts)
}

function requestIsLoopback(request) {
  return LOOPBACK_HOSTS.has(hostHeaderHostname(request.headers.host))
}

function cookieSecure(request) {
  const publicUrl = publicUrlFromConfig()
  const requestHost = hostHeaderHostname(request.headers.host)
  return Boolean(request.socket.encrypted || (publicUrl && publicUrl.protocol === 'https:' && requestHost === publicUrl.hostname.toLowerCase()) || (request.headers['x-forwarded-proto'] === 'https' && publicUrl && requestHost === publicUrl.hostname.toLowerCase()))
}

function signSession(payload) {
  const raw = Buffer.from(JSON.stringify(payload), 'utf8')
  const signature = createHmac('sha256', desktopAuthSecret).update(raw).digest()
  return Buffer.concat([raw, signature]).toString('base64url')
}

function verifySession(value) {
  if (!desktopAuthEnabled || !value) return false
  try {
    const blob = Buffer.from(value, 'base64url')
    if (blob.length <= 32) return false
    const raw = blob.subarray(0, -32)
    const signature = blob.subarray(-32)
    const expected = createHmac('sha256', desktopAuthSecret).update(raw).digest()
    if (!timingSafeEqual(signature, expected)) return false
    const payload = JSON.parse(raw.toString('utf8'))
    return payload?.kind === 'access' && payload?.sub === desktopAuthConfig.username && Number(payload?.exp) > Math.floor(Date.now() / 1000)
  } catch {
    return false
  }
}

function sessionCookie(request) {
  const cookies = String(request.headers.cookie || '').split(';')
  for (const item of cookies) {
    const [name, ...parts] = item.trim().split('=')
    if (name === desktopSessionCookie) return parts.join('=')
  }
  return ''
}

function verifyPassword(password) {
  if (!desktopPasswordHash) return false
  try {
    const actual = scryptSync(String(password || ''), desktopPasswordHash.salt, desktopPasswordHash.digest.length, { N: desktopPasswordHash.N, r: desktopPasswordHash.r, p: desktopPasswordHash.p, maxmem: 128 * desktopPasswordHash.N * desktopPasswordHash.r + 1024 * 1024 })
    return actual.length === desktopPasswordHash.digest.length && timingSafeEqual(actual, desktopPasswordHash.digest)
  } catch {
    return false
  }
}

function desktopAuthAllowed(request) {
  return requestIsLoopback(request) || (desktopAuthEnabled && verifySession(sessionCookie(request)))
}

function authCookie(request, value, maxAge) {
  return `${desktopSessionCookie}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}${cookieSecure(request) ? '; Secure' : ''}`
}

function loginPage(next = '/') {
  const safeNext = String(next || '/').startsWith('/') && !String(next).startsWith('//') ? String(next) : '/'
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hermes Desktop Web — Sign in</title><style>body{font:16px system-ui,sans-serif;max-width:28rem;margin:12vh auto;padding:1rem;color:#eee;background:#171717}form{display:grid;gap:.8rem}input,button{font:inherit;padding:.7rem;border-radius:.4rem;border:1px solid #555}button{cursor:pointer;background:#fff;color:#111}#error{color:#f88;min-height:1.3em}</style></head><body><h1>Sign in to Hermes Desktop Web</h1><form method="post" action="/api/desktop-web-auth/login"><label>Username<input name="username" autocomplete="username" required></label><label>Password<input name="password" type="password" autocomplete="current-password" required></label><input type="hidden" name="next" value="${safeNext.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}"><button type="submit">Sign in</button></form><p id="error"></p></body></html>`
}

function redirect(response, location, headers = {}) {
  response.writeHead(302, { location, 'cache-control': 'no-store', ...headers })
  response.end()
}

function readForm(request) {
  return readBody(request).then(body => new URLSearchParams(body))
}

async function handleDesktopAuth(request, response, parsed) {
  if (parsed.pathname === '/login' && request.method === 'GET') {
    if (requestIsLoopback(request) || (desktopAuthEnabled && verifySession(sessionCookie(request)))) {
      redirect(response, '/')
      return true
    }
    if (!desktopAuthEnabled) { json(response, 503, { error: 'Desktop Web basic authentication is not configured.' }); return true }
    response.writeHead(200, { 'cache-control': 'no-store', 'content-type': 'text/html; charset=utf-8' })
    response.end(loginPage(parsed.searchParams.get('next') || '/'))
    return true
  }
  if (parsed.pathname === '/api/desktop-web-auth/login' && request.method === 'POST') {
    if (!desktopAuthEnabled) { json(response, 503, { error: 'Desktop Web basic authentication is not configured.' }); return true }
    const form = await readForm(request)
    const username = String(form.get('username') || '')
    const password = String(form.get('password') || '')
    const suppliedUsername = Buffer.from(username)
    const configuredUsername = Buffer.from(desktopAuthConfig.username)
    const usernameOk = suppliedUsername.length === configuredUsername.length && timingSafeEqual(suppliedUsername, configuredUsername)
    const passwordOk = verifyPassword(password)
    if (!usernameOk || !passwordOk) { json(response, 401, { error: 'Invalid username or password.' }, { 'www-authenticate': 'Basic realm="Hermes Desktop Web"' }); return true }
    const now = Math.floor(Date.now() / 1000)
    const token = signSession({ sub: desktopAuthConfig.username, kind: 'access', exp: now + desktopAuthConfig.ttl })
    const next = String(form.get('next') || '/')
    const safeNext = next.startsWith('/') && !next.startsWith('//') ? next : '/'
    redirect(response, safeNext, { 'set-cookie': authCookie(request, token, desktopAuthConfig.ttl) })
    return true
  }
  if (parsed.pathname === '/logout' && (request.method === 'GET' || request.method === 'POST')) {
    redirect(response, '/login', { 'set-cookie': authCookie(request, '', 0) })
    return true
  }
  return false
}

function sameOrigin(request) {
  if (['GET', 'HEAD', 'OPTIONS'].includes(String(request.method || 'GET').toUpperCase())) return true
  const origin = request.headers.origin
  if (!origin) return true
  try {
    return new URL(origin).host === request.headers.host
  } catch {
    return false
  }
}

function isProxyPath(pathname) {
  return proxyPrefixes.some(prefix => pathname === prefix || pathname.startsWith(prefix))
}

function readBody(request) {
  return new Promise((resolveBody, reject) => {
    let body = ''
    request.on('data', chunk => {
      body += chunk
      if (Buffer.byteLength(body) > maxBodyBytes) {
        reject(new Error('Request body is too large.'))
        request.destroy()
      }
    })
    request.on('end', () => resolveBody(body))
    request.on('error', reject)
  })
}

async function readJson(request) {
  const body = await readBody(request)
  if (!body.trim()) return {}
  const value = JSON.parse(body)
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('JSON object body is required.')
  return value
}

function inputConnection(body, existing) {
  const url = body.url === undefined ? existing?.url : String(body.url || '').trim()
  if (url && !validateUrl(url)) throw new Error('A valid http(s) backend URL is required.')
  const label = String(body.label ?? existing?.label ?? 'Hermes gateway').trim().slice(0, 64)
  if (!label) throw new Error('Connection label is required.')
  const connection = {
    ...(existing || {}),
    id: String(body.id || existing?.id || `web-${randomUUID()}`),
    kind: 'remote',
    label,
    ...(url ? { url } : {}),
    ...(body.authMode === 'token' || body.authMode === 'oauth' ? { authMode: body.authMode } : {})
  }
  // Browser credentials are intentionally never persisted by this host. The
  // backend login/session cookie is kept by the browser for the active origin.
  return connection
}

async function handleLocalRoute(request, response, parsed) {
  if (!localRoutes.has(parsed.pathname)) return false
  try {
    if (parsed.pathname === '/api/web-connections' && request.method === 'GET') {
      json(response, 200, safeRegistry(readRegistry()))
      return true
    }
      if (parsed.pathname === '/api/web-connection-config' && request.method === 'GET') {
      const connection = selectedConnection(parsed.searchParams.get('connectionId'))
      json(response, 200, {
        envOverride: Boolean(process.env.HERMES_DESKTOP_WEB_PUBLIC_URL),
        mode: connection?.kind === 'local' ? 'local' : 'remote',
        profile: null,
        remoteAuthMode: connection?.authMode || 'token',
        remoteOauthConnected: false,
        remoteTokenPreview: null,
        remoteTokenSet: false,
        secureTokenStorage: false,
        remoteTokenPlainText: false,
        remoteUrl: connection?.kind === 'local' ? '' : (connection?.url || ''),
        cloudOrg: '',
        sshHost: '',
        sshUser: '',
        sshPort: null,
        sshKeyPath: '',
        sshRemoteHermesPath: '',
        sshRemoteProfile: ''
      })
      return true
    }
    if (parsed.pathname === '/api/web-connection-config/save' && request.method === 'POST') {
      const body = await readJson(request)
      const registry = readRegistry()
      if (body.mode === 'local') {
        registry.primary = 'local'
        writeRegistry(registry)
        json(response, 200, {
          ok: true,
          config: {
            mode: 'local', profile: null, remoteUrl: '', remoteAuthMode: 'token',
            remoteOauthConnected: false, remoteTokenSet: false, remoteTokenPreview: null,
            secureTokenStorage: false, remoteTokenPlainText: false,
            envOverride: Boolean(process.env.HERMES_DESKTOP_WEB_PUBLIC_URL),
            cloudOrg: '', sshHost: '', sshUser: '', sshPort: null, sshKeyPath: '',
            sshRemoteHermesPath: '', sshRemoteProfile: ''
          },
          registry: safeRegistry(readRegistry())
        })
        return true
      }
      const selected = selectedConnection()
      const existing = selected?.kind === 'local' ? undefined : selected
      const connection = inputConnection({ ...body, id: existing?.id }, existing)
      const duplicate = registry.connections.find(row => row.id !== connection.id && row.label.toLowerCase() === connection.label.toLowerCase())
      if (duplicate) throw new Error('Connection label must be unique.')
      registry.connections = existing ? registry.connections.map(row => row.id === connection.id ? connection : row) : [...registry.connections, connection]
      if (body.apply === true) registry.primary = connection.id
      else if (!registry.primary) registry.primary = connection.id
      writeRegistry(registry)
      json(response, 200, { ok: true, config: { mode: 'remote', profile: null, remoteUrl: connection.url || '', remoteAuthMode: connection.authMode || 'oauth', remoteOauthConnected: false, remoteTokenSet: false, remoteTokenPreview: null, secureTokenStorage: false, remoteTokenPlainText: false, envOverride: false, cloudOrg: '', sshHost: '', sshUser: '', sshPort: null, sshKeyPath: '', sshRemoteHermesPath: '', sshRemoteProfile: '' }, registry: safeRegistry(registry) })
      return true
    }
    if (parsed.pathname === '/api/web-connection-config/test' && request.method === 'POST') {
      const body = await readJson(request)
      const target = validateUrl(body.remoteUrl || body.url)
      if (!target) throw new Error('A valid http(s) backend URL is required.')
      const result = await probe(target)
      json(response, 200, result)
      return true
    }
    if (parsed.pathname === '/api/web-connection-session' && request.method === 'GET') {
      const connection = selectedConnection(parsed.searchParams.get('connectionId'))
      const result = connection ? await probe(connectionTarget(connection)) : { ok: false, connected: false }
      json(response, 200, { connected: result.ok === true })
      return true
    }
    if (parsed.pathname === '/api/web-connections/save' && request.method === 'POST') {
      const body = await readJson(request)
      const registry = readRegistry()
      const connection = inputConnection(body, body.id ? registry.connections.find(row => row.id === body.id) : undefined)
      if (registry.connections.some(row => row.id !== connection.id && row.label.toLowerCase() === connection.label.toLowerCase())) throw new Error('Connection label must be unique.')
      registry.connections = registry.connections.filter(row => row.id !== connection.id).concat(connection)
      if (!registry.primary) registry.primary = connection.id
      writeRegistry(registry)
      json(response, 200, { ok: true, connection: safeRegistry(registry).connections.find(row => row.id === connection.id), registry: safeRegistry(registry) })
      return true
    }
    if (parsed.pathname === '/api/web-connections/remove' && request.method === 'POST') {
      const body = await readJson(request)
      const registry = readRegistry()
      if (body.id === 'local') throw new Error('The local connection cannot be removed.')
      if (body.id === registry.primary) registry.primary = ''
      registry.connections = registry.connections.filter(row => row.id !== body.id)
      writeRegistry(registry)
      json(response, 200, { ok: true, registry: safeRegistry(registry) })
      return true
    }
    if (parsed.pathname === '/api/web-connections/primary' && request.method === 'POST') {
      const body = await readJson(request)
      const registry = readRegistry()
      if (!registry.connections.some(row => row.id === body.id)) throw new Error('Connection not found.')
      registry.primary = body.id
      writeRegistry(registry)
      json(response, 200, { ok: true, registry: safeRegistry(registry) })
      return true
    }
    if (parsed.pathname === '/api/web-connections/launch-mode' && request.method === 'POST') {
      json(response, 200, { ok: true, registry: safeRegistry(readRegistry()) })
      return true
    }
    if (parsed.pathname === '/api/web-connections/last-used' && request.method === 'POST') {
      json(response, 200, { ok: true, registry: safeRegistry(readRegistry()) })
      return true
    }
    if (parsed.pathname === '/api/web-connections/test' && request.method === 'POST') {
      const body = await readJson(request)
      const connection = selectedConnection(body.id)
      json(response, 200, await probe(connectionTarget(connection)))
      return true
    }
    json(response, 405, { error: 'Method not allowed' })
    return true
  } catch (error) {
    json(response, 400, { error: error instanceof Error ? error.message : 'Desktop Web control request failed' })
    return true
  }
}

function targetPath(target, incomingPath) {
  const prefix = target.pathname.replace(/\/+$/, '')
  const incoming = incomingPath || '/'
  return `${prefix}${incoming.startsWith('/') ? incoming : `/${incoming}`}` || '/'
}

function isOwnedBackendTarget(target) {
  return Boolean(backendUrlOverride && target.origin === backendUrlOverride.origin && target.pathname === backendUrlOverride.pathname)
}

function targetRequestPath(target, incoming, includeOwnedToken = false) {
  const search = new URLSearchParams(incoming.searchParams)
  if (isOwnedBackendTarget(target)) {
    search.delete('ticket')
    search.delete('token')
    if (includeOwnedToken) search.set('token', String(process.env.HERMES_DESKTOP_WEB_BACKEND_TOKEN || ''))
  }
  const suffix = search.toString()
  return `${targetPath(target, incoming.pathname)}${suffix ? `?${suffix}` : ''}`
}

function targetHeaders(request, target) {
  const headers = { ...request.headers }
  headers.host = target.host
  headers.origin = target.origin
  delete headers['content-length']
  // The browser only authenticates to this host with its own HttpOnly cookie.
  // Never relay that credential, or a browser-supplied native token, upstream.
  delete headers.cookie
  delete headers['x-hermes-session-token']
  if (isOwnedBackendTarget(target)) {
    delete headers.authorization
    headers['x-hermes-session-token'] = String(process.env.HERMES_DESKTOP_WEB_BACKEND_TOKEN || '')
  }
  return headers
}

function proxyHttp(request, response, target) {
  const transport = target.protocol === 'https:' ? httpsRequest : httpRequest
  const incoming = new URL(request.url || '/', `http://${host}:${port}`)
  const outgoing = transport({
    hostname: target.hostname,
    port: target.port || (target.protocol === 'https:' ? 443 : 80),
    method: request.method,
    path: targetRequestPath(target, incoming),
    headers: targetHeaders(request, target),
    rejectUnauthorized: true
  }, upstream => {
    response.writeHead(upstream.statusCode || 502, upstream.headers)
    upstream.pipe(response)
  })
  outgoing.on('error', error => {
    if (!response.headersSent) json(response, 502, { error: 'Configured Hermes backend is unavailable.' })
    else response.destroy(error)
  })
  request.pipe(outgoing)
}

function proxyWebSocket(request, socket, head, target) {
  const targetPort = Number(target.port || (target.protocol === 'https:' ? 443 : 80))
  const secure = target.protocol === 'https:'
  const connect = secure ? tls.connect({ host: target.hostname, port: targetPort, servername: target.hostname }) : net.connect(targetPort, target.hostname)
  connect.once('connect', () => {
    const incoming = new URL(request.url || '/', `http://${host}:${port}`)
    const lines = [`${request.method || 'GET'} ${targetRequestPath(target, incoming, true)} HTTP/1.1`]
    for (const [name, value] of Object.entries(request.headers)) {
      if (['host', 'origin', 'connection', 'upgrade', 'content-length', 'cookie', 'authorization', 'x-hermes-session-token'].includes(name.toLowerCase())) continue
      lines.push(`${name}: ${value}`)
    }
    if (isOwnedBackendTarget(target)) lines.push(`x-hermes-session-token: ${process.env.HERMES_DESKTOP_WEB_BACKEND_TOKEN || ''}`)
    lines.push(`host: ${target.host}`, `origin: ${target.origin}`, 'connection: Upgrade', 'upgrade: websocket', '', '')
    connect.write(lines.join('\r\n'))
    if (head?.length) connect.write(head)
    socket.pipe(connect).pipe(socket)
  })
  connect.on('error', () => socket.destroy())
  socket.on('error', () => connect.destroy())
}

async function probe(target) {
  const basePath = target.pathname.replace(/\/+$/, '')
  const probeUrl = `${target.origin}${basePath}/api/status`
  const headers = { accept: 'application/json' }
  if (isOwnedBackendTarget(target)) headers['x-hermes-session-token'] = String(process.env.HERMES_DESKTOP_WEB_BACKEND_TOKEN || '')
  try {
    const response = await fetch(probeUrl, { headers, redirect: 'manual', signal: AbortSignal.timeout(8_000) })
    await response.arrayBuffer()
    return { ok: response.status >= 200 && response.status < 400, reachable: true, authenticated: response.status !== 401 && response.status !== 403, status: response.status }
  } catch (error) {
    return { ok: false, reachable: false, error: error instanceof Error ? error.message : 'Backend probe failed.' }
  }
}

function staticFile(pathname) {
  if (pathname.startsWith('/fonts/')) {
    const name = decodeURIComponent(pathname.slice('/fonts/'.length))
    if (!name || name.includes('/') || name.includes('\\') || !name.endsWith('.woff2')) return null
    const candidate = resolve(uiFontsRoot, name)
    return candidate.startsWith(`${uiFontsRoot}/`) && existsSync(candidate) && statSync(candidate).isFile() ? candidate : null
  }
  let decoded
  try { decoded = decodeURIComponent(pathname) } catch { return null }
  const relative = normalize(decoded).replace(/^[/\\]+/, '')
  const candidate = resolve(join(distRoot, relative))
  return candidate.startsWith(`${distRoot}/`) && existsSync(candidate) && statSync(candidate).isFile() ? candidate : null
}

function serveFile(file, response) {
  response.writeHead(200, { 'cache-control': extname(file) === '.html' ? 'no-store' : 'public, max-age=31536000, immutable', 'content-type': mime[extname(file).toLowerCase()] || 'application/octet-stream' })
  createReadStream(file).pipe(response)
}

const server = createServer(async (request, response) => {
  if (!hostAllowed(request)) return json(response, 400, { error: 'Invalid Host header.' })
  const incoming = new URL(request.url || '/', `http://${host}:${port}`)
  if (!sameOrigin(request)) return json(response, 403, { error: 'Cross-origin state-changing requests are forbidden.' })
  if (await handleDesktopAuth(request, response, incoming)) return
  if (!desktopAuthAllowed(request)) {
    if (incoming.pathname.startsWith('/api/') || incoming.pathname === '/healthz') return json(response, 401, { error: 'Desktop Web authentication required.' }, { 'www-authenticate': 'Basic realm="Hermes Desktop Web"' })
    return redirect(response, `/login?next=${encodeURIComponent(`${incoming.pathname}${incoming.search}`)}`)
  }
  if (incoming.pathname === '/healthz') return json(response, 200, { ok: true, service: 'hermes-desktop-web', port, backend: backendUrlOverride?.origin || null, configured: Boolean(selectedConnection()) })
  if (await handleLocalRoute(request, response, incoming)) return
  if (isProxyPath(incoming.pathname)) {
    try {
      const connection = selectedConnection(incoming.searchParams.get('connectionId'))
      if (!connection) return json(response, 503, { error: 'Configure a Hermes backend in Desktop Web settings first.' })
      return proxyHttp(request, response, connectionTarget(connection))
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : 'Backend proxy failed.' })
    }
  }
  const file = staticFile(incoming.pathname)
  if (file) return serveFile(file, response)
  const index = join(distRoot, 'index.html')
  if (existsSync(index)) return serveFile(index, response)
  return response.end('Desktop Web is not built. Run hermes desktop-web without --skip-build.')
})

server.on('upgrade', async (request, socket, head) => {
  if (!hostAllowed(request)) return socket.destroy()
  const origin = request.headers.origin
  if (origin) {
    try {
      if (new URL(origin).host !== request.headers.host) return socket.destroy()
    } catch {
      return socket.destroy()
    }
  }
  if (!desktopAuthAllowed(request)) return socket.destroy()
  try {
    const incoming = new URL(request.url || '/', `http://${host}:${port}`)
    if (!isProxyPath(incoming.pathname)) return socket.destroy()
    const connection = selectedConnection(incoming.searchParams.get('connectionId'))
    if (!connection) return socket.destroy()
    proxyWebSocket(request, socket, head, connectionTarget(connection))
  } catch {
    socket.destroy()
  }
})

if (!Number.isInteger(port) || port < 0 || port > 65535) {
  console.error('Desktop Web port must be between 0 and 65535.')
  process.exit(2)
}

server.listen(port, host, () => {
  const actualPort = server.address().port
  console.log(`HERMES_DESKTOP_WEB_READY port=${actualPort}`)
  console.log(`  Hermes Desktop Web → http://${host}:${actualPort}`)
})
