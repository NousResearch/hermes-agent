/**
 * Environment-configured HTTP proxy support for the main process's outbound
 * calls to api.github.com.
 *
 * Node's http/https modules do not read `HTTP_PROXY`/`HTTPS_PROXY` the way
 * curl, git and browsers do: `https.get(url, { headers, timeout })` with no
 * agent always dials the host directly, so behind a corporate proxy the
 * desktop's passive update check never left the network and reported
 * "api.github.com did not answer within 10 seconds" to users whose proxy
 * environment was set correctly and visible to Hermes.exe (#114736).
 *
 * The check now runs with the agent built here, so the variables that make
 * git and npm work make it work. An https-proxy-agent is a CONNECT request
 * plus a TLS handshake over that socket, which is short enough to not warrant
 * a new dependency.
 *
 * Same convention as the Python side (agent/proxy_bypass.py): HTTPS_PROXY,
 * HTTP_PROXY, ALL_PROXY in either casing — first non-empty wins — and NO_PROXY
 * decides which hosts stay direct.
 *
 * ponytail: NO_PROXY takes `*`, host, host:port and `.domain`/`*.domain`
 * (no CIDR ranges) — the only caller dials api.github.com; add a range matcher
 * when a caller here anchors on private networks.
 */

import http from 'node:http'
import https from 'node:https'
import type net from 'node:net'
import tls from 'node:tls'

export type ProxyEnv = Record<string, string | undefined>

const PROXY_ENV_KEYS = ['HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'https_proxy', 'http_proxy', 'all_proxy']
const NO_PROXY_ENV_KEYS = ['NO_PROXY', 'no_proxy']

// The request's own `timeout` only starts once a socket is handed to it, so
// the CONNECT exchange needs its own bound: a black-holing proxy would
// otherwise hang the update check forever instead of failing it.
const CONNECT_TIMEOUT_MS = 10_000

function firstNonEmpty(env: ProxyEnv, keys: string[]): string {
  for (const key of keys) {
    const value = String(env?.[key] ?? '').trim()

    if (value) {
      return value
    }
  }

  return ''
}

/** `host`, `host:port` or `[v6]:port` -> lowercased host + port (null when absent). */
function splitHostPort(value: string): { host: string; port: number | null } {
  const raw = String(value || '').trim()

  if (!raw) {
    return { host: '', port: null }
  }

  if (raw.startsWith('[')) {
    const end = raw.indexOf(']')

    if (end > 0) {
      const rest = raw.slice(end + 1)

      return { host: raw.slice(1, end).toLowerCase(), port: /^:\d+$/.test(rest) ? Number(rest.slice(1)) : null }
    }
  }

  // A second colon means a bare IPv6 literal, not a port.
  const colon = raw.lastIndexOf(':')

  if (colon > 0 && raw.indexOf(':') === colon && /^\d+$/.test(raw.slice(colon + 1))) {
    return { host: raw.slice(0, colon).toLowerCase(), port: Number(raw.slice(colon + 1)) }
  }

  return { host: raw.replace(/\.$/, '').toLowerCase(), port: null }
}

function noProxyExcludes(host: string, port: number, env: ProxyEnv): boolean {
  const target = String(host || '')
    .toLowerCase()
    .replace(/\.$/, '')

  const entries = NO_PROXY_ENV_KEYS.map(key => env?.[key] ?? '')
    .join(',')
    .split(/[\s,]+/)
    .filter(Boolean)

  return entries.some(entry => {
    if (entry === '*') {
      return true
    }

    const { host: entryHost, port: entryPort } = splitHostPort(entry)

    // A NO_PROXY entry with a port only bypasses that port.
    if (!entryHost || (entryPort !== null && entryPort !== port)) {
      return false
    }

    // `.example.com` and `*.example.com` both mean the apex plus subdomains.
    const suffix = entryHost.replace(/^\*/, '').replace(/^\./, '')

    return target === suffix || target.endsWith(`.${suffix}`)
  })
}

/**
 * The proxy URL that serves `url`, normalized, or null when no proxy applies:
 * nothing configured, a non-HTTP target, or a NO_PROXY entry matching it.
 * A scheme-less value (`proxy.corp.example:8080`, what corporate docs print)
 * is read as an HTTP proxy, and a malformed value is treated as unset rather
 * than fatal — this only feeds a passive update check.
 */
export function proxyUrlFor(url: string, env: ProxyEnv = process.env): string | null {
  const raw = firstNonEmpty(env, PROXY_ENV_KEYS)

  if (!raw) {
    return null
  }

  let target: URL

  try {
    target = new URL(url)
  } catch {
    return null
  }

  if (target.protocol !== 'http:' && target.protocol !== 'https:') {
    return null
  }

  const port = target.port ? Number(target.port) : target.protocol === 'https:' ? 443 : 80

  if (noProxyExcludes(target.hostname, port, env)) {
    return null
  }

  try {
    const proxy = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`)

    return proxy.protocol === 'http:' || proxy.protocol === 'https:' ? proxy.href : null
  } catch {
    return null
  }
}

function connectThroughProxy(proxy: URL, target: string): Promise<net.Socket> {
  return new Promise((resolve, reject) => {
    const secureProxy = proxy.protocol === 'https:'
    const client = secureProxy ? https : http

    // `agent: false`: the CONNECT response hands the socket to us, and the
    // shared keep-alive agents would keep it in their pool.
    const request = client.request({
      host: proxy.hostname,
      port: Number(proxy.port) || (secureProxy ? 443 : 80),
      method: 'CONNECT',
      path: target,
      agent: false,
      timeout: CONNECT_TIMEOUT_MS,
      headers: proxyAuthHeaders(proxy)
    })

    request.once('connect', (response, socket, head) => {
      if (response.statusCode !== 200) {
        socket.destroy()
        reject(new Error(`Proxy ${proxy.host} refused CONNECT ${target} (HTTP ${response.statusCode}).`))

        return
      }

      // Bytes the proxy sent after its response headers are already target
      // data (the TLS ServerHello can beat our handshake); they must not be
      // dropped on the floor.
      if (head?.length) {
        socket.unshift(head)
      }

      resolve(socket)
    })

    request.once('timeout', () => {
      request.destroy(new Error(`Proxy ${proxy.host} did not answer CONNECT ${target} within ${CONNECT_TIMEOUT_MS}ms.`))
    })

    // A proxy may answer the CONNECT with an ordinary response (407 auth
    // required, 403 policy, 502 it could not reach the host). Node only emits
    // 'connect' for a tunnel, so without this the check would sit until the
    // timeout; failing on the answer is both faster and more honest.
    request.once('response', response => {
      reject(new Error(`Proxy ${proxy.host} answered HTTP ${response.statusCode} instead of tunnelling ${target}.`))
      request.destroy()
    })

    request.once('error', reject)
    request.end()
  })
}

function proxyAuthHeaders(proxy: URL): Record<string, string> {
  if (!proxy.username && !proxy.password) {
    return {}
  }

  const credentials = `${decodeURIComponent(proxy.username)}:${decodeURIComponent(proxy.password)}`

  return { 'Proxy-Authorization': `Basic ${Buffer.from(credentials).toString('base64')}` }
}

/** TLS over the tunnel socket; injectable so tests need no key pair. */
export interface ProxyTunnelDeps {
  tlsConnect?: (options: any, onSecure: () => void) => any
}

/**
 * An https.Agent whose every connection is a CONNECT tunnel through `proxy`.
 * Not keep-alive: the update check runs twice an hour at most, and a pooled
 * tunnel socket through a corporate proxy is a connection to debug, not to
 * hoard.
 */
export class ProxyTunnelAgent extends https.Agent {
  readonly proxyUrl: string

  readonly #proxy: URL
  readonly #tlsConnect: (options: any, onSecure: () => void) => any

  constructor(proxy: URL, deps: ProxyTunnelDeps = {}) {
    super({ keepAlive: false })

    this.proxyUrl = proxy.href
    this.#proxy = proxy
    this.#tlsConnect = deps.tlsConnect || tls.connect
  }

  createConnection(options: any, callback?: any): any {
    const target = `${options.host}:${options.port || 443}`

    connectThroughProxy(this.#proxy, target).then(
      socket => {
        const onError = (error: Error) => callback(error)

        // createConnection has to return a TLS socket, so the handshake —
        // which https.Agent would normally do inside its own createConnection —
        // happens here, over the tunnel.
        const secure = this.#tlsConnect(
          {
            socket,
            servername: options.servername || options.host,
            ALPNProtocols: ['http/1.1'],
            rejectUnauthorized: options.rejectUnauthorized,
            ca: options.ca,
            cert: options.cert,
            key: options.key,
            passphrase: options.passphrase
          },
          () => {
            secure.removeListener('error', onError)
            callback(null, secure)
          }
        )

        secure.once('error', onError)
      },
      (error: Error) => callback(error)
    )
  }
}

/**
 * The `agent` for an https request to `url`, or undefined when the environment
 * names no proxy (which keeps Node's default agent and the pre-#114736
 * behaviour exactly).
 */
export function proxyAgentFor(
  url: string,
  env: ProxyEnv = process.env,
  deps: ProxyTunnelDeps = {}
): ProxyTunnelAgent | undefined {
  const proxyUrl = proxyUrlFor(url, env)

  return proxyUrl ? new ProxyTunnelAgent(new URL(proxyUrl), deps) : undefined
}
