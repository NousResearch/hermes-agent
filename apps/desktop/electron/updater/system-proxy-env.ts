import { session } from 'electron'

/**
 * The OS proxy for the update's child processes (#123387).
 *
 * git, Python's urllib, npm, uv and electron-builder's downloads all honour
 * HTTPS_PROXY/HTTP_PROXY/NO_PROXY, but none of them reads the macOS or Windows
 * system proxy. A Desktop user whose only proxy is the system one got updates
 * that went direct and died where direct access is blocked. Chromium already
 * resolves the system proxy (PAC included), so ask it for the hosts the update
 * reaches and hand the answer to the children as proxy env. Explicit proxy env
 * always wins: it is returned untouched.
 *
 * The env is for the update's children only. The hand-off script, the staged
 * updater and `hermes update` start the relaunched Desktop and gateways, which
 * would otherwise inherit it as if the user had set it: every LAN host the PAC
 * sends direct would be proxied, and the next update would read the stale value
 * as an explicit setting. So every variable the Desktop adds is named in
 * UPDATE_PROXY_MARKER and every NO_PROXY it extends is saved under SAVED_PREFIX;
 * restoreUserProxyEnv (and its ports in posix.sh, windows.ps1 and gateway
 * start) puts the user's own env back.
 */

const PROXY_ENV_KEYS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']
const PROXY_SET_KEYS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']
const NO_PROXY_KEYS = ['NO_PROXY', 'no_proxy']

/** Comma list of the variables the Desktop added to an update child's env. */
export const UPDATE_PROXY_MARKER = 'HERMES_UPDATE_SYSTEM_PROXY'
/** `${SAVED_PREFIX}NO_PROXY` holds the user's own NO_PROXY that the Desktop extended. */
export const SAVED_PREFIX = 'HERMES_UPDATE_USER_'

// git + the GitHub API and assets, the release pointer, npm, PyPI + uv, Electron downloads.
export const UPDATE_HOSTS = [
  'github.com',
  'api.github.com',
  'codeload.github.com',
  'objects.githubusercontent.com',
  'release-assets.githubusercontent.com',
  'hermes-assets.nousresearch.com',
  'registry.npmjs.org',
  'pypi.org',
  'files.pythonhosted.org',
  'releases.astral.sh'
]

// Loopback, RFC 1918 and mDNS names never go through a remote proxy. curl (git)
// and uv read the CIDR entries; urllib and npm match the names.
export const LOCAL_BYPASS = [
  'localhost',
  '127.0.0.1',
  '::1',
  '127.0.0.0/8',
  '10.0.0.0/8',
  '172.16.0.0/12',
  '192.168.0.0/16',
  '.local'
]

function pacKind(answer: string): [string, string] {
  const [kind = '', hostPort = ''] = (answer.split(';')[0] ?? '').trim().split(/\s+/)

  return [kind.toUpperCase(), hostPort]
}

/**
 * Chromium's PAC-style answer ("PROXY host:port; DIRECT") as a proxy URL every
 * update child can use; null for DIRECT and for kinds they cannot use. Python's
 * urllib speaks neither SOCKS nor TLS to a proxy (it sends a plaintext CONNECT),
 * npm reads no SOCKS env, httpx refuses socks4, and the update's own git calls
 * drop injected git config, so only a plain PROXY answer is exportable.
 */
export function proxyUrlFromPac(answer: string): string | null {
  const [kind, hostPort] = pacKind(answer)

  return kind === 'PROXY' && hostPort ? `http://${hostPort}` : null
}

/** Undo what systemProxyEnv added: drop the marked variables, restore saved NO_PROXY values. */
export function restoreUserProxyEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  for (const key of (env[UPDATE_PROXY_MARKER] ?? '').split(',')) {
    if ([...PROXY_SET_KEYS, ...NO_PROXY_KEYS].includes(key)) {
      delete env[key]
    }
  }

  for (const key of NO_PROXY_KEYS) {
    const saved = env[`${SAVED_PREFIX}${key}`]

    if (saved !== undefined) {
      env[key] = saved
      delete env[`${SAVED_PREFIX}${key}`]
    }
  }

  delete env[UPDATE_PROXY_MARKER]

  return env
}

/** The user's own NO_PROXY entries, extended (never replaced) by the local bypass and DIRECT update hosts. */
function noProxyEnv(userEnv: NodeJS.ProcessEnv, direct: string[]): { env: NodeJS.ProcessEnv; added: string[] } {
  const user = `${userEnv.no_proxy ?? ''},${userEnv.NO_PROXY ?? ''}`
  const entries = user.split(',').map((entry: string): string => entry.trim())
  const value = [...new Set([...entries.filter(Boolean), ...LOCAL_BYPASS, ...direct])].join(',')
  const env: NodeJS.ProcessEnv = {}
  const added: string[] = []

  for (const key of NO_PROXY_KEYS) {
    env[key] = value

    if (userEnv[key] === undefined) {
      added.push(key)
    } else {
      env[`${SAVED_PREFIX}${key}`] = userEnv[key]
    }
  }

  return { env, added }
}

export async function systemProxyEnv(
  env: NodeJS.ProcessEnv,
  resolveProxy: (url: string) => Promise<string>,
  hosts: string[] = UPDATE_HOSTS,
  note: (message: string) => void = () => {}
): Promise<NodeJS.ProcessEnv> {
  // A value an earlier update marked as the Desktop's is never the user's own.
  const userEnv = restoreUserProxyEnv({ ...env })

  if (PROXY_ENV_KEYS.some((key: string): boolean => Boolean(userEnv[key]?.trim()))) {
    return {}
  }

  // Each host is asked on its own: a PAC can send each one a different way.
  const answers = await Promise.all(
    hosts.map(async (host: string) => {
      // A PAC script that fails to load leaves Chromium itself going direct.
      const raw = await resolveProxy(`https://${host}/`).catch((): string => 'DIRECT')

      return { host, kind: pacKind(raw)[0] || 'DIRECT', proxy: proxyUrlFromPac(raw) }
    })
  )

  const routed = answers.filter(answer => answer.kind !== 'DIRECT')
  const proxies = new Set(routed.map(answer => answer.proxy))

  if (!routed.length) {
    return {}
  }

  // One env proxy serves every host, so it is exported only when the system
  // sends each update host either through that one plain proxy or direct.
  if (proxies.size > 1 || proxies.has(null)) {
    const answered = routed.map(answer => `${answer.host}=${answer.kind}`).join(', ')

    note(`[updates] system proxy not passed to the update: ${answered} (only one plain PROXY answer can be)`)

    return {}
  }

  const proxy = routed[0].proxy as string
  const direct = answers.filter(answer => answer.kind === 'DIRECT').map(answer => answer.host)
  const noProxy = noProxyEnv(userEnv, direct)

  return {
    ...Object.fromEntries(PROXY_SET_KEYS.map((key: string) => [key, proxy])),
    ...noProxy.env,
    [UPDATE_PROXY_MARKER]: [...PROXY_SET_KEYS, ...noProxy.added].join(',')
  }
}

/** The system proxy env for an update child; empty outside Electron (unit tests, scripts). */
export async function updateProxyEnv(
  env: NodeJS.ProcessEnv = process.env,
  note?: (message: string) => void
): Promise<NodeJS.ProcessEnv> {
  const defaultSession = session?.defaultSession

  return defaultSession ? systemProxyEnv(env, url => defaultSession.resolveProxy(url), UPDATE_HOSTS, note) : {}
}
