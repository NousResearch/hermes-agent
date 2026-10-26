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
 */

const PROXY_ENV_KEYS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']

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

const LOCAL_BYPASS = ['localhost', '127.0.0.1', '::1']

const PAC_SCHEMES: Record<string, string> = {
  PROXY: 'http',
  HTTPS: 'https',
  SOCKS: 'socks5h',
  SOCKS5: 'socks5h',
  SOCKS4: 'socks4'
}

/** Chromium's PAC-style answer ("PROXY host:port; DIRECT") as a proxy URL; null for DIRECT. */
export function proxyUrlFromPac(answer: string): string | null {
  const [kind = '', hostPort = ''] = (answer.split(';')[0] ?? '').trim().split(/\s+/)
  const scheme = PAC_SCHEMES[kind.toUpperCase()]

  return scheme && hostPort ? `${scheme}://${hostPort}` : null
}

export async function systemProxyEnv(
  env: NodeJS.ProcessEnv,
  resolveProxy: (url: string) => Promise<string>,
  hosts: string[] = UPDATE_HOSTS
): Promise<NodeJS.ProcessEnv> {
  if (PROXY_ENV_KEYS.some((key: string): boolean => Boolean(env[key]?.trim()))) {
    return {}
  }

  const answers = await Promise.all(
    hosts.map(async (host: string) => ({
      host,
      // A PAC script that fails to load leaves Chromium itself going direct.
      proxy: proxyUrlFromPac(await resolveProxy(`https://${host}/`).catch((): string => 'DIRECT'))
    }))
  )

  const proxy = answers.find(answer => answer.proxy)?.proxy

  if (!proxy) {
    return {}
  }

  const direct = answers.filter(answer => !answer.proxy).map(answer => answer.host)
  const noProxy = env.NO_PROXY ?? env.no_proxy ?? [...LOCAL_BYPASS, ...direct].join(',')

  return {
    HTTPS_PROXY: proxy,
    https_proxy: proxy,
    HTTP_PROXY: proxy,
    http_proxy: proxy,
    NO_PROXY: noProxy,
    no_proxy: noProxy
  }
}

/** The system proxy env for an update child; empty outside Electron (unit tests, scripts). */
export async function updateProxyEnv(env: NodeJS.ProcessEnv = process.env): Promise<NodeJS.ProcessEnv> {
  const defaultSession = session?.defaultSession

  return defaultSession ? systemProxyEnv(env, url => defaultSession.resolveProxy(url)) : {}
}
