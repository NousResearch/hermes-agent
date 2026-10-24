import { execFileSync } from 'node:child_process'

type ProxyEnv = Record<string, string>

function valueFor(output: string, key: string): string {
  const match = output.match(new RegExp(`^\\s*${key}\\s*:\\s*(.+?)\\s*$`, 'm'))

  return match?.[1]?.trim() || ''
}

function proxyUrl(output: string, prefix: 'HTTP' | 'HTTPS'): string {
  if (valueFor(output, `${prefix}Enable`) !== '1') {
    return ''
  }

  const host = valueFor(output, `${prefix}Proxy`)
  const port = valueFor(output, `${prefix}Port`)

  if (!host || !port) {
    return ''
  }

  return `http://${host.includes(':') && !host.startsWith('[') ? `[${host}]` : host}:${port}`
}

/** Parse the proxy values emitted by `scutil --proxy` on macOS. */
export function parseMacSystemProxy(output: string): ProxyEnv {
  const http = proxyUrl(output, 'HTTP')
  const https = proxyUrl(output, 'HTTPS')

  return {
    ...(http ? { http_proxy: http, HTTP_PROXY: http } : {}),
    ...(https ? { https_proxy: https, HTTPS_PROXY: https } : {}),
    ...(http || https ? { all_proxy: http || https, ALL_PROXY: http || https } : {})
  }
}

/** Read the active macOS system proxy without making it a hard dependency. */
export function readMacSystemProxyEnv(platform = process.platform): ProxyEnv {
  if (platform !== 'darwin') {
    return {}
  }

  try {
    const output = execFileSync('/usr/sbin/scutil', ['--proxy'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 1500
    })

    return parseMacSystemProxy(output)
  } catch {
    return {}
  }
}

/** Add system proxy settings only when the desktop app did not inherit them. */
export function buildUpdateCheckEnv(baseEnv: NodeJS.ProcessEnv, platform = process.platform): ProxyEnv {
  const systemProxy = readMacSystemProxyEnv(platform)
  const env: ProxyEnv = { ...systemProxy }

  for (const key of Object.keys(systemProxy)) {
    if (baseEnv[key]) {
      env[key] = baseEnv[key] as string
    }
  }

  return { ...baseEnv, ...env, GIT_TERMINAL_PROMPT: '0' }
}

export function withoutProxyEnv(baseEnv: NodeJS.ProcessEnv): ProxyEnv {
  const env: ProxyEnv = { ...baseEnv }

  for (const key of ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']) {
    // runGit merges options.env over process.env; empty values therefore
    // explicitly disable inherited proxy variables for the direct retry.
    env[key] = ''
  }

  return env
}

export function hasProxyEnv(env: NodeJS.ProcessEnv): boolean {
  return Boolean(env.http_proxy || env.https_proxy || env.all_proxy || env.HTTP_PROXY || env.HTTPS_PROXY || env.ALL_PROXY)
}

export function isGitNetworkError(output: string): boolean {
  return /(could not resolve host|failed to connect|connection timed out|connection reset|network is unreachable|unable to access|proxyconnect|tls|ssl|operation timed out)/i.test(output)
}
