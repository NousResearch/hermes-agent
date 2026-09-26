const PROXY_ENV_KEYS = [
  'HTTPS_PROXY',
  'https_proxy',
  'HTTP_PROXY',
  'http_proxy',
  'ALL_PROXY',
  'all_proxy'
] as const

const ELECTRON_DOWNLOAD_OVERRIDE_KEYS = ['ELECTRON_MIRROR', 'ELECTRON_NIGHTLY_MIRROR'] as const

function proxyUrlFromElectronResult(resolvedProxy: string): string | null {
  const first = resolvedProxy
    .split(';')
    .map(value => value.trim())
    .find(Boolean)

  if (!first || /^DIRECT$/i.test(first)) {
    return null
  }

  const match = first.match(/^(PROXY|HTTP|HTTPS)\s+([^\s]+)$/i)

  if (!match) {
    return null
  }

  const scheme = match[1]!.toUpperCase() === 'HTTPS' ? 'https' : 'http'

  try {
    const url = new URL(`${scheme}://${match[2]}`)

    return url.hostname ? url.toString().replace(/\/$/, '') : null
  } catch {
    return null
  }
}

/**
 * Bridge Electron/Chromium's effective OS/PAC proxy into the detached source
 * updater. Explicit operator proxy environment always wins; DIRECT and proxy
 * schemes that @electron/get cannot consume remain untouched.
 */
export function sourceUpdateProxyEnvironment(
  resolvedProxy: string,
  currentEnv: NodeJS.ProcessEnv = process.env
): NodeJS.ProcessEnv {
  if (
    PROXY_ENV_KEYS.some(key => Boolean(currentEnv[key])) ||
    ELECTRON_DOWNLOAD_OVERRIDE_KEYS.some(key => Boolean(currentEnv[key]))
  ) {
    return {}
  }

  const proxyUrl = proxyUrlFromElectronResult(resolvedProxy)

  // resolveProxy() was asked about an HTTPS Electron release URL. Scope
  // the bridge to HTTPS so PAC setups with separate HTTP routing keep their
  // original policy.
  return proxyUrl ? { HTTPS_PROXY: proxyUrl } : {}
}
