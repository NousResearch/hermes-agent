/** Disable proxy variables explicitly: child helpers may merge env over process.env. */
export function withoutProxyEnv(baseEnv: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env = { ...baseEnv }

  for (const key of new Set([...Object.keys(env), 'http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'])) {
    if (/^(http|https|all)_proxy$/i.test(key)) {
      env[key] = ''
    }
  }

  return env
}

export function hasProxyEnv(env: NodeJS.ProcessEnv): boolean {
  return Object.entries(env).some(([key, value]) => /^(http|https|all)_proxy$/i.test(key) && Boolean(value))
}

export function isGitNetworkError(output: string): boolean {
  return /(could not resolve host|failed to connect|connection timed out|connection reset|network is unreachable|unable to access|proxyconnect|tls|ssl|operation timed out)/i.test(output)
}
