import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'

import { hiddenWindowsChildOptions } from './windows-child-options'

const execFileAsync = promisify(execFile)

export type MobileCompanionRouteError =
  | 'route-unreachable'
  | 'tailscale-failed'
  | 'tailscale-host-mismatch'
  | 'tailscale-unavailable'
  | 'unsupported-backend'
  | 'unsupported-public-url'

export interface MobileCompanionRouteResult {
  error?: MobileCompanionRouteError
  managed: boolean
  ok: boolean
}

interface RoutePlan {
  localTarget: string
  publicHostname: string
  publicPort: number
}

interface ProbeResponse {
  status: number
}

export interface MobileCompanionRouteDeps {
  env?: NodeJS.ProcessEnv
  platform?: NodeJS.Platform
  probe?: (url: string) => Promise<ProbeResponse>
  run?: (command: string, args: string[]) => Promise<string | void>
}

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase()

  return host === 'localhost' || host === '::1' || host === '[::1]' || host.startsWith('127.')
}

function parseTailscalePublicUrl(rawValue: string): URL | null {
  let url: URL

  try {
    url = new URL(rawValue)
  } catch {
    return null
  }

  const hostname = url.hostname.toLowerCase()

  if (
    url.protocol !== 'https:' ||
    !hostname.endsWith('.ts.net') ||
    hostname === '.ts.net' ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    (url.pathname !== '' && url.pathname !== '/')
  ) {
    return null
  }

  return url
}

export function buildTailscaleRoutePlan(publicUrl: string, backendBaseUrl: string): RoutePlan {
  const external = parseTailscalePublicUrl(publicUrl)

  if (!external) {
    throw new Error('unsupported-public-url')
  }

  let backend: URL

  try {
    backend = new URL(backendBaseUrl)
  } catch {
    throw new Error('unsupported-backend')
  }

  if (
    !['http:', 'https:'].includes(backend.protocol) ||
    !isLoopbackHost(backend.hostname) ||
    !backend.port ||
    backend.username ||
    backend.password ||
    backend.search ||
    backend.hash ||
    (backend.pathname !== '' && backend.pathname !== '/')
  ) {
    throw new Error('unsupported-backend')
  }

  const publicPort = Number(external.port || '443')

  if (!Number.isInteger(publicPort) || publicPort < 1 || publicPort > 65_535) {
    throw new Error('unsupported-public-url')
  }

  return { localTarget: backend.origin, publicHostname: external.hostname.toLowerCase(), publicPort }
}

export function tailscaleBinaryCandidates(
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env
): string[] {
  const candidates: string[] = []

  if (platform === 'win32') {
    for (const root of [env.ProgramFiles, env['ProgramFiles(x86)'], env.LOCALAPPDATA]) {
      if (root) {
        candidates.push(path.win32.join(root, 'Tailscale', 'tailscale.exe'))
      }
    }
  } else if (platform === 'darwin') {
    candidates.push(
      '/Applications/Tailscale.app/Contents/MacOS/Tailscale',
      '/opt/homebrew/bin/tailscale',
      '/usr/local/bin/tailscale'
    )
  }

  candidates.push('tailscale')

  return [...new Set(candidates)]
}

async function defaultRun(command: string, args: string[]): Promise<string> {
  const env = process.platform === 'darwin' ? { ...process.env, TAILSCALE_BE_CLI: '1' } : process.env

  const result = await execFileAsync(
    command,
    args,
    hiddenWindowsChildOptions({ encoding: 'utf8', env, maxBuffer: 1024 * 1024, timeout: 15_000 })
  )

  return result.stdout
}

async function defaultProbe(url: string): Promise<ProbeResponse> {
  const response = await fetch(url, {
    method: 'GET',
    redirect: 'manual',
    signal: AbortSignal.timeout(8_000)
  })

  return { status: response.status }
}

function commandWasMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

export async function probeMobileCompanionRoute(
  publicUrl: string,
  deps: MobileCompanionRouteDeps = {}
): Promise<MobileCompanionRouteResult> {
  if (!parseTailscalePublicUrl(publicUrl)) {
    return { managed: false, ok: true }
  }

  try {
    const response = await (deps.probe ?? defaultProbe)(publicUrl)

    return response.status < 500
      ? { managed: true, ok: true }
      : { error: 'route-unreachable', managed: true, ok: false }
  } catch {
    return { error: 'route-unreachable', managed: true, ok: false }
  }
}

export async function refreshMobileCompanionRoute(
  publicUrl: string,
  backendBaseUrl: string,
  deps: MobileCompanionRouteDeps = {}
): Promise<MobileCompanionRouteResult> {
  let plan: RoutePlan

  try {
    plan = buildTailscaleRoutePlan(publicUrl, backendBaseUrl)
  } catch (error) {
    const code = error instanceof Error ? error.message : ''

    return {
      error: code === 'unsupported-public-url' ? 'unsupported-public-url' : 'unsupported-backend',
      managed: code !== 'unsupported-public-url',
      ok: false
    }
  }

  const run = deps.run ?? defaultRun
  let command: string | null = null

  for (const candidate of tailscaleBinaryCandidates(deps.platform, deps.env)) {
    try {
      const rawStatus = await run(candidate, ['status', '--json'])

      const status = JSON.parse(typeof rawStatus === 'string' ? rawStatus : '') as {
        BackendState?: unknown
        Self?: { DNSName?: unknown; Online?: unknown }
      }

      const dnsName =
        typeof status.Self?.DNSName === 'string' ? status.Self.DNSName.replace(/\.$/, '').toLowerCase() : ''

      if (dnsName !== plan.publicHostname) {
        return { error: 'tailscale-host-mismatch', managed: true, ok: false }
      }

      if (status.BackendState !== 'Running' || status.Self?.Online !== true) {
        return { error: 'tailscale-failed', managed: true, ok: false }
      }

      command = candidate

      break
    } catch (error) {
      if (commandWasMissing(error)) {
        continue
      }

      return { error: 'tailscale-failed', managed: true, ok: false }
    }
  }

  if (!command) {
    return { error: 'tailscale-unavailable', managed: true, ok: false }
  }

  try {
    await run(command, ['serve', '--bg', '--yes', `--https=${plan.publicPort}`, plan.localTarget])
  } catch {
    return { error: 'tailscale-failed', managed: true, ok: false }
  }

  const probe = deps.probe ?? defaultProbe

  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const response = await probe(publicUrl)

      if (response.status < 500) {
        return { managed: true, ok: true }
      }
    } catch {
      // Tailscale Serve can take a moment to publish the replacement target.
    }

    if (attempt < 4) {
      await new Promise(resolve => setTimeout(resolve, 250))
    }
  }

  return { error: 'route-unreachable', managed: true, ok: false }
}
