import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

export const DEFAULT_GATEWAY_TUNNEL_URL = 'http://127.0.0.1:9119'

export interface GatewayTunnelRunResult {
  status: number | null
  stderr?: string | Buffer
  stdout?: string | Buffer
}

export interface GatewayTunnelManagerOptions {
  markerPath: string
  run?: (scriptPath: string, args: string[]) => GatewayTunnelRunResult
  scriptPath: string
  url?: string
}

export interface GatewayTunnelCleanupOptions {
  manager: {
    deactivate: () => unknown
    isOwned: () => boolean
  }
  onError: (error: unknown) => void
  writeLocalConfig: () => void
}

function outputText(value: string | Buffer | undefined): string {
  return value ? String(value).trim() : ''
}

export function createGatewayTunnelManager(options: GatewayTunnelManagerOptions) {
  const { markerPath, scriptPath } = options
  const url = options.url ?? DEFAULT_GATEWAY_TUNNEL_URL

  const run =
    options.run ??
    ((command: string, args: string[]) =>
      spawnSync('/bin/bash', [command, ...args], {
        encoding: 'utf8',
        timeout: 45_000
      }))

  const invoke = (action: 'start' | 'stop') => {
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Gateway tunnel script not found: ${scriptPath}`)
    }

    const result = run(scriptPath, [action])
    const stdout = outputText(result.stdout)
    const stderr = outputText(result.stderr)

    if (result.status !== 0) {
      throw new Error(stderr || stdout || `Gateway tunnel script failed (${action}).`)
    }

    return stdout
  }

  return {
    activate() {
      const output = invoke('start')
      fs.mkdirSync(path.dirname(markerPath), { recursive: true })
      fs.writeFileSync(markerPath, JSON.stringify({ scriptPath, startedAt: new Date().toISOString(), url }, null, 2))

      return { active: true, output, url }
    },

    deactivate() {
      if (!fs.existsSync(markerPath)) {
        return { active: false, output: '', url }
      }

      const output = invoke('stop')
      fs.rmSync(markerPath, { force: true })

      return { active: false, output, url }
    },

    isOwned() {
      return fs.existsSync(markerPath)
    }
  }
}

export function createGatewayTunnelCleanup(options: GatewayTunnelCleanupOptions): () => void {
  return () => {
    if (!options.manager.isOwned()) {
      return
    }

    try {
      options.writeLocalConfig()
      options.manager.deactivate()
    } catch (error) {
      options.onError(error)
    }
  }
}
