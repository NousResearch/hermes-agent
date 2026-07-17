import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayTunnelCleanup, createGatewayTunnelManager } from './gateway-tunnel'

const tempDirs: string[] = []

function testScript(dir: string): string {
  const scriptPath = path.join(dir, 'hermes-tunnel.sh')
  fs.writeFileSync(scriptPath, '#!/bin/sh\n')
  fs.chmodSync(scriptPath, 0o700)

  return scriptPath
}

afterEach(() => {
  vi.restoreAllMocks()

  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('managed gateway tunnel', () => {
  it('starts the configured script and records ownership', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gateway-tunnel-'))
    tempDirs.push(dir)
    const scriptPath = testScript(dir)
    const markerPath = path.join(dir, 'managed.json')
    const run = vi.fn(() => ({ status: 0, stdout: 'Tunnel Hermes actif.\n', stderr: '' }))
    const manager = createGatewayTunnelManager({ markerPath, run, scriptPath })

    expect(manager.activate()).toMatchObject({ active: true, url: 'http://127.0.0.1:9119' })
    expect(run).toHaveBeenCalledWith(scriptPath, ['start'])
    expect(JSON.parse(fs.readFileSync(markerPath, 'utf8'))).toMatchObject({ scriptPath })
  })

  it('stops only a tunnel owned by the desktop command', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gateway-tunnel-'))
    tempDirs.push(dir)
    const scriptPath = testScript(dir)
    const markerPath = path.join(dir, 'managed.json')
    fs.writeFileSync(markerPath, JSON.stringify({ scriptPath }))
    const run = vi.fn(() => ({ status: 0, stdout: 'Tunnel arrêté.\n', stderr: '' }))
    const manager = createGatewayTunnelManager({ markerPath, run, scriptPath })

    expect(manager.deactivate()).toMatchObject({ active: false })
    expect(run).toHaveBeenCalledWith(scriptPath, ['stop'])
    expect(fs.existsSync(markerPath)).toBe(false)
  })

  it('surfaces script failures without claiming ownership', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gateway-tunnel-'))
    tempDirs.push(dir)
    const scriptPath = testScript(dir)
    const markerPath = path.join(dir, 'managed.json')
    const run = vi.fn(() => ({ status: 1, stdout: '', stderr: 'Azure indisponible' }))
    const manager = createGatewayTunnelManager({ markerPath, run, scriptPath })

    expect(() => manager.activate()).toThrow('Azure indisponible')
    expect(fs.existsSync(markerPath)).toBe(false)
  })

  it('restores Local mode and stops an owned tunnel when the main window closes', () => {
    const writeLocalConfig = vi.fn()
    const deactivate = vi.fn()

    const cleanup = createGatewayTunnelCleanup({
      manager: { deactivate, isOwned: () => true },
      onError: vi.fn(),
      writeLocalConfig
    })

    cleanup()

    expect(writeLocalConfig).toHaveBeenCalledOnce()
    expect(deactivate).toHaveBeenCalledOnce()
  })
})
