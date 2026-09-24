/**
 * `hermes:sdk-versions` (#120879): Settings -> Providers gains an "SDK and
 * Runtime Versions" panel so users can see (and copy, for bug reports) which
 * agent SDK and runtime versions their install carries. Versions are read at
 * runtime from each package's installed package.json - never hardcoded - and
 * a missing or unreadable install reports null instead of failing the read.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

const electron = vi.hoisted(() => ({
  handlers: new Map<string, (...args: unknown[]) => unknown>()
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, handler: (...args: unknown[]) => unknown) => electron.handlers.set(channel, handler)
  }
}))

import { collectSdkVersions, registerSdkVersionsIpc, SDK_PACKAGES } from './sdk-versions'

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sdk-versions-'))

const writePackage = (modulesRoot: string, pkg: string, version: string) => {
  const dir = path.join(modulesRoot, ...pkg.split('/'))
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: pkg, version }))
}

describe('collectSdkVersions', () => {
  it('reads versions from the managed Hermes node tree (both layouts) and the app node_modules', () => {
    const hermesHome = path.join(scratch, 'home-a')
    const appPath = path.join(scratch, 'app-a')
    // posix managed prefix: <HERMES_HOME>/node/lib/node_modules
    writePackage(path.join(hermesHome, 'node', 'lib', 'node_modules'), '@anthropic-ai/claude-code', '2.0.1')
    // Windows managed prefix: <HERMES_HOME>/node/node_modules
    writePackage(path.join(hermesHome, 'node', 'node_modules'), '@openai/codex', '0.42.0')
    writePackage(path.join(appPath, 'node_modules'), '@google/genai', '1.9.0')

    const info = collectSdkVersions({ appPath, hermesHome, env: {}, platform: 'linux' })

    const byName = new Map(info.sdks.map(sdk => [sdk.name, sdk.version]))
    expect(byName.get('@anthropic-ai/claude-code')).toBe('2.0.1')
    expect(byName.get('@openai/codex')).toBe('0.42.0')
    expect(byName.get('@google/genai')).toBe('1.9.0')
    // Never installed anywhere reachable: null, not a guessed string.
    expect(byName.get('@modelcontextprotocol/sdk')).toBeNull()
  })

  it('prefers the managed Hermes install over the app-bundled copy', () => {
    const hermesHome = path.join(scratch, 'home-b')
    const appPath = path.join(scratch, 'app-b')
    writePackage(path.join(hermesHome, 'node', 'lib', 'node_modules'), '@google/genai', '9.9.9')
    writePackage(path.join(appPath, 'node_modules'), '@google/genai', '1.0.0')

    const info = collectSdkVersions({ appPath, hermesHome, env: {}, platform: 'linux' })

    expect(info.sdks.find(sdk => sdk.name === '@google/genai')?.version).toBe('9.9.9')
  })

  it('reports null for a package whose package.json is unreadable', () => {
    const hermesHome = path.join(scratch, 'home-c')
    const pkgDir = path.join(hermesHome, 'node', 'lib', 'node_modules', '@openai', 'codex')
    fs.mkdirSync(pkgDir, { recursive: true })
    fs.writeFileSync(path.join(pkgDir, 'package.json'), 'not json')

    const info = collectSdkVersions({ appPath: path.join(scratch, 'app-c'), hermesHome, env: {}, platform: 'linux' })

    expect(info.sdks.find(sdk => sdk.name === '@openai/codex')?.version).toBeNull()
  })

  it('lists every SDK the panel promises, in order, plus the Node/Electron runtimes', () => {
    const info = collectSdkVersions({ appPath: scratch, hermesHome: scratch, env: {}, platform: 'linux' })

    expect(info.sdks.map(sdk => sdk.name)).toEqual([...SDK_PACKAGES])
    expect(info.node).toBe(process.versions.node)
    expect(info).toHaveProperty('electron')
  })
})

describe('hermes:sdk-versions', () => {
  it('serves the collected versions over IPC', async () => {
    registerSdkVersionsIpc({ appPath: scratch, hermesHome: scratch, env: {}, platform: 'linux' })
    const handler = electron.handlers.get('hermes:sdk-versions')

    expect(handler).toBeDefined()

    const info = (await handler!({})) as { electron: string; node: string; sdks: { name: string }[] }

    expect(info.node).toBe(process.versions.node)
    expect(info.sdks.length).toBe(SDK_PACKAGES.length)
  })
})
