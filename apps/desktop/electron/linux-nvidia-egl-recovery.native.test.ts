import { execFileSync, spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import electron from 'electron'
import { buildSync } from 'esbuild'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import { launchBuildIdentity } from './launch-build-identity'
import { nvidiaEglMarkerPath, readNvidiaEglMarker, writeNvidiaEglMarker } from './linux-nvidia-egl-fallback'

interface FixtureEvent {
  event: string
  software?: boolean
  marker?: unknown
}

const nativeAvailable = process.platform !== 'linux' || Boolean(process.env.DISPLAY)
const children: Array<ReturnType<typeof spawn>> = []
const stamp = (commit: string) => ({ commit, builtAt: '2026-10-23T00:00:00Z', dirty: false })
let root: string
let firstArtifact: string
let updatedArtifact: string

function buildArtifact(name: string, commit: string): string {
  const artifact = path.join(root, name)
  mkdirSync(artifact)
  writeFileSync(
    path.join(artifact, 'package.json'),
    JSON.stringify({
      name: 'hermes-recovery-fixture',
      version: '0.0.0',
      main: 'main.mjs',
      type: 'module'
    })
  )
  buildSync({
    entryPoints: [path.join(import.meta.dirname, 'fixtures/nvidia-recovery.ts')],
    outfile: path.join(artifact, 'main.mjs'),
    bundle: true,
    platform: 'node',
    format: 'esm',
    external: ['electron'],
    define: { __HERMES_INSTALL_STAMP__: JSON.stringify(stamp(commit)) }
  })

  return artifact
}

function start(artifact: string, userData: string, quitBeforeReveal = false) {
  const env = { ...process.env }
  delete env.ELECTRON_RUN_AS_NODE
  const nonce = `${Date.now()}-${children.length}`
  const eventsFile = path.join(userData, `events-${nonce}.jsonl`)
  const commandFile = path.join(userData, `command-${nonce}`)
  const options = { userData, eventsFile, commandFile, quitBeforeReveal }

  const child = spawn(String(electron), [artifact, `--recovery-fixture=${JSON.stringify(options)}`], {
    env,
    windowsHide: true,
    stdio: ['ignore', 'ignore', 'pipe']
  })

  children.push(child)
  let output = ''
  let failed = false
  child.stderr!.on('data', data => {
    output += String(data)
  })
  child.on('error', error => {
    failed = true
    output += String(error)
  })
  const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))

  return {
    child,
    exited,
    async event(name: string): Promise<FixtureEvent> {
      const deadline = Date.now() + 15_000

      while (Date.now() < deadline) {
        const events: FixtureEvent[] = existsSync(eventsFile)
          ? readFileSync(eventsFile, 'utf8')
              .trim()
              .split('\n')
              .filter(Boolean)
              .map(line => JSON.parse(line))
          : []

        const event = events.find(event => event.event === name)

        if (event) {
          return event
        }

        if (failed || child.exitCode !== null || child.signalCode !== null) {
          throw new Error(`Fixture exited: ${output}`)
        }

        await new Promise(resolve => setTimeout(resolve, 75))
      }

      throw new Error(`Fixture timed out waiting for ${name}: ${output}`)
    },
    async quit(): Promise<void> {
      writeFileSync(commandFile, 'quit')
      await Promise.race([
        exited,
        new Promise<never>((_, reject) => {
          const timer = setTimeout(() => reject(new Error(`Fixture did not quit: ${output}`)), 10_000)
          timer.unref()
        })
      ])
    }
  }
}

describe.skipIf(!nativeAvailable)('native Electron locking and persisted NVIDIA recovery policy (no rendering)', () => {
  beforeAll(() => {
    root = mkdtempSync(path.join(tmpdir(), 'nvidia-native-'))
    firstArtifact = buildArtifact('first', 'a'.repeat(40))
    updatedArtifact = buildArtifact('updated', 'b'.repeat(40))
  })
  afterEach(async () => {
    for (const child of children.splice(0)) {
      if (child.pid && child.exitCode === null && child.signalCode === null) {
        const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))

        if (process.platform === 'win32') {
          execFileSync('taskkill', ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true, stdio: 'ignore' })
        } else {
          child.kill('SIGKILL')
        }

        await exited
      }
    }
  })
  afterAll(() => {
    if (root) {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('a secondary Electron process leaves state intact, then a normal restart stays native', async () => {
    const userData = path.join(root, 'second-launch')
    mkdirSync(userData)
    const primary = start(firstArtifact, userData)
    expect((await primary.event('ready')).software).toBe(false)
    const before = readFileSync(nvidiaEglMarkerPath(userData), 'utf8')
    const secondary = start(firstArtifact, userData)
    await secondary.event('secondary')
    await secondary.exited
    expect(readFileSync(nvidiaEglMarkerPath(userData), 'utf8')).toBe(before)
    await primary.quit()
    const restarted = start(firstArtifact, userData)
    expect((await restarted.event('ready')).software).toBe(false)
    await restarted.quit()
  })

  it.each(['booting', 'fallback'] as const)(
    'a new stamped artifact recovers old %s state at the same app version',
    async state => {
      const userData = path.join(root, `update-${state}`)
      mkdirSync(userData)
      writeNvidiaEglMarker(userData, {
        state,
        version: '0.0.0',
        driverVersion: '600.1.2',
        build: launchBuildIdentity({ appVersion: '0.0.0', installStamp: stamp('a'.repeat(40)) })
      })
      const updated = start(updatedArtifact, userData)
      expect((await updated.event('ready')).software).toBe(false)
      await updated.quit()
      const restarted = start(updatedArtifact, userData)
      expect((await restarted.event('ready')).software).toBe(false)
      await restarted.quit()
      expect(readNvidiaEglMarker(userData)).toEqual({ state: 'ok' })
    }
  )

  it('a clean exit before reveal does not poison the next process', async () => {
    const userData = path.join(root, 'early-quit')
    mkdirSync(userData)
    const early = start(firstArtifact, userData, true)
    await early.event('early-exit')
    await early.exited
    const restarted = start(firstArtifact, userData)
    expect((await restarted.event('ready')).software).toBe(false)
    await restarted.quit()
  })
})
