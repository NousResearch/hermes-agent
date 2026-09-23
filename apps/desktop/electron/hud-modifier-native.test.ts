import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'

import { test } from 'vitest'

import { HudModifierMonitor, type HudModifierStatus, resolveHudModifierMonitorPath } from './hud-modifier-monitor'

test.skipIf(process.platform !== 'darwin')(
  'executes native gesture adapters and real controller without posting input',
  async () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'hermes-hud-native-'))
    const native = resolve(import.meta.dirname, 'native')

    try {
      const gesture = resolve(dir, 'gesture')
      execFileSync('xcrun', [
        '--sdk',
        'macosx',
        'clang',
        '-std=c11',
        '-Wall',
        '-Wextra',
        '-Werror',
        resolve(native, 'hud-modifier-gesture.test.c'),
        '-o',
        gesture
      ])
      assert.match(execFileSync(gesture, [], { encoding: 'utf8', timeout: 5_000 }), /assertions passed/)
      const adapter = resolve(dir, 'adapter')
      execFileSync('xcrun', [
        '--sdk',
        'macosx',
        'clang',
        '-fobjc-arc',
        '-fblocks',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-framework',
        'Cocoa',
        '-framework',
        'CoreGraphics',
        resolve(native, 'hud-modifier-monitor.test.m'),
        '-o',
        adapter
      ])
      assert.match(execFileSync(adapter, [], { encoding: 'utf8', timeout: 5_000 }), /assertions passed/)
      execFileSync(
        process.execPath,
        [resolve(import.meta.dirname, '../scripts/build-hud-modifier-monitor.mjs'), '--out-dir', resolve(dir, 'dist')],
        { timeout: 60_000 }
      )
      const binary = resolveHudModifierMonitorPath(dir)
      assert.deepEqual(
        execFileSync('xcrun', ['lipo', '-archs', binary], { encoding: 'utf8' }).trim().split(/\s+/).sort(),
        ['arm64', 'x86_64']
      )
      const check = spawnSync(binary, ['--check'], { encoding: 'utf8', timeout: 5_000 })
      assert.equal(check.error, undefined)
      assert.equal(check.stderr, '')
      assert.ok(check.status === 0 || check.status === 2)
      const permission = JSON.parse(check.stdout)
      assert.deepEqual(
        permission,
        check.status === 0 ? { type: 'ready' } : { type: 'error', code: 'permission-required' }
      )
      const invalid = spawnSync(binary, ['--check', '--request-permission'], { encoding: 'utf8', timeout: 5_000 })
      assert.equal(invalid.status, 64)
      assert.deepEqual(JSON.parse(invalid.stdout), { type: 'error', code: 'unavailable' })
      const statuses: HudModifierStatus[] = []
      const monitor = new HudModifierMonitor({ appPath: dir })

      try {
        const status = await new Promise<HudModifierStatus>((settle, reject) => {
          const timer = setTimeout(() => reject(new Error('native helper did not settle')), 10_000)
          monitor.start(
            () => {},
            value => {
              statuses.push(value)

              if (value.type === 'ready' || value.type === 'error') {
                clearTimeout(timer)
                settle(value)
              }
            }
          )
        })

        assert.equal(statuses[0].type, 'starting')
        assert.deepEqual(status, permission)
        console.info(`HUD native check: ${JSON.stringify(status)}; no input posted`)
      } finally {
        monitor.stop()
      }
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  },
  75_000
)
