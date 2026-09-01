import assert from 'node:assert/strict'

import { test } from 'vitest'

import { readFallbackClipboardPng } from './clipboard-image-fallback'

const PNG_A = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1])
const PNG_B = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 2])

function ladder(overrides: Record<string, unknown> = {}) {
  const calls: string[] = []

  // Overrides may supply a reader result as a function; wrap it so the rung
  // invocation is always recorded without changing what it returns.
  const wrap =
    (name: string) =>
    (): Buffer | null => {
      calls.push(name === 'readWsl' ? 'wsl' : 'wayland')
      const fn = overrides[name] as (() => Buffer | null) | undefined

      return typeof fn === 'function' ? fn() : null
    }

  const rungs = {
    isWsl: false,
    platform: 'linux' as NodeJS.Platform,
    ...overrides,
    readWsl: wrap('readWsl'),
    readWayland: wrap('readWayland')
  }

  return { rungs, calls }
}

test('WSL rung runs first and its PNG wins — Wayland reader never spawns', () => {
  const { rungs, calls } = ladder({ isWsl: true, readWsl: (() => PNG_A) as any })
  const result = readFallbackClipboardPng(rungs)
  assert.ok(result && result.equals(PNG_A))
  assert.deepEqual(calls, ['wsl'])
})

test('WSL rung empty on a Linux host falls through to the Wayland rung', () => {
  const { rungs, calls } = ladder({ isWsl: true, readWayland: (() => PNG_B) as any })
  const result = readFallbackClipboardPng(rungs)
  assert.ok(result && result.equals(PNG_B))
  assert.deepEqual(calls, ['wsl', 'wayland'])
})

test('plain Linux (not WSL) goes straight to the Wayland rung', () => {
  const { rungs, calls } = ladder({ readWayland: (() => PNG_B) as any })
  const result = readFallbackClipboardPng(rungs)
  assert.ok(result && result.equals(PNG_B))
  assert.deepEqual(calls, ['wayland'])
})

test('non-Linux platforms (macOS/Windows native) never reach the Wayland rung', () => {
  for (const platform of ['darwin', 'win32']) {
    const { rungs, calls } = ladder({ platform })
    const result = readFallbackClipboardPng(rungs)
    assert.equal(result, null)
    assert.deepEqual(calls, [])
  }
})

test('all rungs empty returns null (handler must answer "no image")', () => {
  const { rungs } = ladder()
  assert.equal(readFallbackClipboardPng(rungs), null)
})

test('Wayland rung returning a PNG ends the ladder — no extra rung after it', () => {
  const { rungs, calls } = ladder({ readWayland: (() => PNG_B) as any })
  readFallbackClipboardPng(rungs)
  assert.deepEqual(calls, ['wayland'])
})
