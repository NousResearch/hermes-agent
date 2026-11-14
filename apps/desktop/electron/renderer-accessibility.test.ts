import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  bridgeRendererAccessibilityConfig,
  enableRendererAccessibility,
  shouldEnableRendererAccessibility
} from './renderer-accessibility'
import { readDesktopLaunchConfig } from './renderer-heap-flags'

test('shouldEnableRendererAccessibility leaves automatic detection on the typed platforms', () => {
  assert.equal(shouldEnableRendererAccessibility({}, 'darwin'), false)
  assert.equal(shouldEnableRendererAccessibility({}, 'win32'), false)
})

test('shouldEnableRendererAccessibility never enables on platforms without the typed API', () => {
  // Linux AT reaches the renderer through the platform's ATK stack, not
  // through app.setAccessibilitySupportEnabled.
  assert.equal(shouldEnableRendererAccessibility({}, 'linux'), false)
})

test('shouldEnableRendererAccessibility honors the opt-out env bridge', () => {
  for (const off of ['0', 'false', 'no', 'off', 'OFF', ' 0 ', 'disabled', ' Disabled ']) {
    assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: off }, 'darwin'), false)
    assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: off }, 'win32'), false)
  }

  // Only an explicit positive value forces the tree.
  assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: undefined }, 'darwin'), false)
  assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '' }, 'win32'), false)
  assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '1' }, 'darwin'), true)
})

test('enableRendererAccessibility flips the injected Electron switch exactly when enabled', () => {
  const recorded: boolean[] = []
  const appApi = { setAccessibilitySupportEnabled: (enabled: boolean) => recorded.push(enabled) }

  enableRendererAccessibility({ appApi, env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '1' }, platform: 'darwin' })
  enableRendererAccessibility({ appApi, env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '1' }, platform: 'win32' })
  enableRendererAccessibility({ appApi, env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '0' }, platform: 'darwin' })
  enableRendererAccessibility({ appApi, env: {}, platform: 'linux' })

  enableRendererAccessibility({ appApi, env: {}, platform: 'darwin' })
  enableRendererAccessibility({ appApi, env: {}, platform: 'win32' })

  assert.deepEqual(recorded, [true, true])
})

test('packaged config preserves dictation opt-ins and explicit environment overrides', () => {
  for (const configured of [undefined, false, true]) {
    const yaml = configured === undefined ? '' : `desktop:\n  renderer_accessibility: ${configured}\n`
    const config = readDesktopLaunchConfig(yaml)
    const env: { HERMES_DESKTOP_RENDERER_ACCESSIBILITY?: string } = {}
    bridgeRendererAccessibilityConfig(config.rendererAccessibility, env)

    assert.equal(shouldEnableRendererAccessibility(env, 'darwin'), configured === true)
    assert.equal(shouldEnableRendererAccessibility(env, 'win32'), configured === true)

    for (const override of ['0', '1', '']) {
      const overridden = { HERMES_DESKTOP_RENDERER_ACCESSIBILITY: override }
      bridgeRendererAccessibilityConfig(config.rendererAccessibility, overridden)
      assert.equal(overridden.HERMES_DESKTOP_RENDERER_ACCESSIBILITY, override)
    }
  }
})
