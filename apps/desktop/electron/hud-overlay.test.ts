import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyHudElectronOverlay, hudVibrancySupported } from './hud-overlay'

test('macOS uses the floating window level and all-spaces visibility', () => {
  const calls: string[] = []

  const win = {
    setAlwaysOnTop(flag: boolean, level?: string) {
      calls.push(`alwaysOnTop:${flag}:${level}`)
    },
    setVisibleOnAllWorkspaces(visible: boolean, options?: { visibleOnFullScreen?: boolean }) {
      calls.push(`allWorkspaces:${visible}:${options?.visibleOnFullScreen === true}`)
    }
  }

  applyHudElectronOverlay(win, 'darwin')

  assert.deepEqual(calls, ['alwaysOnTop:true:floating', 'allWorkspaces:true:true'])
})

test('Linux and Windows only set the screen-saver always-on-top level', () => {
  for (const platform of ['linux', 'win32']) {
    const calls: string[] = []

    const win = {
      setAlwaysOnTop(flag: boolean, level?: string) {
        calls.push(`alwaysOnTop:${flag}:${level}`)
      },
      setVisibleOnAllWorkspaces() {
        calls.push('allWorkspaces')
      }
    }

    applyHudElectronOverlay(win, platform)

    assert.deepEqual(calls, ['alwaysOnTop:true:screen-saver'])
  }
})

test('macOS 27 keeps the HUD alive by declining transparent-window vibrancy', () => {
  assert.equal(hudVibrancySupported(25), true)
  assert.equal(hudVibrancySupported(26), false)
  assert.equal(hudVibrancySupported(27), true)
  assert.equal(hudVibrancySupported(0), true)
})
