import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyHudElectronOverlay } from './hud-overlay'

test('macOS uses the floating panel level and all-spaces visibility', () => {
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
