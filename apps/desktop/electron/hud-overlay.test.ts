import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyHudElectronOverlay, setHudAlwaysOnTop } from './hud-overlay'

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

test('unpinning hands the band back to ordinary z-order, on every platform', () => {
  for (const platform of ['darwin', 'linux', 'win32']) {
    const calls: string[] = []

    const win = {
      setAlwaysOnTop(flag: boolean, level?: string) {
        calls.push(`alwaysOnTop:${flag}:${level ?? ''}`)
      },
      setVisibleOnAllWorkspaces(visible: boolean) {
        calls.push(`allWorkspaces:${visible}`)
      }
    }

    setHudAlwaysOnTop(win, false, platform)

    // No level: an unpinned bar must not re-float, and on macOS it must stop
    // striding across virtual desktops too.
    assert.deepEqual(
      calls,
      platform === 'darwin' ? ['alwaysOnTop:false:', 'allWorkspaces:false'] : ['alwaysOnTop:false:']
    )
  }
})

test('re-pinning a bar restores exactly the levels a born-pinned one gets', () => {
  for (const platform of ['darwin', 'linux', 'win32']) {
    const spawned: string[] = []
    const toggled: string[] = []

    const recorder = (calls: string[]) => ({
      setAlwaysOnTop(flag: boolean, level?: string) {
        calls.push(`alwaysOnTop:${flag}:${level ?? ''}`)
      },
      setVisibleOnAllWorkspaces(visible: boolean, options?: { visibleOnFullScreen?: boolean }) {
        calls.push(`allWorkspaces:${visible}:${options?.visibleOnFullScreen === true}`)
      }
    })

    applyHudElectronOverlay(recorder(spawned), platform)
    setHudAlwaysOnTop(recorder(toggled), false, platform)
    setHudAlwaysOnTop(recorder(toggled), true, platform)

    // Only as many trailing calls as a spawn would have made: macOS also
    // claims all-spaces, the others float at one level.
    assert.deepEqual(toggled.slice(-spawned.length), spawned)
  }
})
