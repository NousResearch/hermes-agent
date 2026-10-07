import { contrastRatio } from '@rabbit/shared/color'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { useTheme } from './context'
import { skinToDesktopTheme } from './skin'

// What the gateway and the Electron bridge report for a STOCK config
// (`display.skin: default`): the CLI's classic gold skin, by name `default`.
const stockDefaultSkin = {
  name: 'default',
  description: 'Classic Rabbit — gold and kawaii',
  colors: { banner_text: '#FFF8DC', status_bar_bg: '#1a1a2e', ui_accent: '#FFBF00', banner_border: '#CD7F32' }
}

const BACKEND_THEMES_KEY = 'rabbit-desktop-backend-themes-v1'

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)
const paintedSkin = () => window.document.documentElement.dataset.rabbitTheme

type ThemeApi = ReturnType<typeof useTheme>

/** A fresh renderer launch: module state reloads, localStorage survives. */
async function launch(localSkin: typeof stockDefaultSkin | null) {
  cleanup()
  Object.defineProperty(window, 'rabbitDesktop', {
    configurable: true,
    value: localSkin ? { localSkin: { profile: 'default', skin: localSkin } } : {}
  })
  vi.resetModules()

  const [context, sync, skinCommand] = await Promise.all([
    import('./context'),
    import('./backend-sync'),
    import('./use-skin-command')
  ])

  // The pre-mount boot paint, i.e. the very first frame of the launch.
  const bootPaint = paintedSkin()
  const api: { theme: ThemeApi | null; skin: ((arg: string) => string) | null } = { skin: null, theme: null }

  function Probe() {
    api.theme = context.useTheme()
    api.skin = skinCommand.useSkinCommand()

    return null
  }

  render(
    <context.ThemeProvider>
      <Probe />
    </context.ThemeProvider>
  )

  // gateway.ready: the connect-time seed of the backend's active skin.
  const connect = () => act(() => sync.ingestBackendSkin(stockDefaultSkin, { apply: false }))

  return { api, bootPaint, connect }
}

describe('Classic Rabbit is an explicit Desktop pick, never inferred from stock config (#76579)', () => {
  beforeEach(() => window.localStorage.clear())

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
    Reflect.deleteProperty(window, 'rabbitDesktop')
    vi.resetModules()
  })

  it.each([
    ['display.skin: default', stockDefaultSkin],
    ['display.skin unset', null]
  ])('a stock user (%s) with no Desktop pick paints Rabbit on boot, connect and relaunch', async (_label, local) => {
    let run = await launch(local)
    expect(run.bootPaint).toBe('rabbit')
    run.connect()
    run.connect() // reconnect re-seed
    expect(run.api.theme?.themeName).toBe('rabbit')
    expect(paintedSkin()).toBe('rabbit')

    run = await launch(local)
    expect(run.bootPaint).toBe('rabbit')
    run.connect()
    expect(paintedSkin()).toBe('rabbit')
  })

  it('a Classic pick paints gold/navy (dark mode) and survives connect, reconnect and relaunch; a later Rabbit pick sticks', async () => {
    let run = await launch(stockDefaultSkin)
    expect(run.api.theme?.availableThemes.find(t => t.name === 'classic')?.label).toBe('Classic Rabbit')

    act(() => run.api.theme?.setMode('dark'))
    act(() => run.api.theme?.setTheme('classic'))
    expect(paintedSkin()).toBe('classic')
    expect(cssVar('--theme-background-seed')).toBe('#1a1a2e')
    expect(cssVar('--theme-primary').toLowerCase()).toBe('#ffbf00')

    run.connect()
    run.connect()
    expect(run.api.theme?.themeName).toBe('classic')

    run = await launch(stockDefaultSkin)
    expect(run.bootPaint).toBe('classic')
    run.connect()
    expect(paintedSkin()).toBe('classic')
    expect(cssVar('--theme-background-seed')).toBe('#1a1a2e')

    act(() => run.api.theme?.setTheme('rabbit'))
    run = await launch(stockDefaultSkin)
    run.connect()
    expect(run.bootPaint).toBe('rabbit')
    expect(paintedSkin()).toBe('rabbit')
  })

  it('/skin gold selects Classic; /skin acme still reaches Rabbit via the legacy alias; /skin default stays the Desktop default (Rabbit)', async () => {
    const run = await launch(stockDefaultSkin)

    act(() => void run.api.skin?.('gold'))
    expect(run.api.theme?.themeName).toBe('classic')

    act(() => void run.api.skin?.('acme'))
    expect(run.api.theme?.themeName).toBe('rabbit')

    act(() => void run.api.skin?.('rabbit'))
    expect(run.api.theme?.themeName).toBe('rabbit')

    act(() => void run.api.skin?.('default'))
    expect(run.api.theme?.themeName).toBe('rabbit')
  })

  it('a cache left by the reverted #130015 build (CLI `default` as "Classic Rabbit") shows ONE Classic and is purged', async () => {
    // Exactly what that build wrote: the converted CLI `default` skin, relabelled.
    window.localStorage.setItem(
      BACKEND_THEMES_KEY,
      JSON.stringify({
        default: {
          ...skinToDesktopTheme(stockDefaultSkin),
          description: stockDefaultSkin.description,
          label: 'Classic Rabbit'
        }
      })
    )

    const run = await launch(stockDefaultSkin)
    run.connect()

    const themes = run.api.theme?.availableThemes ?? []
    expect(themes.filter(t => t.label === 'Classic Rabbit').map(t => t.name)).toEqual(['classic'])
    expect(themes.some(t => t.name === 'default')).toBe(false)
    expect(run.api.skin?.('list').match(/Classic Rabbit/g)).toHaveLength(1)
    expect(JSON.parse(window.localStorage.getItem(BACKEND_THEMES_KEY) ?? '{}')).not.toHaveProperty('default')
  })

  it('Classic follows the light/dark toggle: a light surface in light mode, gold on navy in dark, both readable', async () => {
    const run = await launch(stockDefaultSkin)
    act(() => run.api.theme?.setTheme('classic'))

    act(() => run.api.theme?.setMode('light'))
    expect(window.document.documentElement.dataset.rabbitMode).toBe('light')
    expect(cssVar('--theme-background-seed').toLowerCase()).toBe('#f5f5f5')
    expect(contrastRatio(cssVar('--theme-foreground'), cssVar('--theme-background-seed'))).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(cssVar('--theme-primary'), cssVar('--theme-sidebar-seed'))).toBeGreaterThanOrEqual(4.5)

    act(() => run.api.theme?.setMode('dark'))
    expect(window.document.documentElement.dataset.rabbitMode).toBe('dark')
    expect(cssVar('--theme-background-seed')).toBe('#1a1a2e')
    expect(contrastRatio(cssVar('--theme-foreground'), cssVar('--theme-background-seed'))).toBeGreaterThanOrEqual(4.5)
  })
})
