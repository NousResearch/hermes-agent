import { contrastRatio } from '@hermes/shared/color'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { useTheme } from './context'
import { skinToDesktopTheme } from './skin'

// What the gateway and the Electron bridge report for a STOCK config
// (`display.skin: default`): the CLI's classic gold skin, by name `default`.
const stockDefaultSkin = {
  name: 'default',
  description: 'Classic Hermes — gold and kawaii',
  colors: { banner_text: '#FFF8DC', status_bar_bg: '#1a1a2e', ui_accent: '#FFBF00', banner_border: '#CD7F32' }
}

const BACKEND_THEMES_KEY = 'hermes-desktop-backend-themes-v1'

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)
const paintedSkin = () => window.document.documentElement.dataset.hermesTheme

type ThemeApi = ReturnType<typeof useTheme>

/** A fresh renderer launch: module state reloads, localStorage survives. */
async function launch(localSkin: typeof stockDefaultSkin | null) {
  cleanup()
  // Appearance picks persist through the real policy; only transport is mocked.
  const writeConfig = vi.fn().mockResolvedValue({ ok: true })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api: writeConfig, ...(localSkin ? { localSkin: { profile: 'default', skin: localSkin } } : {}) }
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

  return { api, bootPaint, connect, writeConfig }
}

describe('Classic Hermes is an explicit Desktop pick, never inferred from stock config (#76579)', () => {
  beforeEach(() => window.localStorage.clear())

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.resetModules()
  })

  it.each([
    ['display.skin: default', stockDefaultSkin],
    ['display.skin unset', null]
  ])('a stock user (%s) with no Desktop pick paints Nous on boot, connect and relaunch', async (_label, local) => {
    let run = await launch(local)
    expect(run.bootPaint).toBe('nous')
    run.connect()
    run.connect() // reconnect re-seed
    expect(run.api.theme?.themeName).toBe('nous')
    expect(paintedSkin()).toBe('nous')

    run = await launch(local)
    expect(run.bootPaint).toBe('nous')
    run.connect()
    expect(paintedSkin()).toBe('nous')
  })

  it('a Classic pick paints gold/navy (dark mode) and survives connect, reconnect and relaunch; a later Nous pick sticks', async () => {
    let run = await launch(stockDefaultSkin)
    expect(run.api.theme?.availableThemes.find(t => t.name === 'classic')?.label).toBe('Classic Hermes')

    await act(async () => {
      run.api.theme?.setMode('dark')
      run.api.theme?.setTheme('classic')
    })
    expect(run.writeConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/config',
        method: 'PUT',
        profile: 'default',
        body: { config: { desktop: { theme: 'classic' } } }
      })
    )
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

    act(() => run.api.theme?.setTheme('nous'))
    run = await launch(stockDefaultSkin)
    run.connect()
    expect(run.bootPaint).toBe('nous')
    expect(paintedSkin()).toBe('nous')
  })

  it('/skin gold and /skin hermes select Classic; /skin default stays the Desktop default (Nous)', async () => {
    const run = await launch(stockDefaultSkin)

    act(() => void run.api.skin?.('gold'))
    expect(run.api.theme?.themeName).toBe('classic')

    act(() => void run.api.skin?.('nous'))
    act(() => void run.api.skin?.('hermes'))
    expect(run.api.theme?.themeName).toBe('classic')

    act(() => void run.api.skin?.('default'))
    expect(run.api.theme?.themeName).toBe('nous')
  })

  it('a cache left by the reverted #130015 build (CLI `default` as "Classic Hermes") shows ONE Classic and is purged', async () => {
    // Exactly what that build wrote: the converted CLI `default` skin, relabelled.
    window.localStorage.setItem(
      BACKEND_THEMES_KEY,
      JSON.stringify({
        default: {
          ...skinToDesktopTheme(stockDefaultSkin),
          description: stockDefaultSkin.description,
          label: 'Classic Hermes'
        }
      })
    )

    const run = await launch(stockDefaultSkin)
    run.connect()

    const themes = run.api.theme?.availableThemes ?? []
    expect(themes.filter(t => t.label === 'Classic Hermes').map(t => t.name)).toEqual(['classic'])
    expect(themes.some(t => t.name === 'default')).toBe(false)
    expect(run.api.skin?.('list').match(/Classic Hermes/g)).toHaveLength(1)
    expect(JSON.parse(window.localStorage.getItem(BACKEND_THEMES_KEY) ?? '{}')).not.toHaveProperty('default')
  })

  it('Classic follows the light/dark toggle: a light surface in light mode, gold on navy in dark, both readable', async () => {
    const run = await launch(stockDefaultSkin)
    act(() => run.api.theme?.setTheme('classic'))

    act(() => run.api.theme?.setMode('light'))
    expect(window.document.documentElement.dataset.hermesMode).toBe('light')
    expect(cssVar('--theme-background-seed').toLowerCase()).toBe('#f5f5f5')
    expect(contrastRatio(cssVar('--theme-foreground'), cssVar('--theme-background-seed'))).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(cssVar('--theme-primary'), cssVar('--theme-sidebar-seed'))).toBeGreaterThanOrEqual(4.5)

    act(() => run.api.theme?.setMode('dark'))
    expect(window.document.documentElement.dataset.hermesMode).toBe('dark')
    expect(cssVar('--theme-background-seed')).toBe('#1a1a2e')
    expect(contrastRatio(cssVar('--theme-foreground'), cssVar('--theme-background-seed'))).toBeGreaterThanOrEqual(4.5)
  })
})
