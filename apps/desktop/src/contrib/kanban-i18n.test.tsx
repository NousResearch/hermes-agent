import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { PALETTE_AREA, usePaletteContributions } from '@/app/command-palette/contrib'
import { ROUTES_AREA, SIDEBAR_NAV_AREA, type SidebarNavContribution } from '@/app/routes'
import { createPluginContext } from '@/contrib/plugin'
import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'
import type { HermesConfigRecord } from '@/hermes'
import { I18nProvider, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { contributedKeybindHandler, keybindAction, KEYBINDS_AREA } from '@/lib/keybinds/actions'
import { bindingsFor, resetBinding, setBinding } from '@/store/keybinds'

import { KANBAN_LOCALES } from '../plugins/kanban/i18n'
import plugin from '../plugins/kanban/plugin'

// Keep the real registration and UI path without opening a board socket.
vi.mock('../plugins/kanban/api', async () => ({
  ...(await vi.importActual('../plugins/kanban/api')),
  bindApi: () => () => {}
}))
let i18n: I18nContextValue
const disposers: Array<() => void> = []

function Contributions() {
  i18n = useI18n()
  const nav = useContributions(SIDEBAR_NAV_AREA)
  const palette = usePaletteContributions()

  return (
    <>
      {nav.map(c => (
        <a key={c.id}>{(c.data as SidebarNavContribution).label}</a>
      ))}
      {palette.map(c => (
        <button key={c.id} type="button">
          {c.label}
        </button>
      ))}
    </>
  )
}

function unload() {
  // Match bundled-loader disposal order, including registrations made later.
  disposers.splice(0).forEach(dispose => dispose())
}

afterEach(() => {
  cleanup()
  unload()
  resetBinding('kanban.newTask')
  setRuntimeI18nLocale('en')
})

it('relabels Kanban after delayed config load and locale switches without replacing the board or bindings', async () => {
  setRuntimeI18nLocale('en')
  plugin.register(createPluginContext('kanban', dispose => disposers.push(dispose)))
  const route = registry.getArea(ROUTES_AREA).find(c => c.id === 'kanban:page')
  const handler = contributedKeybindHandler('kanban.newTask')
  setBinding('kanban.newTask', ['mod+alt+k'])
  let resolveConfig!: (config: HermesConfigRecord) => void

  const config = new Promise<HermesConfigRecord>(resolve => {
    resolveConfig = resolve
  })

  const client = { getConfig: () => config, saveConfig: async () => ({ ok: true }) }
  render(
    <I18nProvider configClient={client} initialLocale="en">
      <Contributions />
    </I18nProvider>
  )
  expect(screen.getByText(KANBAN_LOCALES.en!.nav as string)).toBeTruthy()
  await act(async () => {
    resolveConfig({ display: { language: 'zh' } })
    await config
  })

  for (const locale of ['zh', 'ja', 'zh-hant', 'en'] as const) {
    if (locale !== 'zh') {
      await act(() => i18n.setLocale(locale))
    }

    const messages = KANBAN_LOCALES[locale]!
    expect(screen.getByText(messages.nav as string)).toBeTruthy()
    expect(screen.getByRole('button', { name: messages.openBoard as string })).toBeTruthy()
    expect(screen.getByRole('button', { name: messages.newTaskCommand as string })).toBeTruthy()
    expect(keybindAction('kanban.newTask')?.label).toBe(messages.newTaskCommand)
    expect(contributedKeybindHandler('kanban.newTask')).toBe(handler)
    expect(bindingsFor('kanban.newTask')).toEqual(['mod+alt+k'])
    expect(registry.getArea(ROUTES_AREA).find(c => c.id === 'kanban:page')).toBe(route)
    expect(registry.getArea(SIDEBAR_NAV_AREA).filter(c => c.id === 'kanban:nav')).toHaveLength(1)
  }

  await act(() => unload())
  await act(() => i18n.setLocale('zh'))

  for (const area of [SIDEBAR_NAV_AREA, PALETTE_AREA, KEYBINDS_AREA, ROUTES_AREA]) {
    expect(registry.getArea(area).some(c => c.source === 'plugin:kanban')).toBe(false)
  }

  await act(() => plugin.register(createPluginContext('kanban', dispose => disposers.push(dispose))))
  expect(screen.getByText(KANBAN_LOCALES.zh!.nav as string)).toBeTruthy()
  expect(bindingsFor('kanban.newTask')).toEqual(['mod+alt+k'])
})
