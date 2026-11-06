import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { PLUGIN_ID, PROMPT_REFINEMENT_LOCALES } from './i18n'
import { createRefineControl, SETTINGS_PATH } from './refine-control'
import { createSettingsStore, SETTINGS_KEY } from './settings'

beforeAll(() => {
  registerPluginLocales(PLUGIN_ID, PROMPT_REFINEMENT_LOCALES)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function mount(initial: Record<string, unknown> = { mode: 'manual' }) {
  const data = new Map<string, unknown>([[SETTINGS_KEY, initial]])

  const store = createSettingsStore({
    get: (key, fallback) => (data.has(key) ? data.get(key) : fallback),
    set: (key, value) => void data.set(key, value)
  })

  const RefineControl = createRefineControl(store)
  render(<RefineControl />)

  return { data, store }
}

const pill = () => screen.getByRole('button', { name: /^Prompt refinement settings/ })

async function openMenu() {
  fireEvent.keyDown(pill(), { key: 'Enter' })

  return screen.findByRole('menu')
}

/** `group` scopes the choice to one section (Depth and Style both have "Auto"). */
async function choose(name: string, group?: string) {
  const scope = group ? within(await screen.findByRole('group', { name: group })) : screen

  fireEvent.click(await scope.findByRole('menuitemradio', { name }))
}

const checked = () =>
  screen
    .getAllByRole('menuitemradio')
    .filter(item => item.getAttribute('aria-checked') === 'true')
    .map(item => item.textContent)

describe('Refine control in the composer', () => {
  it('is one pill naming the current mode, with no separate refine button', () => {
    const { store } = mount({ mode: 'off' })

    expect(pill().textContent).toContain('Refine off')
    act(() => store.update({ mode: 'auto' }))
    expect(pill().textContent).toContain('Refine: Auto')
    act(() => store.update({ mode: 'manual' }))
    expect(pill().textContent).toContain('Refine: Approval-First')

    expect(screen.queryByRole('button', { name: 'Refine prompt' })).toBeNull()
    expect(screen.getAllByRole('button')).toHaveLength(1)
  })

  it('switches mode, depth and style from the menu and saves each choice', async () => {
    const { data, store } = mount({ mode: 'off' })

    await openMenu()
    await choose('Automatic')
    expect(store.get().mode).toBe('auto')

    await openMenu()
    await choose('Polish')
    expect(store.get()).toMatchObject({ autoDepth: false, depth: 'polish' })

    await openMenu()
    await choose('Technical')
    expect(store.get().style).toBe('technical')

    expect(data.get(SETTINGS_KEY)).toMatchObject({
      mode: 'auto',
      autoDepth: false,
      depth: 'polish',
      style: 'technical'
    })

    await openMenu()
    await choose('Auto', 'Depth')
    expect(store.get()).toMatchObject({ autoDepth: true, depth: 'polish', style: 'technical' })
  })

  it('marks the current choices as checked: Auto depth by default', async () => {
    mount({ mode: 'manual', depth: 'clarify', style: 'research' })
    await openMenu()

    expect(checked()).toEqual(['Approval-First', 'Auto', 'Research'])
  })

  it('marks a fixed depth once Auto depth is off', async () => {
    mount({ mode: 'manual', autoDepth: false, depth: 'clarify', style: 'research' })
    await openMenu()

    expect(checked()).toEqual(['Approval-First', 'Clarify', 'Research'])
  })

  it('choosing Custom with no guidance yet opens the settings page to write it', async () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => {})
    const { store } = mount({ mode: 'manual' })

    await openMenu()
    await choose('Custom')

    expect(store.get().style).toBe('custom')
    expect(navigate).toHaveBeenCalledWith(SETTINGS_PATH)
  })

  it('choosing Custom with guidance already written stays in the composer', async () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => {})
    mount({ mode: 'manual', customStyle: 'Write for a senior engineer.' })

    await openMenu()
    await choose('Custom')

    expect(navigate).not.toHaveBeenCalled()
  })
})
