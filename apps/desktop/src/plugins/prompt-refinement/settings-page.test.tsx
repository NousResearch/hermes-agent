import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { PLUGIN_ID, PROMPT_REFINEMENT_LOCALES } from './i18n'
import { createSettingsStore, MAX_CUSTOM_STYLE_CHARS, SETTINGS_KEY } from './settings'
import { createSettingsPage } from './settings-page'

beforeAll(() => {
  registerPluginLocales(PLUGIN_ID, PROMPT_REFINEMENT_LOCALES)
})

afterEach(cleanup)

function mount(initial: Record<string, unknown> = { mode: 'manual' }) {
  const data = new Map<string, unknown>([[SETTINGS_KEY, initial]])

  const store = createSettingsStore({
    get: (key, fallback) => (data.has(key) ? data.get(key) : fallback),
    set: (key, value) => void data.set(key, value)
  })

  const Page = createSettingsPage(store)
  render(<Page />)

  return { data, store }
}

const slider = () => screen.getByRole('slider', { name: 'Depth' }) as HTMLInputElement

describe('Depth slider', () => {
  it('starts at Deepen and moves through Polish, Clarify, Deepen, saving each', () => {
    const { data, store } = mount({ mode: 'manual', autoDepth: false })

    expect(slider().value).toBe('2')
    expect(slider().getAttribute('aria-valuetext')).toBe('Deepen')

    fireEvent.change(slider(), { target: { value: '0' } })
    expect(store.get().depth).toBe('polish')
    expect(screen.getByText('Polish')).toBeTruthy()
    expect(screen.getByText(/Nothing is added, and it stays about the same length/)).toBeTruthy()
    expect(data.get(SETTINGS_KEY)).toMatchObject({ depth: 'polish' })

    fireEvent.change(slider(), { target: { value: '1' } })
    expect(store.get().depth).toBe('clarify')
    expect(slider().getAttribute('aria-valuetext')).toBe('Clarify')
  })

  it('is disabled while refinement is Off', () => {
    mount({ mode: 'off', autoDepth: false })

    expect(slider().disabled).toBe(true)
  })

  it('"Choose depth automatically" is on by default and holds the slider until turned off', () => {
    const { data } = mount()
    const auto = screen.getByRole('switch', { name: 'Choose depth automatically' })

    expect(auto.getAttribute('aria-checked')).toBe('true')
    expect(slider().disabled).toBe(true)

    fireEvent.click(auto)
    expect(data.get(SETTINGS_KEY)).toMatchObject({ autoDepth: false })
    expect(slider().disabled).toBe(false)
  })

  it('the short-prompt limit applies in Approval-First too', () => {
    mount({ mode: 'manual' })

    expect(screen.getByText('Prompts under 4 words are sent as typed, without refining.')).toBeTruthy()
    expect((screen.getByRole('button', { name: '8' }) as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('Style and chat context', () => {
  it('shows the custom guidance box only for the Custom style, and saves what is typed', () => {
    const { data, store } = mount({ mode: 'manual', style: 'technical' })

    expect(screen.queryByRole('textbox', { name: 'Custom style' })).toBeNull()
    expect(screen.getByText(/Exact versions, files, commands/)).toBeTruthy()

    act(() => store.update({ style: 'custom' }))
    const box = screen.getByRole('textbox', { name: 'Custom style' }) as HTMLTextAreaElement
    expect(box.maxLength).toBe(MAX_CUSTOM_STYLE_CHARS)

    fireEvent.change(box, { target: { value: 'Write for a senior engineer. ' } })
    expect(data.get(SETTINGS_KEY)).toMatchObject({ customStyle: 'Write for a senior engineer. ' })
  })

  it('chat context is an off-by-default switch that saves', () => {
    const { data } = mount()
    const toggle = screen.getByRole('switch', { name: 'Use recent chat context' })

    expect(toggle.getAttribute('aria-checked')).toBe('false')
    fireEvent.click(toggle)
    expect(data.get(SETTINGS_KEY)).toMatchObject({ useChatContext: true })
  })

  it('style and context controls are disabled while refinement is off', () => {
    mount({ mode: 'off' })

    expect((screen.getByRole('switch', { name: 'Use recent chat context' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('combobox', { name: 'Style' }) as HTMLButtonElement).disabled).toBe(true)
  })
})
