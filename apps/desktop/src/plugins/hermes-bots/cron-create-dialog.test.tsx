/**
 * The Create-job dialog: who it says the job belongs to, and where the run's
 * output is delivered.
 *
 * #93572: the dialog's `bot` prop is the pane's create target — an owner
 * OBJECT for roster-scoped bots, a bare profile name otherwise. Building the
 * label with `displayName({ name: bot }, $botMeta.get()[bot])` rendered
 * "[object Object]" and keyed the meta map with an object. The prop is now
 * normalized to a roster row at the component boundary and the meta lookup
 * goes through the object-aware `botRosterMeta`.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { useI18n } from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// The harness supplies the provider and registry normally installed by the host.
// eslint-disable-next-line no-restricted-imports
import { createPluginI18n, I18nProvider } from '@/i18n'
// eslint-disable-next-line no-restricted-imports
import { setRuntimeI18nLocale } from '@/i18n/runtime'

import { BOTS_LOCALES } from './i18n'
import { setPluginCtx } from './shared'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const { notify, request } = vi.hoisted(() => ({
  notify: vi.fn(),
  request: vi.fn(async (_method: string, _params: Record<string, unknown>) => ({}))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, notify, request }
  }
})

const { $botMeta } = await import('./data')
const { CreateRoutineDialog } = await import('./cron')

/** The control under a `labeled(...)` caption — the label is presentational,
 *  so it carries no `for`/`id` pair to query by. */
function controlUnder(caption: string) {
  const field = screen.getByText(caption).parentElement!

  return within(field).getByRole('combobox')
}

/** Name + instruction; the schedule picker already defaults to a valid daily. */
function fillRequiredFields() {
  fireEvent.change(screen.getByPlaceholderText('Morning briefing'), { target: { value: 'Morning digest' } })
  fireEvent.change(screen.getByPlaceholderText(/Summarize my unread Slack threads/), {
    target: { value: 'Summarize yesterday.' }
  })
}

let disposeLocales: () => void

beforeEach(() => {
  vi.clearAllMocks()
  $botMeta.set({})
  const i18n = createPluginI18n('hermes-bots', dispose => dispose)
  disposeLocales = i18n.register(BOTS_LOCALES)
  setPluginCtx({ i18n } as PluginContext)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  setPluginCtx(null)
  setRuntimeI18nLocale('en')
})

function SwitchLanguage() {
  const { locale, setLocale } = useI18n()

  return (
    <button onClick={() => void setLocale(locale === 'en' ? 'ko' : 'en')} type="button">
      Switch language
    </button>
  )
}

it('updates a mounted weekly schedule through the provider while retaining its schedule', async () => {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SwitchLanguage />
      <CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />
    </I18nProvider>
  )

  const selected = (label: string) => screen.getAllByRole('combobox').find(box => box.textContent === label)
  fireEvent.click(selected('Every day')!)
  fireEvent.click(screen.getByRole('option', { name: 'Every week' }))
  fireEvent.click(selected('9:00 AM')!)
  fireEvent.click(screen.getByRole('option', { name: '3:30 PM' }))
  expect(screen.getByText('Runs every Monday at 3:30 PM · 30 15 * * 1')).toBeTruthy()

  // The dialog makes its sibling inert; the test control still calls the real
  // provider as Settings does when changing the locale of a mounted surface.
  fireEvent.click(screen.getByText('Switch language'))
  expect(selected('매주')).toBeTruthy()
  expect(selected('월요일')).toBeTruthy()
  expect(selected('오후 3:30')).toBeTruthy()
  expect(screen.getByText('매주 월요일 오후 3:30에 실행 · 30 15 * * 1')).toBeTruthy()

  fireEvent.click(screen.getByText('Switch language'))
  expect(selected('Every week')).toBeTruthy()
  expect(selected('Monday')).toBeTruthy()
  expect(selected('3:30 PM')).toBeTruthy()
  expect(screen.getByText('Runs every Monday at 3:30 PM · 30 15 * * 1')).toBeTruthy()
  fillRequiredFields()
  fireEvent.click(screen.getByRole('button', { name: 'Create cron' }))

  await waitFor(() => expect(request).toHaveBeenCalled())
  expect(request.mock.calls[0][1]).toMatchObject({ schedule: '30 15 * * 1', profile: 'ops' })
})

it('shows Korean delay units while preserving the schedule sent to the backend', async () => {
  setRuntimeI18nLocale('ko')
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />
    </I18nProvider>
  )

  const selected = (label: string) => screen.getAllByRole('combobox').find(box => box.textContent === label)!
  fireEvent.click(selected('매일'))
  fireEvent.click(screen.getByRole('option', { name: '일정 시간 후 한 번…' }))

  expect(selected('분 후')).toBeTruthy()
  fireEvent.click(selected('분 후'))

  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['분 후', '시간 후', '일 후'])

  fireEvent.click(screen.getByRole('option', { name: '일 후' }))
  expect(screen.getByText('지금부터 30일 후 한 번 실행 · 30d')).toBeTruthy()

  const inputs = screen.getAllByRole('textbox')
  fireEvent.change(inputs[0], { target: { value: 'Digest' } })
  fireEvent.change(
    inputs.find(input => input.tagName === 'TEXTAREA')!,
    {
      target: { value: 'Summarize yesterday.' }
    }
  )
  fireEvent.click(screen.getByRole('button', { name: 'Cron 생성' }))

  await waitFor(() => expect(request).toHaveBeenCalled())
  expect(request.mock.calls[0][1]).toMatchObject({ schedule: '30d', profile: 'ops' })
})

describe('the dialog names the bot, never its object', () => {
  it('resolves an owner OBJECT through the roster-aware meta lookup', () => {
    $botMeta.set({ ops: { title: 'Ops Bot' } })

    render(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)

    const dialog = screen.getByRole('dialog')

    expect(dialog.textContent).toContain('Ops Bot')
    expect(dialog.textContent).not.toContain('[object Object]')
  })

  it('normalizes the bare-name arm to the same label', () => {
    $botMeta.set({ ops: { title: 'Ops Bot' } })

    render(<CreateRoutineDialog bot="ops" onClose={() => undefined} open />)

    expect(screen.getByRole('dialog').textContent).toContain('Ops Bot')
  })
})

describe('where the run\u2019s output lands', () => {
  it('offers run history and the bot\u2019s own chat', () => {
    render(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)

    fireEvent.click(controlUnder('Send results to'))

    const options = screen.getAllByRole('option').map(option => option.textContent)

    expect(options).toContain('Run history only')
    expect(options.some(option => option?.includes('chat (bot responds)'))).toBe(true)
  })

  it('sends no deliver param by default \u2014 history only', async () => {
    render(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)
    fillRequiredFields()

    fireEvent.click(screen.getByRole('button', { name: 'Create cron' }))

    await waitFor(() => expect(request).toHaveBeenCalled())

    const [, params] = request.mock.calls[0]

    expect(params).not.toHaveProperty('deliver')
    expect(params).toMatchObject({ action: 'add', name: '[bot:ops] Morning digest', profile: 'ops' })
  })

  it('sends the BARE bot-chat token on the profile-scoped create', async () => {
    render(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)
    fillRequiredFields()

    fireEvent.click(controlUnder('Send results to'))
    fireEvent.click(screen.getByRole('option', { name: /chat \(bot responds\)/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Create cron' }))

    await waitFor(() => expect(request).toHaveBeenCalled())

    const [, params] = request.mock.calls[0]

    // The job is created in the bot's OWN cron store (profile scoping above),
    // so the bare token resolves to that profile machine-locally. A named
    // token built from a Desktop-side alias could name a profile the backend
    // does not have — the #82530 alias trap.
    expect(params.deliver).toBe('bot-chat')
    expect(params.profile).toBe('ops')
  })

  it('returns the picker to history when the dialog is reopened', async () => {
    const { rerender } = render(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)

    fireEvent.click(controlUnder('Send results to'))
    fireEvent.click(screen.getByRole('option', { name: /chat \(bot responds\)/ }))
    // Cancel resets; a reopened dialog must never inherit the last target.
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    rerender(<CreateRoutineDialog bot={{ name: 'ops' }} onClose={() => undefined} open />)

    await waitFor(() => expect(controlUnder('Send results to').textContent).toBe('Run history only'))
  })
})
