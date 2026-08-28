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
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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

  return { ...sdk, host: { ...sdk.host, notify, request } }
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

beforeEach(() => {
  vi.clearAllMocks()
  $botMeta.set({})
})

afterEach(() => {
  cleanup()
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
