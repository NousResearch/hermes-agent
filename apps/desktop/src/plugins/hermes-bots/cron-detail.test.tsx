/**
 * Bot Mode's cronjob rows were inert: the only interactive controls were the
 * enable switch and the hover-only delete button, so clicking a job to see
 * what it runs, when it runs next, or why it stopped did nothing at all. The
 * gateway already ships every one of those facts with `cron.manage list`, so
 * the inspector reads the record the pane is already holding — no extra RPC,
 * and no second mutation path beside the row's own switch and delete.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { RoutineJob } from './types'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const request = vi.fn(async () => ({}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, request } }
})

const { RoutineDetailDialog, RoutineRow, routineDetailIssue, routineDetailRows } = await import('./cron')

const activeJob: RoutineJob = {
  deliver: 'bot-chat',
  enabled: true,
  job_id: 'job-1',
  last_run_at: '2026-08-23T09:00:00Z',
  last_status: 'success',
  name: '[bot:notetaker] Morning digest',
  next_run_at: '2026-08-23T10:00:00Z',
  prompt_preview: 'Summarize yesterday and post it.',
  repeat: 'forever',
  schedule: 'every 1440m'
}

const valueOf = (rows: Array<{ label: string; value: string }>, label: string) =>
  rows.find(row => row.label === label)?.value

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('the facts the row never showed', () => {
  it('carries only the fields the gateway actually sent', () => {
    const rows = routineDetailRows({ enabled: true, job_id: 'bare', name: 'Bare', schedule: 'every 1h' })
    const labels = rows.map(row => row.label)

    // A job that has never run carries no last_run_at/last_status/model; those
    // rows must be absent rather than rendering an empty or "undefined" value.
    expect(labels).not.toContain('Last run')
    expect(labels).not.toContain('Last result')
    expect(labels).not.toContain('Model')
    expect(rows.every(row => row.value.trim().length > 0)).toBe(true)
  })

  it('promises a next run only while the job is active', () => {
    const active = routineDetailRows(activeJob)

    expect(valueOf(active, 'Status')).toBe('Active')
    expect(valueOf(active, 'Next run')).toBeTruthy()

    const paused = routineDetailRows({ ...activeJob, enabled: false, state: 'paused' })

    expect(valueOf(paused, 'Status')).toBe('Paused')
    expect(valueOf(paused, 'Next run')).toBeUndefined()
    expect(valueOf(paused, 'Last run')).toBeTruthy()
  })

  it('shows the raw schedule only when the humanized label dropped something', () => {
    // "every 1440m" humanizes to "Daily" — the raw form still carries the cadence.
    expect(valueOf(routineDetailRows(activeJob), 'Schedule (raw)')).toBe('every 1440m')

    // A schedule the label passes through unchanged would only be duplicated.
    const passthrough = routineDetailRows({ ...activeJob, schedule: '0 9 * * 1-5' })

    expect(valueOf(passthrough, 'Schedule (raw)')).toBeUndefined()
    expect(valueOf(passthrough, 'Schedule')).toBe('0 9 * * 1-5')
  })

  it('explains a failing or scheduler-paused job in failure order', () => {
    expect(routineDetailIssue(activeJob)).toBeNull()
    expect(routineDetailIssue({ ...activeJob, paused_reason: 'too many failures' })).toBe('too many failures')
    expect(
      routineDetailIssue({ ...activeJob, last_delivery_error: 'telegram 401', paused_reason: 'too many failures' })
    ).toBe('telegram 401')
    expect(
      routineDetailIssue({
        ...activeJob,
        last_delivery_error: 'telegram 401',
        last_fire_error: 'model timeout',
        paused_reason: 'too many failures'
      })
      // The run that never happened outranks the delivery of a run that did.
    ).toBe('model timeout')
  })
})

describe('the row is reachable', () => {
  it('opens THIS job from its own title control', async () => {
    const opened: RoutineJob[] = []

    render(<RoutineRow job={activeJob} onOpen={job => opened.push(job)} owner={{ name: 'notetaker' }} />)

    screen.getByRole('button', { name: /Morning digest/ }).click()

    expect(opened).toEqual([activeJob])
  })

  it('cannot swallow the switch or the delete control', () => {
    render(<RoutineRow job={activeJob} onOpen={() => undefined} owner={{ name: 'notetaker' }} />)

    const opener = screen.getByRole('button', { name: /Morning digest/ })

    // The switch and delete button must be SIBLINGS of the opener: nested
    // inside it, a click on either would also open the inspector (and nested
    // interactive elements are invalid markup).
    expect(within(opener).queryByRole('switch')).toBeNull()
    expect(within(opener).queryByRole('button')).toBeNull()
    expect(screen.getAllByRole('switch')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /delete/i })).toBeTruthy()
  })

  it('refuses to run a legacy delegated routine, and says why', () => {
    const legacy: RoutineJob = {
      ...activeJob,
      prompt_preview: 'You are running the scheduled routine "Morning digest" for agent \'notetaker\'.'
    }

    render(<RoutineRow job={legacy} onOpen={() => undefined} owner={{ name: 'notetaker' }} />)

    // Its prompt was built by interpolation, so the switch stays locked until
    // the user recreates the job through the hardened create path.
    expect(screen.getByRole('switch')).toHaveProperty('disabled', true)
    expect(screen.getByText(/Paused for security/)).toBeTruthy()
  })
})

describe('the inspector', () => {
  it('renders the job\u2019s instruction and its failure', () => {
    render(
      <RoutineDetailDialog job={{ ...activeJob, last_fire_error: 'model timeout' }} onClose={() => undefined} open />
    )

    const dialog = screen.getByRole('dialog')

    expect(within(dialog).getByText('Morning digest')).toBeTruthy()
    expect(within(dialog).getByText('Summarize yesterday and post it.')).toBeTruthy()
    expect(within(dialog).getByText('model timeout')).toBeTruthy()
    // The routing tag is plumbing, not a title.
    expect(dialog.textContent).not.toMatch(/\[bot:notetaker\]/)
  })

  it('stays shut without a job to inspect', () => {
    render(<RoutineDetailDialog job={null} onClose={() => undefined} open />)

    expect(screen.queryByRole('dialog')).toBeNull()

    cleanup()
    render(<RoutineDetailDialog job={activeJob} onClose={() => undefined} open={false} />)

    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
