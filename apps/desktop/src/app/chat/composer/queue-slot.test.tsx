import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import { reconcilePendingSubmissions } from '@/store/pending-submissions'

import { renderComposerQueueSlot } from './queue-slot'

afterEach(() => { cleanup(); $queuedPromptsBySession.set({}); localStorage.clear() })

function renderSlot(
  request = vi.fn(async () => ({})),
  overrides: Partial<Parameters<typeof renderComposerQueueSlot>[0]> = {}
) {
  const inputs = {
    activeQueueSessionKey: 'stored',
    beginQueuedEdit: vi.fn(),
    busy: true,
    drainNextQueued: vi.fn(async () => undefined),
    exitQueuedEdit: vi.fn(),
    gateway: { request } as unknown as HermesGateway,
    queueEdit: null,
    queueParked: false,
    queuedPrompts: getQueuedPrompts('stored'),
    redirectQueuedText: vi.fn(async () => true),
    sendQueuedNow: vi.fn(),
    sessionId: 'runtime-1',
    steerQueuedNow: vi.fn(),
    t: en,
    ...overrides
  }

  render(
    <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
      {renderComposerQueueSlot(inputs)}
    </I18nProvider>
  )

  return inputs
}

async function expand() {
  fireEvent.click(await screen.findByRole('button', { name: /queued/i }))
}

function rowButtons(text: string) {
  let row: HTMLElement | null = screen.getByText(text)

  while (row && !row.querySelector(`[aria-label="${en.composer.queueDelete}"]`)) {
    row = row.parentElement
  }

  return within(row as HTMLElement)
}

const labels = (row: ReturnType<typeof within>) =>
  row
    .queryAllByRole('button')
    .map((button: HTMLElement) => button.getAttribute('aria-label'))
    .filter(Boolean)

it('Delete on a server-queued card cancels its admission at the authority', async () => {
  reconcilePendingSubmissions('stored', [{ admission_id: 'adm-1', status: 'queued', user: 'later please' }])
  const request = vi.fn(async () => ({ admission_id: 'adm-1', status: 'terminal', outcome: 'cancelled' }))
  renderSlot(request)

  fireEvent.click(await screen.findByRole('button', { name: /queued/i }))
  fireEvent.click(screen.getByRole('button', { name: en.composer.queueDelete }))

  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('prompt.cancel', { session_id: 'runtime-1', admission_id: 'adm-1' }))
})

it('a server-queued card shows the Steer, Edit and Next its owner serves (RT-35)', async () => {
  reconcilePendingSubmissions('stored', [
    { admission_id: 'adm-1', status: 'queued', user: 'later please', controls: ['steer', 'replace'] }
  ])
  enqueueQueuedPrompt('stored', { text: 'local words', attachments: [] })
  renderSlot()

  await expand()
  expect(labels(rowButtons('later please')).sort()).toEqual(
    [en.composer.queueDelete, en.composer.queueEdit, en.composer.queueSteer, en.composer.queueSendNext].sort()
  )
  // The local row keeps its own controls, Next included.
  expect(labels(rowButtons('local words'))).toContain(en.composer.queueSendNext)
})

it('a server-queued card without owner verbs keeps Delete and Next (Next withdraws the admission)', async () => {
  reconcilePendingSubmissions('stored', [{ admission_id: 'adm-1', status: 'queued', user: 'later please' }])
  renderSlot()
  await expand()

  expect(labels(rowButtons('later please')).sort()).toEqual([en.composer.queueDelete, en.composer.queueSendNext].sort())
})

it('Next on a server-queued card withdraws the admission then redirects the live turn', async () => {
  reconcilePendingSubmissions('stored', [
    { admission_id: 'adm-1', status: 'queued', user: 'later please', controls: ['steer', 'replace'] }
  ])

  const request = vi.fn(async () => ({}))
  const inputs = renderSlot(request)
  await expand()

  fireEvent.click(rowButtons('later please').getByRole('button', { name: en.composer.queueSendNext }))

  // Send-now is forbidden here: it ends in the hard Stop, and the withdrawn row is already
  // gone by then - that path lost the words (2026-10-20). Withdraw the admission, then
  // deliver the captured text as a redirect so the gateway restarts the live request.
  await vi.waitFor(() => expect(inputs.redirectQueuedText).toHaveBeenCalledWith('later please'))
  expect(request).toHaveBeenCalledWith('prompt.cancel', expect.objectContaining({ admission_id: 'adm-1' }))
  expect(inputs.sendQueuedNow).not.toHaveBeenCalled()
})

it('Steer on a server-queued card asks the authority to steer that admission', async () => {
  reconcilePendingSubmissions('stored', [
    { admission_id: 'adm-1', status: 'queued', user: 'later please', controls: ['steer', 'replace'] }
  ])
  const request = vi.fn(async () => ({ admission_id: 'adm-1', status: 'terminal', outcome: 'steered' }))
  const inputs = renderSlot(request)
  await expand()

  fireEvent.click(rowButtons('later please').getByRole('button', { name: en.composer.queueSteer }))

  await vi.waitFor(() =>
    expect(request).toHaveBeenCalledWith('prompt.steer_queued', { session_id: 'runtime-1', admission_id: 'adm-1' })
  )
  expect(inputs.steerQueuedNow).not.toHaveBeenCalled()
})

it('Edit on a server-queued card opens it in the composer', async () => {
  reconcilePendingSubmissions('stored', [
    { admission_id: 'adm-1', status: 'queued', user: 'later please', controls: ['steer', 'replace'] }
  ])
  const inputs = renderSlot()
  await expand()

  fireEvent.click(rowButtons('later please').getByRole('button', { name: en.composer.queueEdit }))

  expect(inputs.beginQueuedEdit).toHaveBeenCalledWith(expect.objectContaining({ id: 'adm-1', serverStatus: 'queued' }))
})
