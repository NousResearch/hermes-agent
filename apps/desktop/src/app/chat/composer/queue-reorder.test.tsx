import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  type QueuedPromptEntry
} from '@/store/composer-queue'
import { reconcilePendingSubmissions } from '@/store/pending-submissions'

import { canMoveQueuedEntry, movedQueueIds, serverMoveAnchor } from './queue-reorder'
import { renderComposerQueueSlot } from './queue-slot'

afterEach(() => {
  cleanup()
  $queuedPromptsBySession.set({})
  localStorage.clear()
})

const texts = () => getQueuedPrompts('stored').map(entry => entry.displayText ?? entry.text)

function renderSlot(request = vi.fn(async (..._args: unknown[]) => ({}))) {
  const queuedPrompts = getQueuedPrompts('stored')

  render(
    <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
      {renderComposerQueueSlot({
        activeQueueSessionKey: 'stored',
        beginQueuedEdit: vi.fn(),
        busy: true,
        drainNextQueued: vi.fn(async () => undefined),
        exitQueuedEdit: vi.fn(),
        gateway: { request } as unknown as HermesGateway,
        queueEdit: null,
        queueParked: false,
        queuedPrompts,
        sendQueuedNow: vi.fn(),
        sessionId: 'runtime-1',
        steerQueuedNow: vi.fn(),
        t: en
      })}
    </I18nProvider>
  )

  return request
}

function row(text: string) {
  let node: HTMLElement | null = screen.getByText(text)

  while (node && !node.querySelector(`[aria-label="${en.composer.queueDelete}"]`)) {
    node = node.parentElement
  }

  return within(node as HTMLElement)
}

async function expand() {
  fireEvent.click(await screen.findByRole('button', { name: /queued/i }))
}

describe('movedQueueIds', () => {
  it('names the row a single move carried, both ways for an adjacent swap', () => {
    expect(movedQueueIds(['a', 'b', 'c', 'd'], ['d', 'a', 'b', 'c'])).toEqual(['d'])
    expect(movedQueueIds(['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'a'])).toEqual(['a'])
    expect(movedQueueIds(['a', 'b', 'c'], ['b', 'a', 'c'])).toEqual(['b', 'a'])
    expect(movedQueueIds(['a', 'b'], ['a', 'b'])).toEqual([])
  })
})

describe('serverMoveAnchor', () => {
  const entry = (id: string, serverStatus?: string): QueuedPromptEntry => ({
    attachments: [],
    id,
    queuedAt: 0,
    serverStatus,
    text: id
  })

  it('anchors on the next still-queued server row, else the back of the FIFO', () => {
    const next = [entry('moved', 'queued'), entry('local'), entry('lost', 'unknown'), entry('s2', 'queued')]
    expect(serverMoveAnchor(next, 'moved')).toBe('s2')
    expect(serverMoveAnchor(next, 's2')).toBeNull()
  })

  it('moves a local row anywhere, a server row only when its owner serves move', () => {
    expect(canMoveQueuedEntry({ serverStatus: undefined })).toBe(true)
    expect(canMoveQueuedEntry({ serverStatus: 'queued', serverControls: ['steer', 'replace'] })).toBe(false)
    expect(canMoveQueuedEntry({ serverStatus: 'queued', serverControls: ['move'] })).toBe(true)
    expect(canMoveQueuedEntry({ serverStatus: 'unknown', serverControls: ['move'] })).toBe(false)
  })
})

describe('queue panel reorder', () => {
  it('▲ / ▼ reorder local rows in place, and the edge buttons are disabled', async () => {
    for (const text of ['first', 'second', 'third']) {
      enqueueQueuedPrompt('stored', { attachments: [], text })
    }

    const request = renderSlot()
    await expand()

    expect(row('first').getByRole('button', { name: en.composer.queueMoveUp })).toHaveProperty('disabled', true)
    expect(row('third').getByRole('button', { name: en.composer.queueMoveDown })).toHaveProperty('disabled', true)

    fireEvent.click(row('third').getByRole('button', { name: en.composer.queueMoveUp }))
    expect(texts()).toEqual(['first', 'third', 'second'])
    expect(request).not.toHaveBeenCalled()
  })

  it('moving a server row asks its owner to move the admission ahead of the next queued one', async () => {
    reconcilePendingSubmissions('stored', [
      { admission_id: 'adm-1', controls: ['steer', 'replace', 'move'], status: 'queued', user: 'one' },
      { admission_id: 'adm-2', controls: ['steer', 'replace', 'move'], status: 'queued', user: 'two' },
      { admission_id: 'adm-3', controls: ['steer', 'replace', 'move'], status: 'queued', user: 'three' }
    ])

    const request = renderSlot(vi.fn(async (..._args: unknown[]) => ({ admission_id: 'adm-3', status: 'queued' })))
    await expand()

    fireEvent.click(row('three').getByRole('button', { name: en.composer.queueMoveUp }))

    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('prompt.move_queued', {
        admission_id: 'adm-3',
        before_admission_id: 'adm-2',
        session_id: 'runtime-1'
      })
    )
    expect(texts()).toEqual(['one', 'three', 'two'])
  })

  it('a refused server move puts the previous order back', async () => {
    reconcilePendingSubmissions('stored', [
      { admission_id: 'adm-1', controls: ['move'], status: 'queued', user: 'one' },
      { admission_id: 'adm-2', controls: ['move'], status: 'queued', user: 'two' }
    ])

    renderSlot(
      vi.fn(async () => {
        throw new Error('admission_started')
      })
    )
    await expand()

    fireEvent.click(row('one').getByRole('button', { name: en.composer.queueMoveDown }))

    await vi.waitFor(() => expect(texts()).toEqual(['one', 'two']))
  })

  it('a server row from an owner without move shows no reorder controls', async () => {
    reconcilePendingSubmissions('stored', [
      { admission_id: 'adm-1', controls: ['steer', 'replace'], status: 'queued', user: 'old owner' }
    ])
    enqueueQueuedPrompt('stored', { attachments: [], text: 'local words' })
    renderSlot()
    await expand()

    expect(row('old owner').queryByRole('button', { name: en.composer.queueMoveUp })).toBeNull()
    expect(row('local words').getByRole('button', { name: en.composer.queueMoveUp })).toBeTruthy()
  })
})
