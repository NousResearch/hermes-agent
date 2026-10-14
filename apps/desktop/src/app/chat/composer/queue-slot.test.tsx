import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { reconcilePendingSubmissions } from '@/store/pending-submissions'

import { renderComposerQueueSlot } from './queue-slot'

afterEach(() => { cleanup(); $queuedPromptsBySession.set({}); localStorage.clear() })

it('Delete on a server-queued card cancels its admission at the authority', async () => {
  reconcilePendingSubmissions('stored', [{ admission_id: 'adm-1', status: 'queued', user: 'later please' }])
  const request = vi.fn(async () => ({ admission_id: 'adm-1', status: 'terminal', outcome: 'cancelled' }))

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
        queuedPrompts: getQueuedPrompts('stored'),
        sendQueuedNow: vi.fn(),
        sessionId: 'runtime-1',
        steerQueuedNow: vi.fn(),
        t: en
      })}
    </I18nProvider>
  )

  fireEvent.click(await screen.findByRole('button', { name: /queued/i }))
  fireEvent.click(screen.getByRole('button', { name: en.composer.queueDelete }))

  await vi.waitFor(() => expect(request).toHaveBeenCalledWith('prompt.cancel', { session_id: 'runtime-1', admission_id: 'adm-1' }))
})
