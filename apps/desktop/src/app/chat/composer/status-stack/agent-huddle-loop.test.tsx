import { act, cleanup, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { useRealtimeConversation } from '../hooks/use-realtime-conversation'

import { ComposerStatusStack } from './index'

/**
 * The load-bearing seam of the whole feature: hiding the report on screen must
 * NOT disconnect it from the Live agent.
 *
 * The composer lane and the voice loop are rendered together here against the
 * real `$subagentsBySession`: the same terminal transition that drives the
 * huddle's `done` state still has to push the report text into the
 * announcement queue, which is what makes Czesiek speak it and answer from it.
 */

const REPORT = 'Sprawdziłem historię: **pusta** pamięć trwała, brak `hermes-vault/CORE_MEMORY.md`.'

const mocks = vi.hoisted(() => ({ notify: vi.fn((_text: string) => true) }))

vi.mock('@/lib/live-voice/start', () => ({
  startLiveVoice: async () => ({ notify: mocks.notify, setMuted: vi.fn(), stop: vi.fn() })
}))
vi.mock('@/lib/use-enter-animation', () => ({ useEnterAnimation: () => undefined }))

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
  vi.clearAllMocks()
  mocks.notify.mockImplementation(() => true)
})

/** The composer as it exists in the app: the status lane above the composer
 *  plus the Live conversation it belongs to. */
function Harness({ sessionId }: { sessionId: string }) {
  useRealtimeConversation({
    busy: () => false,
    enabled: true,
    failureLabel: 'Blad',
    markSpoken: vi.fn(),
    messages: () => [],
    onFatalError: vi.fn(),
    onSubmit: vi.fn(),
    sessionId
  })

  return <ComposerStatusStack queue={null} sessionId={sessionId} />
}

it('voices the finished report to the Live agent while the lane shows only the huddle', async () => {
  upsertSubagent('owner', { goal: 'Zbadaj system pamięci', status: 'running', subagent_id: 'worker' })

  const view = render(
    <MemoryRouter>
      <Harness sessionId="owner" />
    </MemoryRouter>
  )

  const huddle = () => view.container.querySelector('[data-testid="agent-huddle"]')

  // Live work: the scene is up, and it is wordless.
  expect(huddle()?.getAttribute('data-state')).toBe('waiting')

  act(() => {
    upsertSubagent('owner', { subagent_id: 'worker', text: 'Czytam pliki pamięci' }, false, 'subagent.progress')
  })
  expect(huddle()?.getAttribute('data-state')).toBe('talking')
  expect(view.container.textContent).toContain('Czesiek rozmawia z Hermesem')

  // The worker settles with a report.
  act(() => {
    upsertSubagent(
      'owner',
      { goal: 'Zbadaj system pamięci', status: 'completed', subagent_id: 'worker', summary: REPORT },
      false,
      'subagent.complete'
    )
  })
  expect(huddle()?.getAttribute('data-state')).toBe('done')

  // (b) Nothing of that report is painted in the lane above the composer:
  // neither the markdown source nor its rendered prose.
  const lane = view.container.querySelector('[data-slot="composer-status-stack"]')?.textContent ?? ''

  expect(lane).not.toContain(REPORT)
  expect(lane).not.toContain('pusta')
  expect(lane).not.toContain('CORE_MEMORY.md')
  expect(lane).not.toContain('**')

  // (d) …and it still reaches the Live agent, verbatim, through the queue.
  await waitFor(() => expect(mocks.notify).toHaveBeenCalled())

  const delivered = mocks.notify.mock.calls.map(([text]) => text)

  expect(delivered.some(text => text.includes('Zbadaj system pamięci') && text.includes(REPORT))).toBe(true)
})
