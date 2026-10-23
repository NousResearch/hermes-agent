import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { $automationComposer, openAutomationComposer } from '@/store/automation-composer'
import { $activeSessionId, $sessions } from '@/store/session'

import { AutomationComposerMount } from './automation-composer-mount'

vi.mock('@/store/session-control', async () => {
  const { atom } = await import('nanostores')

  return {
    $sessionControlBySession: atom({}),
    refreshSessionControl: vi.fn(),
    runSessionControlAction: vi.fn()
  }
})

vi.mock('@/app/session/hooks/use-prompt-actions/queue-if-busy', () => ({
  queueKickoffIfSessionBusy: vi.fn(() => 'idle')
}))

vi.mock('@/store/route-tiles', () => ({ openRouteTile: vi.fn() }))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

const session = (id: string, title: string) => ({ id, title, _lineage_ids: [], _lineage_root_id: id })

/** Two composers mounted at once, the way route tiles and bot chats render them. */
async function renderTwoComposers() {
  const onSubmitText = vi.fn().mockResolvedValue(true)

  await act(async () => {
    render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <AutomationComposerMount onSubmitText={onSubmitText} sessionId="session-a" />
        <AutomationComposerMount onSubmitText={onSubmitText} sessionId="session-b" />
      </I18nProvider>
    )
  })

  return onSubmitText
}

afterEach(() => {
  $automationComposer.set({ open: false, sessionId: null, type: 'goal', mode: 'create', submitting: false, error: null })
  $activeSessionId.set(null)
  $sessions.set([])
  vi.clearAllMocks()
})

describe('AutomationComposerMount with two mounted composers', () => {
  it('renders one dialog, naming the session that owns it', async () => {
    $sessions.set([session('session-a', 'Alpha chat'), session('session-b', 'Beta chat')] as never)

    await renderTwoComposers()
    expect(screen.queryAllByRole('dialog')).toHaveLength(0)

    act(() => openAutomationComposer('goal', 'session-a'))

    // One dialog, not one per mounted composer, and it names the owner's conversation.
    expect(screen.getAllByRole('dialog')).toHaveLength(1)
    expect(screen.getByText('Runs in this conversation: Alpha chat')).toBeTruthy()
  })

  it('moves the single dialog to the composer that opens its own session', async () => {
    $sessions.set([session('session-a', 'Alpha chat'), session('session-b', 'Beta chat')] as never)

    await renderTwoComposers()

    act(() => openAutomationComposer('goal', 'session-a'))
    expect(screen.getByText('Runs in this conversation: Alpha chat')).toBeTruthy()

    act(() => openAutomationComposer('loop', 'session-b'))

    expect(screen.getAllByRole('dialog')).toHaveLength(1)
    expect(screen.getByText('Runs in this conversation: Beta chat')).toBeTruthy()
  })

  it('keeps the captured title when a different session is active in the sidebar', async () => {
    $sessions.set([
      session('session-a', 'Alpha chat'),
      session('session-b', 'Beta chat'),
      session('session-c', 'Gamma chat')
    ] as never)
    $activeSessionId.set('session-c')

    await renderTwoComposers()

    act(() => openAutomationComposer('goal', 'session-a'))

    expect(screen.getByText('Runs in this conversation: Alpha chat')).toBeTruthy()
    expect(screen.queryByText('Runs in this conversation: Gamma chat')).toBeNull()
  })

  it('renders nothing while the dialog belongs to a session no mounted composer owns', async () => {
    $sessions.set([session('session-a', 'Alpha chat'), session('session-b', 'Beta chat')] as never)

    await renderTwoComposers()

    act(() => openAutomationComposer('goal', 'session-elsewhere'))

    expect(screen.queryAllByRole('dialog')).toHaveLength(0)
  })
})
