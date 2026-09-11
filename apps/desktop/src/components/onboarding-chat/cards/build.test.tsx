import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { $handoffError } from '@/app/contrib/handoff-receipt'
import { $activeSessionId, $selectedStoredSessionId, setSessionOwnerHint } from '@/store/session'

import { $setupHandoff, resetSetupHandoffForTests } from '../setup-profile'

import { HandoffCard, ProgressCard } from './build'

const identity = vi.hoisted(() => ({ message: 'progress-a' }))
vi.mock('@assistant-ui/react', () => ({
  useAuiState: (select: (state: { message: { id: string } }) => unknown) =>
    select({ message: { id: identity.message } })
}))

beforeEach(() => {
  $activeSessionId.set('guide-runtime')
  $selectedStoredSessionId.set('guide-stored')
  setSessionOwnerHint('guide-stored', { connectionId: 'guide-source', profile: 'hermes-setup' })
  resetSetupHandoffForTests()
  $handoffError.set(null)
})

it('waits for a settled directive and raises the handoff once across remounts', async () => {
  const attrs = { task: 'Tracker', brief: 'Build my tracker', plan: 'plugin' }
  const { rerender, unmount } = render(<HandoffCard attrs={attrs} locked />)

  expect($setupHandoff.get()).toBeNull()
  rerender(<HandoffCard attrs={attrs} locked={false} />)
  await waitFor(() => expect($setupHandoff.get()).not.toBeNull())
  const requested = $setupHandoff.get()
  expect(requested).toEqual({
    ...attrs,
    phase: 'pending',
    guide: {
      connectionId: 'guide-source',
      profile: 'hermes-setup',
      storedId: 'guide-stored',
      runtimeId: 'guide-runtime'
    }
  })
  unmount()
  render(<HandoffCard attrs={attrs} locked={false} />)
  expect($setupHandoff.get()).toBe(requested)
})

it('shows the actual failure and a deliberate retry rather than claiming an alternate build', () => {
  $setupHandoff.set({ task: 'Tracker', brief: 'Build my tracker', plan: 'build', phase: 'error' })
  $handoffError.set('Provider unavailable')
  render(<HandoffCard attrs={{ task: 'Tracker', brief: 'Build my tracker' }} locked={false} />)
  expect(screen.queryByText(/building here instead/)).toBeNull()
  expect(screen.getByText('Provider unavailable')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Retry first build' }))
  expect($setupHandoff.get()?.phase).toBe('pending')
  expect(screen.queryByRole('button', { name: 'Retry first build' })).toBeNull()
})

it('derives progress from the owning transcript across out-of-order remounts', () => {
  const messages = ['a', 'b'].map(id => ({
    id: `progress-${id}`,
    role: 'assistant' as const,
    parts: [{ type: 'text' as const, text: `::onboarding{step="progress" title="Step ${id}"}` }]
  }))

  const view = { ...PRIMARY_SESSION_VIEW, $messages: atom(messages) }

  const mount = (id: string) => {
    identity.message = `progress-${id}`

    return render(
      <SessionViewProvider value={view}>
        <ProgressCard attrs={{ title: `Step ${id}` }} locked={false} />
      </SessionViewProvider>
    )
  }

  let card = mount('a')
  card.unmount()
  card = mount('b')
  expect(card.container.textContent).toBe('Step aStep b')
  card.unmount()
  card = mount('a')
  expect(card.container.textContent).toBe('Step a')
  card.unmount()
  view.$messages.set([])
  card = mount('c')
  expect(card.container.textContent).toBe('Step c')
})

it('animates progress only while the owning turn streams', () => {
  const attrs = { title: 'Building' }
  const { container, rerender } = render(<ProgressCard attrs={attrs} locked />)
  expect(container.querySelector('.animate-pulse')).not.toBeNull()
  expect(container.textContent).toBe('Building…')
  rerender(<ProgressCard attrs={attrs} locked={false} />)
  expect(container.querySelector('.animate-pulse')).toBeNull()
  expect(container.textContent).toBe('Building')
})
