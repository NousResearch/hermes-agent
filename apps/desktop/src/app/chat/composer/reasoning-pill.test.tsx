import { act, cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { applySessionInfoStatePatch, sessionInfoStatePatch } from '@/app/session/hooks/use-message-stream/utils'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionId,
  $defaultReasoningEffort,
  $selectedStoredSessionId,
  setCurrentReasoningEffort
} from '@/store/session'
import { $sessionStates, publishSessionState } from '@/store/session-states'

import { ReasoningPill } from './reasoning-pill'

const modelState = (over: Partial<ChatBarState['model']> = {}): ChatBarState['model'] => ({
  canSwitch: true,
  model: 'gpt-6',
  provider: 'openai',
  reasoningMenuContent: <div>menu</div>,
  ...over
})

const tileView = (reasoningEffort: string, reasoningEffortWire = ''): SessionView => ({
  kind: 'tile',
  $awaitingResponse: atom(false),
  $busy: atom(false),
  $cwd: atom(''),
  $fast: atom(false),
  $lastVisibleIsUser: atom(false),
  $messages: atom([]),
  $messagesEmpty: atom(true),
  $model: atom('tile/claude-sonnet'),
  $provider: atom('anthropic'),
  $reasoningEffort: atom(reasoningEffort),
  $reasoningEffortPending: atom(false),
  $reasoningEffortWire: atom(reasoningEffortWire),
  $runtimeId: atom('tile-runtime'),
  $storedId: atom('stored-tile'),
  $turnStartedAt: atom<number | null>(null)
})

afterEach(() => {
  cleanup()
  $defaultReasoningEffort.set('')
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
})

describe('ReasoningPill', () => {
  it('shows a clamped pick as what the route sends, never as a distinct level (#61634)', () => {
    // The gateway says this route clamps `ultra` to `max`: compact "Ultra→Max".
    const { unmount } = render(
      <SessionViewProvider value={tileView('ultra', 'max')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    const pill = screen.getByTestId('reasoning-pill')

    expect(pill.textContent).toBe('Ultra→Max')
    unmount()

    // A verbatim wire level (or one the gateway has not stamped yet) makes no claim.
    render(
      <SessionViewProvider value={tileView('high', 'high')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    expect(screen.getByTestId('reasoning-pill').textContent).toBe('High')
  })

  it("shows THIS surface's live effort, falling back to the profile default when the session has none", () => {
    $defaultReasoningEffort.set('high')

    const { unmount } = render(
      <SessionViewProvider value={tileView('low')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    expect(screen.getByTestId('reasoning-pill').textContent).toBe('Low')
    unmount()

    render(
      <SessionViewProvider value={tileView('')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    expect(screen.getByTestId('reasoning-pill').textContent).toBe('High')
  })

  it("never paints the profile default while a resumed session's own effort is still loading (#79807)", () => {
    // Profile says ultra; the session being reopened is pinned to max.
    $defaultReasoningEffort.set('ultra')
    setCurrentReasoningEffort('')
    $selectedStoredSessionId.set('stored-1')
    $activeSessionId.set(null)

    render(
      <SessionViewProvider value={PRIMARY_SESSION_VIEW}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    const label = () => screen.getByTestId('reasoning-pill').textContent

    // session.resume in flight: no runtime slice yet.
    expect(label()).not.toContain('Ultra')

    // Cold resume answered before the agent build, so no effort was reported.
    act(() => {
      publishSessionState('rt-1', { ...createClientSessionState('stored-1'), reasoningEffortPending: true })
      $activeSessionId.set('rt-1')
    })
    expect(label()).not.toContain('Ultra')

    // The built agent's session.info carries the session's real level.
    act(() => {
      const state = $sessionStates.get()['rt-1']!
      publishSessionState('rt-1', applySessionInfoStatePatch(state, sessionInfoStatePatch({ reasoning_effort: 'max' })))
    })
    expect(label()).toBe('Max')
  })

  it('falls back to the profile default once the runtime reports the session has no pin of its own', () => {
    $defaultReasoningEffort.set('ultra')
    $selectedStoredSessionId.set('stored-1')

    act(() => {
      publishSessionState('rt-1', { ...createClientSessionState('stored-1'), reasoningEffortPending: true })
      $activeSessionId.set('rt-1')
    })

    render(
      <SessionViewProvider value={PRIMARY_SESSION_VIEW}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    act(() => {
      const state = $sessionStates.get()['rt-1']!
      publishSessionState('rt-1', applySessionInfoStatePatch(state, sessionInfoStatePatch({ reasoning_effort: '' })))
    })
    expect(screen.getByTestId('reasoning-pill').textContent).toBe('Ultra')
  })

  it('hides when the catalog says the model has no reasoning control, but not while that is unknown', () => {
    const { unmount } = render(
      <SessionViewProvider value={tileView('medium')}>
        <ReasoningPill disabled={false} model={modelState({ supportsReasoning: false })} />
      </SessionViewProvider>
    )

    expect(screen.queryByTestId('reasoning-pill')).toBeNull()
    unmount()

    render(
      <SessionViewProvider value={tileView('medium')}>
        <ReasoningPill disabled={false} model={modelState({ supportsReasoning: undefined })} />
      </SessionViewProvider>
    )

    expect(screen.getByTestId('reasoning-pill')).toBeTruthy()
  })
})
