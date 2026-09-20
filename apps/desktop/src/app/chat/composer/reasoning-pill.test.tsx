import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $defaultReasoningEffort } from '@/store/session'

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
  $reasoningEffortWire: atom(reasoningEffortWire),
  $runtimeId: atom('tile-runtime'),
  $storedId: atom('stored-tile'),
  $turnStartedAt: atom<number | null>(null)
})

afterEach(() => {
  cleanup()
  $defaultReasoningEffort.set('')
})

describe('ReasoningPill', () => {
  it('shows a clamped pick as what the route sends, never as a distinct level (#61634)', () => {
    // The gateway says this route clamps `ultra` to `max`: compact "Ultra→Max",
    // tooltip in the CLI's `/reasoning` wording.
    const { unmount } = render(
      <SessionViewProvider value={tileView('ultra', 'max')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    const pill = screen.getByTestId('reasoning-pill')

    expect(pill.textContent).toBe('Ultra→Max')
    expect(pill.getAttribute('aria-label')).toBe('Effort: Ultra (sends Max on this route)')
    unmount()

    // A verbatim wire level (or one the gateway has not stamped yet) makes no claim.
    render(
      <SessionViewProvider value={tileView('high', 'high')}>
        <ReasoningPill disabled={false} model={modelState()} />
      </SessionViewProvider>
    )

    expect(screen.getByTestId('reasoning-pill').textContent).toBe('High')
    expect(screen.getByTestId('reasoning-pill').getAttribute('aria-label')).toBe('Effort: High')
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
