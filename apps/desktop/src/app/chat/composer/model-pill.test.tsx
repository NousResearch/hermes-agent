import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import {
  $activeSessionId,
  $currentModel,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentUsage
} from '@/store/session'

import { contextTokensRemaining, contextWindowUsage, ModelPill } from './model-pill'

const modelState = (over: Partial<ChatBarState['model']> = {}): ChatBarState['model'] => ({
  canSwitch: true,
  model: 'gpt-6',
  provider: 'openai',
  ...over
})

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentModelSource('')
  setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
})

// #62055: a manual composer pick is sticky and silently overrides the
// Settings → Model default for every NEW chat. The pill must say so.
describe('ModelPill pinned-override badge', () => {
  it('shows the pin dot on a draft running a manual pick', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set(null)

    render(<ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />)

    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
  })

  it('stays quiet when the composer reflects the profile default', () => {
    setCurrentModel('google/gemma-4-26b-a4b-it:free')
    setCurrentModelSource('default')
    $activeSessionId.set(null)

    render(<ModelPill disabled={false} model={modelState()} />)

    expect(screen.queryByTestId('model-pinned-dot')).toBeNull()
  })

  it('stays quiet on a live session (footer shows that session, not the pin)', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set('live-1')

    render(<ModelPill disabled={false} model={modelState()} />)

    expect(screen.queryByTestId('model-pinned-dot')).toBeNull()
  })

  it('is exercised in both render paths', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set(null)

    // Fallback (no live menu) path.
    const { unmount } = render(
      <ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />
    )

    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
    unmount()

    // Live-menu (dropdown) path.
    render(
      <ModelPill
        disabled={false}
        model={modelState({ model: 'deepseek/deepseek-v4-flash', modelMenuContent: <div /> })}
      />
    )
    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
    expect($currentModel.get()).toBe('deepseek/deepseek-v4-flash')
  })
})

describe('ModelPill per-surface model label', () => {
  it('shows the chat-bar model even when the primary global differs', () => {
    setCurrentModel('primary/model')
    $activeSessionId.set('primary-runtime')

    const tileView: SessionView = {
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
      $reasoningEffort: atom('high'),
      $runtimeId: atom('tile-runtime'),
      $storedId: atom('stored-tile'),
      $usage: atom({ calls: 0, context_max: 200_000, context_used: 50_000, input: 0, output: 0, total: 0 })
    }

    render(
      <SessionViewProvider value={tileView}>
        <ModelPill
          disabled={false}
          model={modelState({ model: 'tile/claude-sonnet', provider: 'anthropic', modelMenuContent: <div /> })}
        />
      </SessionViewProvider>
    )

    expect(screen.getByText('Sonnet · High')).toBeTruthy()
    expect(screen.queryByText(/primary/i)).toBeNull()
    expect(screen.getByTestId('context-window').getAttribute('aria-label')).toBe(
      'Context window. 25% full. 50k / 200k tokens used.'
    )
  })
})

describe('ModelPill context window', () => {
  it('shows a context ring immediately before the model picker', () => {
    setCurrentUsage({ calls: 0, context_max: 200_000, context_used: 50_000, input: 0, output: 0, total: 0 })

    render(<ModelPill disabled={false} model={modelState()} />)

    const counter = screen.getByTestId('context-window')

    expect(counter.getAttribute('aria-label')).toBe('Context window. 25% full. 50k / 200k tokens used.')
    expect(counter.nextElementSibling?.getAttribute('aria-label')).toBe('Open model picker')
  })

  it('derives the ring usage from the reported percentage when exact usage is unavailable', () => {
    expect(
      contextTokensRemaining({ calls: 0, context_max: 128_000, context_percent: 25, input: 0, output: 0, total: 0 })
    ).toBe(96_000)
    expect(
      contextWindowUsage({ calls: 0, context_max: 128_000, context_percent: 25, input: 0, output: 0, total: 0 })
    ).toMatchObject({ max: 128_000, percent: 25, used: 32_000 })
  })

  it('keeps an empty ring visible until the selected session reports its context window', () => {
    render(<ModelPill disabled={false} model={modelState()} />)

    expect(screen.getByTestId('context-window').getAttribute('aria-label')).toBe(
      'Context window data is not available yet.'
    )
  })

  it('stays out of the compact composer controls', () => {
    setCurrentUsage({ calls: 0, context_max: 200_000, context_used: 50_000, input: 0, output: 0, total: 0 })

    render(<ModelPill compact disabled={false} model={modelState()} />)

    expect(screen.queryByTestId('context-window')).toBeNull()
  })
})
