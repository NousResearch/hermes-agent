import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $activeSessionId, $currentModel, setCurrentModel, setCurrentModelSource } from '@/store/session'

import { ModelPill } from './model-pill'

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
      $turnStartedAt: atom<number | null>(null)
    }

    render(
      <SessionViewProvider value={tileView}>
        <ModelPill
          disabled={false}
          model={modelState({ model: 'tile/claude-sonnet', provider: 'anthropic', modelMenuContent: <div /> })}
        />
      </SessionViewProvider>
    )

    expect(screen.getByText('Sonnet')).toBeTruthy()
    expect(screen.queryByText(/primary/i)).toBeNull()
  })
})

describe('ModelPill provider beside model', () => {
  it('shows the provider dimmed beside the model name', () => {
    render(<ModelPill disabled={false} model={modelState({ model: 'qwen3-max', provider: 'custom:my_pool' })} />)

    expect(screen.getByText('Qwen3 Max')).toBeTruthy()
    expect(screen.getByText('custom:my_pool')).toBeTruthy()
  })

  it('hides the provider when the model id already carries it', () => {
    render(<ModelPill disabled={false} model={modelState({ model: 'openai/gpt-6', provider: 'openai' })} />)

    expect(screen.getByText('GPT-6')).toBeTruthy()
    expect(screen.queryByText('openai')).toBeNull()
  })

  it('hides the provider when none is known', () => {
    render(<ModelPill disabled={false} model={modelState({ model: 'gpt-6', provider: '' })} />)

    expect(screen.getByText('GPT-6')).toBeTruthy()
    expect(screen.queryByText(/provider/i)).toBeNull()
  })

  it('updates provider when switching between providers serving the same model id', () => {
    const { rerender } = render(
      <ModelPill disabled={false} model={modelState({ model: 'gpt-5.5', provider: 'openai' })} />
    )
    expect(screen.getByText('GPT-5.5')).toBeTruthy()
    expect(screen.getByText('openai')).toBeTruthy()

    // Switch route to a different provider serving the same model
    rerender(<ModelPill disabled={false} model={modelState({ model: 'gpt-5.5', provider: 'nous' })} />)
    expect(screen.getByText('GPT-5.5')).toBeTruthy()
    expect(screen.getByText('nous')).toBeTruthy()
    expect(screen.queryByText('openai')).toBeNull()

    // Switch back
    rerender(<ModelPill disabled={false} model={modelState({ model: 'gpt-5.5', provider: 'openai' })} />)
    expect(screen.getByText('openai')).toBeTruthy()
    expect(screen.queryByText('nous')).toBeNull()
  })

  it('renders provider safely without raw html injection', () => {
    const malicious = '<script>alert("xss")</script>'
    render(<ModelPill disabled={false} model={modelState({ model: 'gpt-6', provider: malicious })} />)

    // React escapes text content; verify literal text is in the DOM and not an executable script element
    expect(screen.getByText(malicious)).toBeTruthy()
    expect(document.querySelector('script')).toBeNull()
  })

  it('preserves provider in the DOM even when visually truncated with a long label', () => {
    const longProvider = 'extremely-long-custom-enterprise-gateway-cluster-us-east-1'
    render(<ModelPill disabled={false} model={modelState({ model: 'gpt-6', provider: longProvider })} />)

    // CSS truncate class is applied to prevent layout overflow while preserving DOM text
    const providerSpan = screen.getByText(longProvider)
    expect(providerSpan).toBeTruthy()
    expect(providerSpan.className).toContain('truncate')
  })

  it('renders loading spinner and hides provider when model is loading or empty', () => {
    const { rerender } = render(
      <ModelPill disabled={false} model={modelState({ model: '', provider: 'openai' })} />
    )

    // Model name is empty -> spinner is shown, provider text is omitted
    expect(screen.queryByText('openai')).toBeNull()

    // Once model lands, spinner is replaced with model name and provider
    rerender(<ModelPill disabled={false} model={modelState({ model: 'gpt-6', provider: 'openai' })} />)
    expect(screen.getByText('GPT-6')).toBeTruthy()
  })
})
