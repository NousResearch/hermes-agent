import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Locale } from '@/i18n'

import { useRestingComposerPlaceholder } from './placeholder'

interface PlaceholderProbeProps {
  followUpPlaceholders: readonly string[]
  locale: Locale
  newSessionPlaceholders: readonly string[]
  onConversationChanged?: (previousSessionId: string | null | undefined) => void
  sessionId?: string | null
}

function PlaceholderProbe(props: PlaceholderProbeProps) {
  const placeholder = useRestingComposerPlaceholder(props)

  return <p data-testid="placeholder">{placeholder}</p>
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useRestingComposerPlaceholder', () => {
  it('refreshes the resting placeholder when the locale changes without changing sessions', () => {
    const onConversationChanged = vi.fn()

    const { rerender } = render(
      <PlaceholderProbe
        followUpPlaceholders={['Send a follow-up']}
        locale="en"
        newSessionPlaceholders={['What are we building?']}
        onConversationChanged={onConversationChanged}
      />
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('What are we building?')

    rerender(
      <PlaceholderProbe
        followUpPlaceholders={['发送后续消息']}
        locale="zh"
        newSessionPlaceholders={['我们要构建什么？']}
        onConversationChanged={onConversationChanged}
      />
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('我们要构建什么？')
    expect(onConversationChanged).not.toHaveBeenCalled()
  })

  it('keeps the starter placeholder when a new session receives its first id', () => {
    const onConversationChanged = vi.fn()

    const { rerender } = render(
      <PlaceholderProbe
        followUpPlaceholders={['Send a follow-up']}
        locale="en"
        newSessionPlaceholders={['What are we building?']}
        onConversationChanged={onConversationChanged}
      />
    )

    rerender(
      <PlaceholderProbe
        followUpPlaceholders={['Send a follow-up']}
        locale="en"
        newSessionPlaceholders={['What are we building?']}
        onConversationChanged={onConversationChanged}
        sessionId="session-1"
      />
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('What are we building?')
    expect(onConversationChanged).not.toHaveBeenCalled()
  })
})
