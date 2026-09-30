// #38372: sticky user messages are now an appearance preference. Off and the
// user bubble must lose `position: sticky` (and its z-index — only needed to
// float above clipped siblings) so it scrolls in normal flow, and the 2-line
// clamp must lift so long prompts render at full height like every other
// message. On (the default) nothing changes.
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $stickyUserMessagesEnabled } from '@/store/sticky-user-messages'

import { assistantMessage, stubThreadEnvironment, ThreadRuntime, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()

afterEach(() => {
  cleanup()
  $stickyUserMessagesEnabled.set(true)
})

function root(): HTMLElement {
  return screen.getByText('pin me please').closest('[data-slot="aui_user-message-root"]') as HTMLElement
}

describe('sticky user message preference (#38372)', () => {
  it('pins the bubble by default', () => {
    $stickyUserMessagesEnabled.set(true)
    render(
      <ThreadRuntime messages={[userMessage('user-1', 'pin me please'), assistantMessage()]}>
        <Thread />
      </ThreadRuntime>
    )

    expect(root().className).toContain('sticky')
    expect(root().className).toContain('z-40')
    // The 2-line clamp only makes sense on a pinned bubble.
    expect(root().querySelector('.sticky-human-clamp')).not.toBeNull()
  })

  it('scrolls the bubble in normal flow when the preference is off', () => {
    $stickyUserMessagesEnabled.set(false)
    render(
      <ThreadRuntime messages={[userMessage('user-1', 'pin me please'), assistantMessage()]}>
        <Thread />
      </ThreadRuntime>
    )

    const className = root().className
    expect(className).not.toContain('sticky')
    expect(className).not.toContain('z-40')
    // No pin, no clamp: long prompts render at full height in normal flow.
    expect(root().querySelector('.sticky-human-clamp')).toBeNull()
  })
})
