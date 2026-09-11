import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { StickyHumanMessageContainer } from './user-message'

describe('sticky user message glass surface', () => {
  it('stays translucent without changing its sticky mask or layout', () => {
    const { container } = render(
      <StickyHumanMessageContainer attachments={<div data-testid="attachments" />} messageId="user-1">
        <div data-testid="bubble" />
      </StickyHumanMessageContainer>
    )

    const row = container.querySelector('[data-slot="aui_user-message-root"]')

    expect(row).not.toBeNull()
    expect(row?.hasAttribute('data-glass-opaque')).toBe(false)
    for (const className of [
      'sticky',
      'bg-(--ui-chat-surface-background)',
      '-mx-4',
      'w-[calc(100%+2rem)]',
      'px-4',
      'pb-(--conversation-turn-gap)'
    ]) {
      expect(row?.classList.contains(className)).toBe(true)
    }
    expect(screen.getByTestId('bubble').parentElement).toBe(row)
    expect(screen.getByTestId('attachments').previousElementSibling).toBe(row)
  })
})
