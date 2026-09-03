import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

import { MessageAttachmentIndicator } from './user-message'

const nativeResizeObserver = globalThis.ResizeObserver

beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class ResizeObserverMock {
      disconnect() {}
      observe() {}
      unobserve() {}
    }
  )
})

afterAll(() => {
  vi.stubGlobal('ResizeObserver', nativeResizeObserver)
})

describe('MessageAttachmentIndicator', () => {
  it('keeps sent attachments behind a compact clipboard-count button', async () => {
    const { container } = render(
      <MessageAttachmentIndicator
        attachmentRefs={['@file:notes.md', '@file:report.pdf']}
        label="2 attachments"
      />
    )

    const trigger = screen.getByRole('button', { name: '2 attachments' })

    expect(trigger.className).toContain('size-7')
    expect(container.querySelector('[data-slot="aui_user-message-attachment-popover"]')).toBeNull()

    await act(async () => {
      fireEvent.click(trigger)
    })

    expect(container.ownerDocument.querySelector('[data-slot="aui_user-message-attachment-popover"]')).not.toBeNull()
    expect(screen.getByText('notes.md')).toBeDefined()
    expect(screen.getByText('report.pdf')).toBeDefined()
  })

  it('renders nothing without attachment references', () => {
    const { container } = render(<MessageAttachmentIndicator attachmentRefs={[]} label="0 attachments" />)

    expect(container.childElementCount).toBe(0)
  })
})
