import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ contributions: [] as Array<{ id: string; data?: unknown }> }))

vi.mock('@/contrib/react/use-contributions', () => ({
  useContributions: () => mocks.contributions
}))

import { ChatImageActions } from './image-contribution-actions'

afterEach(() => {
  cleanup()
  mocks.contributions = []
})

describe('ChatImageActions', () => {
  it('renders nothing without plugin contributions', () => {
    const { container } = render(<ChatImageActions src="https://example.com/image.png" />)
    expect(container.firstChild).toBeNull()
  })

  it('passes image source and tool provenance to a contributed action', () => {
    const onSelect = vi.fn()
    mocks.contributions = [
      {
        id: 'plugin:qc',
        data: { codicon: 'pass', label: 'Open in task QC', onSelect }
      }
    ]

    render(
      <ChatImageActions
        src="https://example.com/image.png"
        toolName="plugin__generate_image"
      />
    )
    fireEvent.click(screen.getByRole('button', { name: 'Open in task QC' }))

    expect(onSelect).toHaveBeenCalledWith({
      src: 'https://example.com/image.png',
      toolName: 'plugin__generate_image'
    })
  })

  it('honors contribution predicates for task-specific actions', () => {
    mocks.contributions = [
      {
        id: 'plugin:video-only',
        data: {
          label: 'Video QC',
          onSelect: vi.fn(),
          when: ({ toolName }: { toolName?: string }) => toolName?.includes('generate_video') === true
        }
      }
    ]

    render(<ChatImageActions src="https://example.com/image.png" toolName="generate_image" />)
    expect(screen.queryByRole('button', { name: 'Video QC' })).toBeNull()
  })
})
