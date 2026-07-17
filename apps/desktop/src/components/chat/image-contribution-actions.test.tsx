import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ contributions: [] as Array<{ id: string; data?: unknown }> }))

vi.mock('@/contrib/react/use-contributions', () => ({
  useContributions: () => mocks.contributions
}))

import { ChatImageActions, toolResultForImageAction } from './image-contribution-actions'

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
    const toolResult = { id: 'job-1', status: 'completed' }
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
        toolResult={toolResult}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: 'Open in task QC' }))

    expect(onSelect).toHaveBeenCalledWith({
      src: 'https://example.com/image.png',
      toolName: 'plugin__generate_image',
      toolResult
    })
  })

  it('passes only bounded media provenance fields to plugins', () => {
    const preview = toolResultForImageAction({
      credential: 'must-not-leak',
      id: 'job-1',
      params: { prompt: 'p'.repeat(5000), secret: 'drop-me' },
      results: { rawUrl: 'https://example.com/result.png', token: 'drop-me' }
    })

    expect(preview).toEqual({
      id: 'job-1',
      params: { prompt: 'p'.repeat(4096) },
      results: { rawUrl: 'https://example.com/result.png' }
    })
  })

  it('preserves read-only generation fields needed by contributed actions', () => {
    const preview = toolResultForImageAction({
      structuredContent: {
        items: [{
          id: 'job-1',
          model: 'video-model',
          params: {
            aspect_ratio: '9:16',
            duration: 5,
            height: 1280,
            medias: [{ data: { id: 'start-1', type: 'image_job', url: 'https://example.com/start.png' }, role: 'start_image' }],
            prompt: 'Walk toward the camera',
            width: 720
          },
          results: { rawUrl: 'https://example.com/result.mp4' },
          status: 'completed',
          type: 'video'
        }]
      }
    }) as { structuredContent: { items: Array<Record<string, unknown>> } }

    expect(preview.structuredContent.items[0]).toMatchObject({
      id: 'job-1',
      model: 'video-model',
      results: { rawUrl: 'https://example.com/result.mp4' },
      status: 'completed',
      type: 'video'
    })
    expect(preview.structuredContent.items[0].params).toMatchObject({
      aspect_ratio: '9:16', duration: 5, height: 1280, prompt: 'Walk toward the camera', width: 720
    })
  })

  it('bounds total sanitizer traversal across wide nested metadata', () => {
    const input = {
      items: Array.from({ length: 20 }, (_, outer) => ({
        data: {
          items: Array.from({ length: 20 }, (_, inner) => ({ id: `${outer}-${inner}`, prompt: 'metadata' }))
        },
        id: String(outer)
      }))
    }

    const preview = toolResultForImageAction(input)
    let nodes = 0

    const visit = (value: unknown) => {
      if (value === null || value === undefined) {
        return
      }

      nodes += 1

      if (Array.isArray(value)) {
        value.forEach(visit)
      } else if (typeof value === 'object') {
        Object.values(value).forEach(visit)
      }
    }

    visit(preview)
    expect(nodes).toBeLessThanOrEqual(256)
  })

  it('honors contribution predicates for task-specific actions', () => {
    const when = vi.fn(({ toolName }: { toolName?: string }) => toolName?.includes('generate_video') === true)

    mocks.contributions = [
      {
        id: 'plugin:video-only',
        data: {
          label: 'Video QC',
          onSelect: vi.fn(),
          when
        }
      }
    ]

    render(
      <ChatImageActions
        src="https://example.com/image.png?X-Amz-Signature=secret"
        toolName="generate_image"
        toolResult={{ prompt: 'private prompt', url: 'https://example.com/private.png?token=secret' }}
      />
    )
    expect(screen.queryByRole('button', { name: 'Video QC' })).toBeNull()
    expect(when).toHaveBeenCalledWith({ src: 'https://example.com/image.png', toolName: 'generate_image' })
  })
})
