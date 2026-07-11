import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { writeKey } from '@/lib/storage'

import { VideoStudioView } from './index'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

beforeEach(() => {
  window.localStorage.clear()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api: vi.fn(async ({ path }: { path: string }) => {
        if (path === '/api/capabilities/video-library/libraries') {
          return {
            data: {
              libraries: [
                {
                  id: 'beef-noodle',
                  mode: 'linked',
                  name: '牛肉面资产库',
                  root: '/vault/牛肉面资产库',
                  source_roots: ['/vault/material'],
                  taxonomy: 'beef-noodle-v1'
                }
              ]
            },
            error: null,
            ok: true
          }
        }

        return { data: null, error: { code: 'TEST', message: 'not needed by this test' }, ok: false }
      })
    }
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('VideoStudioView named library integration', () => {
  it('renders only the unified material library entry', async () => {
    render(<VideoStudioView />)

    await waitFor(() => expect(screen.getByLabelText('素材来源')).toBeTruthy())
    fireEvent.change(screen.getByLabelText('素材来源'), { target: { value: 'local' } })
    expect(screen.getByLabelText('视频资产库')).toBeTruthy()
    expect(screen.getAllByText('素材库')).toHaveLength(1)
    expect(screen.queryByText('本地素材')).toBeNull()
    expect(screen.queryByText('视频素材库')).toBeNull()
    expect(screen.queryByText('Obsidian 具名资产库')).toBeNull()
  })

  it('does not restore a named library from the saved MoneyPrinter draft', async () => {
    writeKey(
      'hermes-video-studio-moneyprinter-draft-v1',
      JSON.stringify({ selectedLibraryId: 'beef-noodle', videoScript: '后厨现煮。' })
    )

    render(<VideoStudioView />)

    await waitFor(() => expect(screen.getByLabelText('素材来源')).toBeTruthy())
    expect(screen.queryByLabelText('视频资产库')).toBeNull()
  })

  it('creates an online-source video without requiring a named local library', async () => {
    writeKey(
      'hermes-video-studio-moneyprinter-draft-v1',
      JSON.stringify({ videoScript: '牛肉面出锅。', videoSource: 'pexels', videoSubject: '牛肉面' })
    )
    const requests: Array<{ body?: Record<string, unknown>; path: string }> = []
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async (request: { body?: Record<string, unknown>; path: string }) => {
          requests.push(request)
          if (request.path === '/api/capabilities/video-library/libraries') {
            return { data: { libraries: [] }, error: null, ok: true }
          }
          if (request.path === '/api/capabilities/moneyprinter/videos') {
            return {
              data: { task: { id: 'online-task', progress: 0, state: 'queued', videos: [] } },
              error: null,
              ok: true
            }
          }
          return { data: null, error: { code: 'TEST', message: 'not needed by this test' }, ok: false }
        })
      }
    })

    render(<VideoStudioView />)
    await waitFor(() => expect(screen.getByLabelText('素材来源')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'AI 自动匹配并生成视频' }))

    await waitFor(() =>
      expect(requests.find(request => request.path.endsWith('/moneyprinter/videos'))?.body).toMatchObject({
        video_source: 'pexels',
        video_subject: '牛肉面'
      })
    )
    expect(requests.some(request => request.path.endsWith('/video-library/timelines'))).toBe(false)
  })
})
