import { cleanup, render, screen, waitFor } from '@testing-library/react'
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
  it('does not restore a named library from the saved MoneyPrinter draft', async () => {
    writeKey(
      'hermes-video-studio-moneyprinter-draft-v1',
      JSON.stringify({ selectedLibraryId: 'beef-noodle', videoScript: '后厨现煮。' })
    )

    render(<VideoStudioView />)

    await waitFor(() => expect(screen.getByLabelText('视频资产库')).toBeTruthy())
    expect((screen.getByLabelText('视频资产库') as HTMLSelectElement).value).toBe('')
  })
})
