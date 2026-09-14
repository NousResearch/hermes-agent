import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $selectedStoredSessionId, $sessions, setSessionOwnerHint } from '@/store/session'
import { $toolDisclosureStates } from '@/store/tool-view'

import { stubThreadEnvironment } from '../test-utils'

stubThreadEnvironment()

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'live-image-message', status: { type: 'running' } }, thread: { isRunning: true } })
}))

const { ToolFallback } = await import('./fallback')

const IMAGE =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII='

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  vi.restoreAllMocks()
})

describe('live tool image inspection', () => {
  it('opens a local input before tool completion, enlarges it, and keeps inspection independent of the running turn', async () => {
    $activeSessionId.set('runtime-image-session')
    $selectedStoredSessionId.set('image-session')
    setSessionOwnerHint('image-session', { connectionId: 'image-gateway', profile: 'artist' })
    const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
    vi.stubGlobal('hermesDesktop', { api })
    window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop

    const props = {
      args: { image_url: './renders/intermediate.png', question: 'Check this render' },
      toolCallId: 'image-call',
      toolName: 'vision_analyze',
      argsText: '',
      addResult: vi.fn(),
      resume: vi.fn(),
      respondToApproval: vi.fn(),
      type: 'tool-call' as const,
      status: { type: 'running' as const }
    }

    render(<ToolFallback {...props} />)
    expect(api).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /open image/i }))
    const thumbnail = await screen.findByRole('img')
    expect(thumbnail.getAttribute('src')).toBe(IMAGE)
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'image-gateway',
        profile: 'artist',
        path: '/api/fs/read-data-url?path=.%2Frenders%2Fintermediate.png&session_id=image-session'
      })
    )
    fireEvent.click(thumbnail)
    expect(screen.getByRole('dialog')).toBeTruthy()
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(screen.getByRole('img').getAttribute('src')).toBe(IMAGE)
    fireEvent.click(screen.getByRole('button', { name: /Analyzing image/i }))
    fireEvent.click(screen.getByRole('button', { name: /open image/i }))
    expect(screen.getByRole('img').getAttribute('src')).toBe(IMAGE)
    expect(api).toHaveBeenCalledTimes(1)
  })

  it('shows a recoverable failure without changing source ownership or falling back to local disk', async () => {
    $activeSessionId.set('runtime-failed-image-session')
    $selectedStoredSessionId.set('failed-image-session')
    $sessions.set([{ id: 'failed-image-session', connection_id: 'remote-artist', profile: 'artist' } as never])
    const api = vi.fn().mockRejectedValueOnce(new Error('File unavailable')).mockResolvedValue({ dataUrl: IMAGE })
    const readFileDataUrl = vi.fn()
    window.hermesDesktop = { api, readFileDataUrl } as unknown as typeof window.hermesDesktop
    render(
      <ToolFallback
        addResult={vi.fn()}
        args={{ image_url: './lost.png' }}
        argsText=""
        respondToApproval={vi.fn()}
        resume={vi.fn()}
        status={{ type: 'running' }}
        toolCallId="lost"
        toolName="vision_analyze"
        type="tool-call"
      />
    )
    fireEvent.click(screen.getByRole('button', { name: /open image/i }))
    expect(await screen.findByText('Preview unavailable')).toBeTruthy()
    expect(readFileDataUrl).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect((await screen.findByRole('img')).getAttribute('src')).toBe(IMAGE)
    expect(api).toHaveBeenCalledTimes(2)
    expect(api.mock.calls.every(([request]) => request.connectionId === 'remote-artist')).toBe(true)
  })
})
