import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import {
  $activeSessionId,
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests
} from '@/store/session'
import { $toolDisclosureStates } from '@/store/tool-view'

import { stubThreadEnvironment } from '../test-utils'

stubThreadEnvironment()
vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'gallery-regression', status: { type: 'running' } }, thread: { isRunning: true } })
}))
const { ToolFallback } = await import('./fallback')

const IMAGE =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII='

const OTHER =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII=\n'

const props = {
  type: 'tool-call' as const,
  toolName: 'vision_analyze',
  toolCallId: 'gallery-call',
  argsText: '',
  status: { type: 'running' as const },
  addResult: vi.fn(),
  resume: vi.fn(),
  respondToApproval: vi.fn()
}

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $cronSessions.set([])
  $messagingSessions.set([])
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
})

for (const [name, store] of [
  ['cron', $cronSessions],
  ['messaging', $messagingSessions]
] as const) {
  it(`uses the existing ${name} origin and recovers from an initial error when reopened`, async () => {
    $activeSessionId.set('runtime-gallery')
    $selectedStoredSessionId.set('stored-gallery')
    store.set([{ id: 'stored-gallery', connection_id: 'archive-owner', profile: 'artist' } as never])
    const api = vi.fn().mockRejectedValueOnce(new Error('temporary failure')).mockResolvedValue({ dataUrl: IMAGE })
    window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
    render(<ToolFallback {...props} args={{ image_url: './archive.png' }} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open image' }))
    await screen.findByText('Preview unavailable')
    fireEvent.click(screen.getByRole('button', { name: /Analyzing image/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Open image' }))
    expect((await screen.findByRole('img')).getAttribute('src')).toBe(IMAGE)
    expect(api).toHaveBeenCalledTimes(2)
    expect(api).toHaveBeenLastCalledWith(
      expect.objectContaining({
        connectionId: 'archive-owner',
        profile: 'artist',
        path: '/api/fs/read-data-url?path=.%2Farchive.png&session_id=stored-gallery'
      })
    )
  })
}

it('bounds a large gallery and lets keyboard navigation reach the final image', async () => {
  $activeSessionId.set('runtime-gallery')
  $selectedStoredSessionId.set('stored-gallery')
  $cronSessions.set([{ id: 'stored-gallery', connection_id: 'archive-owner', profile: 'artist' } as never])
  const pending: Array<() => void> = []
  const api = vi.fn(() => new Promise(resolve => pending.push(() => resolve({ dataUrl: IMAGE }))))
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  const result = { result: Array.from({ length: 40 }, (_, i) => `MEDIA:./render-${i}.png`).join('\n') }
  render(<ToolFallback {...props} args={{}} result={result} />)
  expect(api).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Open image (40)' }))
  await waitFor(() => expect(api.mock.calls.length).toBeGreaterThan(0))
  expect(api.mock.calls.length).toBeLessThanOrEqual(3)
  expect(screen.queryAllByRole('button', { name: /^Open image \d/ }).length).toBeLessThanOrEqual(5)
  const gallery = screen.getByRole('region', { name: 'Image gallery' })
  fireEvent.keyDown(gallery, { key: 'End' })
  expect(await screen.findByText('Image 40 of 40')).toBeTruthy()
  // Cancelled queued requests must not start behind a newly selected page.
  pending.splice(0).forEach(resolve => resolve())
  await waitFor(() => expect(api.mock.calls.length).toBeGreaterThan(3))
  expect(api.mock.calls.length).toBeLessThanOrEqual(6)
  pending.splice(0).forEach(resolve => resolve())
  await waitFor(() => expect(api.mock.calls.length).toBe(8))
  pending.splice(0).forEach(resolve => resolve())
  await screen.findByRole('img', { name: /Image 40 of 40/ })
})

it('navigates numbered images inside the lightbox without stealing arrow keys from the composer', async () => {
  $activeSessionId.set('runtime-keyboard')
  $selectedStoredSessionId.set('stored-keyboard')
  $messagingSessions.set([{ id: 'stored-keyboard', connection_id: 'image-owner', profile: 'default' } as never])
  window.hermesDesktop = {
    api: vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  } as unknown as typeof window.hermesDesktop
  render(
    <>
      <ToolFallback {...props} args={{}} result={{ result: 'MEDIA:./first.png\nMEDIA:./second.png' }} />
      <input aria-label="Steer" />
    </>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Open image (2)' }))
  fireEvent.click(await screen.findByRole('img', { name: /Image 1 of 2/ }))
  const dialog = screen.getByRole('dialog')
  fireEvent.keyDown(dialog, { key: 'ArrowRight' })
  await waitFor(() => expect(dialog.querySelector('img')?.getAttribute('alt')).toContain('Image 2 of 2'))
  fireEvent.keyDown(dialog, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  fireEvent.keyDown(screen.getByRole('textbox', { name: 'Steer' }), { key: 'ArrowLeft' })
  expect(screen.getByRole('img', { name: /Image 2 of 2/ })).toBeTruthy()
})

it('labels the visible thumbnail range and updates it on page navigation', async () => {
  $activeSessionId.set('runtime-range')
  $selectedStoredSessionId.set('stored-range')
  $cronSessions.set([{ id: 'stored-range', connection_id: 'range-owner', profile: 'default' }] as never)
  window.hermesDesktop = {
    api: vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  } as unknown as typeof window.hermesDesktop
  render(
    <ToolFallback {...props} args={{}} result={{ images: Array.from({ length: 7 }, (_, i) => `./image-${i}.png`) }} />
  )
  fireEvent.click(screen.getByRole('button', { name: 'Open image (7)' }))
  expect(await screen.findByText('Previews 1–5 of 7')).toBeTruthy()
  fireEvent.keyDown(screen.getByRole('region', { name: 'Image gallery' }), { key: 'End' })
  expect(await screen.findByText('Previews 6–7 of 7')).toBeTruthy()
  await screen.findByRole('img', { name: /Image 7 of 7/ })
})
