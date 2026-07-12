import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { downloadGatewayMediaFile, isRemoteGateway } from '@/lib/media'
import { ensureGatewayProfile } from '@/store/profile'

import { ArtifactsView } from './index'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn()
}))

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: vi.fn(async () => undefined),
  filePathFromMediaPath: (path: string) => path.replace(/^file:\/\//, ''),
  isRemoteGateway: vi.fn(() => false)
}))

vi.mock('@/store/profile', () => ({
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

const getMessagesMock = vi.mocked(getSessionMessages)
const listSessionsMock = vi.mocked(listAllProfileSessions)
const openExternal = vi.fn(async (_href: string) => undefined)
const downloadGatewayMediaFileMock = vi.mocked(downloadGatewayMediaFile)
const isRemoteGatewayMock = vi.mocked(isRemoteGateway)
const ensureGatewayProfileMock = vi.mocked(ensureGatewayProfile)

const session = {
  ended_at: null,
  id: 'session-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1000,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  profile: 'default',
  source: null,
  started_at: 1000,
  title: 'Artifact session',
  tool_call_count: 0
}

function renderArtifacts() {
  return render(
    <MemoryRouter initialEntries={['/artifacts']}>
      <ArtifactsView />
    </MemoryRouter>
  )
}

describe('ArtifactsView open affordances', () => {
  beforeEach(() => {
    openExternal.mockClear()
    downloadGatewayMediaFileMock.mockClear()
    isRemoteGatewayMock.mockReturnValue(false)
    ensureGatewayProfileMock.mockClear()
    listSessionsMock.mockResolvedValue({ sessions: [session] } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        openExternal
      }
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
    Reflect.deleteProperty(window, 'matchMedia')
  })

  it('labels a file row as an explicit open action and shows its typed thumbnail', async () => {
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/deployment-report.md', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    const open = await screen.findByRole('button', { name: 'Open deployment-report.md' })
    expect(screen.getByLabelText('Preview for deployment-report.md')).toBeTruthy()

    fireEvent.click(open)

    await waitFor(() => expect(ensureGatewayProfileMock).toHaveBeenCalledWith('default'))
    await waitFor(() => expect(openExternal).toHaveBeenCalledOnce())
    expect(openExternal.mock.calls[0]?.[0]).toContain('deployment-report.md')
  })

  it('keeps image zoom while adding a separate explicit open action', async () => {
    getMessagesMock.mockResolvedValue({
      messages: [
        {
          content: '![Portrait](https://example.com/output/portrait.png)',
          role: 'assistant',
          timestamp: 2000
        }
      ]
    } as never)

    renderArtifacts()

    const open = await screen.findByRole('button', { name: 'Open portrait.png' })
    expect(screen.queryByRole('img', { name: 'portrait.png' })).toBeNull()

    fireEvent.click(await screen.findByRole('button', { name: 'Load preview for portrait.png' }))
    expect(await screen.findByRole('img', { name: 'portrait.png' })).toBeTruthy()

    fireEvent.click(open)

    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('https://example.com/output/portrait.png'))
  })

  it('downloads a remote gateway file through the authenticated bridge without externalizing a token URL', async () => {
    isRemoteGatewayMock.mockReturnValue(true)
    listSessionsMock.mockResolvedValue({ sessions: [{ ...session, profile: 'work' }] } as never)
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/private-report.pdf', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    fireEvent.click(await screen.findByRole('button', { name: 'Open private-report.pdf' }))

    await waitFor(() => expect(ensureGatewayProfileMock).toHaveBeenCalledWith('work'))
    await waitFor(() => expect(downloadGatewayMediaFileMock).toHaveBeenCalledWith('/tmp/private-report.pdf', 'work'))
    expect(openExternal).not.toHaveBeenCalled()
  })

  it.each([
    ['![](data:image/png;base64,c21hbGw=)', 'Embedded image'],
    ['Generated /images/output/chart.png', 'chart.png']
  ])('does not offer Open for an untrusted or non-openable image source', async (content, label) => {
    getMessagesMock.mockResolvedValue({
      messages: [{ content, role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    await screen.findByText(label)
    expect(screen.queryByRole('button', { name: `Open ${label}` })).toBeNull()

    if (content.includes('data:image')) {
      expect(document.body.textContent).not.toContain('c21hbGw=')
      expect(document.body.textContent).toContain('data:image/png;…')
    }
  })

  it('shows a recoverable fatal state instead of claiming there are no artifacts', async () => {
    listSessionsMock.mockRejectedValueOnce(new Error('offline'))

    renderArtifacts()

    expect(await screen.findByText("Couldn't load artifacts")).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(screen.queryByText('No artifacts found')).toBeNull()
  })

  it('discloses partial results when a recent chat transcript cannot be read', async () => {
    listSessionsMock.mockResolvedValue({
      sessions: [session, { ...session, id: 'session-2', profile: 'work', title: 'Unavailable chat' }]
    } as never)
    getMessagesMock.mockImplementation(async id => {
      if (id === 'session-2') {
        throw new Error('transcript unavailable')
      }

      return {
        messages: [{ content: 'Generated /tmp/available.pdf', role: 'assistant', timestamp: 2000 }]
      } as never
    })

    renderArtifacts()

    expect(await screen.findByText('Some artifacts may be missing')).toBeTruthy()
    expect(screen.getByText("1 of 2 recent chats couldn't be read.")).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Load 1 more chats' })).toBeNull()
    expect(screen.getByText('available.pdf')).toBeTruthy()
  })

  it('discloses unavailable profiles separately from failed chat transcripts', async () => {
    listSessionsMock.mockResolvedValue({
      errors: [{ error: 'locked database', profile: 'work' }],
      sessions: [session],
      total: 1
    } as never)
    getMessagesMock.mockResolvedValue({ messages: [] } as never)

    renderArtifacts()

    expect(await screen.findByText('Some artifacts may be missing')).toBeTruthy()
    expect(screen.getByText("1 profile couldn't be read.")).toBeTruthy()
  })

  it('renders a terminal failure when every returned transcript fails', async () => {
    listSessionsMock.mockResolvedValue({ sessions: [session], total: 1 } as never)
    getMessagesMock.mockRejectedValue(new Error('unreadable transcript'))

    renderArtifacts()

    expect(await screen.findByText("Couldn't load artifacts")).toBeTruthy()
    expect(screen.queryByText('No artifacts found')).toBeNull()
  })

  it('keeps only the newest overlapping refresh result', async () => {
    getMessagesMock.mockResolvedValueOnce({
      messages: [{ content: 'Generated /tmp/initial.pdf', role: 'assistant', timestamp: 1000 }]
    } as never)
    renderArtifacts()
    expect(await screen.findByText('initial.pdf')).toBeTruthy()

    let resolveOlder!: (value: unknown) => void
    let resolveNewer!: (value: unknown) => void

    const older = new Promise(resolve => {
      resolveOlder = resolve
    })

    const newer = new Promise(resolve => {
      resolveNewer = resolve
    })

    getMessagesMock.mockReturnValueOnce(older as never).mockReturnValueOnce(newer as never)

    fireEvent.keyDown(window, { key: 'r' })
    await waitFor(() => expect(getMessagesMock).toHaveBeenCalledTimes(2))
    fireEvent.keyDown(window, { key: 'r' })
    await waitFor(() => expect(getMessagesMock).toHaveBeenCalledTimes(3))

    await act(async () => {
      resolveNewer({
        messages: [{ content: 'Generated /tmp/newest.pdf', role: 'assistant', timestamp: 3000 }]
      })
      await newer
    })
    expect(await screen.findByText('newest.pdf')).toBeTruthy()

    await act(async () => {
      resolveOlder({
        messages: [{ content: 'Generated /tmp/stale.pdf', role: 'assistant', timestamp: 2000 }]
      })
      await older
    })
    expect(screen.queryByText('stale.pdf')).toBeNull()
    expect(screen.getByText('newest.pdf')).toBeTruthy()
  })

  it('does not search inside an embedded image base64 payload', async () => {
    getMessagesMock.mockResolvedValue({
      messages: [{ content: '![Campaign board](data:image/png;base64,c21hbGw=)', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()
    expect(await screen.findByText('Campaign board')).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox', { name: 'Search artifacts...' }), {
      target: { value: 'c21hbGw' }
    })

    expect(await screen.findByText('No artifacts found')).toBeTruthy()
    expect(screen.queryByText('Campaign board')).toBeNull()
  })

  it('activates the owning profile before opening the source chat', async () => {
    listSessionsMock.mockResolvedValue({ sessions: [{ ...session, profile: 'work' }] } as never)
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/work-report.pdf', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()
    fireEvent.click(await screen.findByRole('button', { name: 'Open source chat for work-report.pdf' }))

    await waitFor(() => expect(ensureGatewayProfileMock).toHaveBeenCalledWith('work'))
  })

  it('discloses the recent-chat indexing scope and can load more history', async () => {
    listSessionsMock.mockResolvedValue({ sessions: [session], total: 75 } as never)
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/scoped-report.pdf', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    expect(await screen.findByText('Showing artifacts from 1 of 75 recent chats.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Load 30 more chats' }))

    await waitFor(() => expect(listSessionsMock).toHaveBeenCalledWith(60, 1))
  })

  it('uses action-complete cards instead of a horizontally clipped table on narrow screens', async () => {
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn(() => ({
        addEventListener: vi.fn(),
        matches: true,
        removeEventListener: vi.fn()
      }))
    })
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/mobile-report.pdf', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    expect(await screen.findByTestId('artifact-mobile-list')).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.getByRole('button', { name: 'Open mobile-report.pdf' }).className).toContain('min-h-11')
    expect(screen.getByRole('button', { name: 'Open source chat for mobile-report.pdf' }).className).toContain(
      'min-h-11'
    )
    expect(screen.getByRole('button', { name: 'Copy path' }).className).toContain('min-h-11')
  })

  it('uses mobile-sized pagination controls for populated multi-page results', async () => {
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn(() => ({
        addEventListener: vi.fn(),
        matches: true,
        removeEventListener: vi.fn()
      }))
    })
    getMessagesMock.mockResolvedValue({
      messages: [
        {
          content: Array.from({ length: 101 }, (_, index) => `/tmp/mobile-${index}.pdf`).join('\n'),
          role: 'assistant',
          timestamp: 2000
        }
      ]
    } as never)

    renderArtifacts()

    const next = await screen.findByRole('button', { name: 'Go to next page' })
    expect(next.className).toContain('max-md:min-h-11')
    expect(screen.getByRole('button', { name: 'Go to items page 2' }).className).toContain('max-md:size-11')
  })
})
