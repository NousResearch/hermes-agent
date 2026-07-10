import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { downloadGatewayMediaFile, isRemoteGateway } from '@/lib/media'

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

const getMessagesMock = vi.mocked(getSessionMessages)
const listSessionsMock = vi.mocked(listAllProfileSessions)
const openExternal = vi.fn(async (_href: string) => undefined)
const downloadGatewayMediaFileMock = vi.mocked(downloadGatewayMediaFile)
const isRemoteGatewayMock = vi.mocked(isRemoteGateway)

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
  })

  it('labels a file row as an explicit open action and shows its typed thumbnail', async () => {
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/deployment-report.md', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    const open = await screen.findByRole('button', { name: 'Open deployment-report.md' })
    expect(screen.getByLabelText('Preview for deployment-report.md')).toBeTruthy()

    fireEvent.click(open)

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
    expect(await screen.findByRole('img', { name: 'portrait.png' })).toBeTruthy()

    fireEvent.click(open)

    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('https://example.com/output/portrait.png'))
  })

  it('downloads a remote gateway file through the authenticated bridge without externalizing a token URL', async () => {
    isRemoteGatewayMock.mockReturnValue(true)
    getMessagesMock.mockResolvedValue({
      messages: [{ content: 'Generated /tmp/private-report.pdf', role: 'assistant', timestamp: 2000 }]
    } as never)

    renderArtifacts()

    fireEvent.click(await screen.findByRole('button', { name: 'Open private-report.pdf' }))

    await waitFor(() => expect(downloadGatewayMediaFileMock).toHaveBeenCalledWith('/tmp/private-report.pdf'))
    expect(openExternal).not.toHaveBeenCalled()
  })

  it.each([
    ['![Embedded](data:image/png;base64,c21hbGw=)', 'Embedded image'],
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
})
