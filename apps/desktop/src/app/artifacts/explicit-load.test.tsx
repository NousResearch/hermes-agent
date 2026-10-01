import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { __resetLinkTitleCache } from '@/lib/external-link'

import { ArtifactsView } from './index'

const fixture = vi.hoisted(() => ({ content: '' }))
vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({
    sessions: [{ id: 'fixture', title: 'Fixture' }]
  }),
  getSessionMessages: async () => ({
    messages: [{ role: 'assistant', timestamp: 1000, content: fixture.content }]
  })
}))

afterEach(() => {
  cleanup()
  __resetLinkTitleCache()
  vi.unstubAllGlobals()
})

function gallery() {
  return render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )
}

it('does not load remote artifact images or titles merely by opening the gallery', async () => {
  fixture.content = 'https://example.com/private.png?data=fixture https://example.com/private-page?data=fixture'
  const fetchLinkTitle = vi.fn().mockResolvedValue('Remote title')
  const openExternal = vi.fn()
  vi.stubGlobal('hermesDesktop', { fetchLinkTitle, openExternal })
  const { container } = gallery()
  await screen.findByRole('link')
  await waitFor(() => expect(container.querySelector('article')).not.toBeNull())
  expect(fetchLinkTitle).not.toHaveBeenCalled()
  expect(container.querySelector('img[src^="https://example.com"]')).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
  await waitFor(() => expect(container.querySelector('img[src^="https://example.com/private.png"]')).not.toBeNull())
  expect(fetchLinkTitle).not.toHaveBeenCalled()

  fireEvent.click(screen.getByRole('link'), { ctrlKey: true, metaKey: true })
  await waitFor(() => expect(fetchLinkTitle).toHaveBeenCalledWith('https://example.com/private-page?data=fixture'))
  expect(openExternal).toHaveBeenCalledWith('https://example.com/private-page?data=fixture')
  await screen.findByRole('link', { name: 'Remote title' })
})

it.each(['', new Error('offline')])('keeps an opened link usable when title resolution returns %s', async result => {
  fixture.content = 'https://example.com/private-page'
  const fetchLinkTitle = result instanceof Error ? vi.fn().mockRejectedValue(result) : vi.fn().mockResolvedValue(result)
  const openExternal = vi.fn()
  vi.stubGlobal('hermesDesktop', { fetchLinkTitle, openExternal })
  gallery()
  const link = await screen.findByRole('link')
  const label = link.textContent
  expect(fetchLinkTitle).not.toHaveBeenCalled()
  fireEvent.click(link, { ctrlKey: true, metaKey: true })
  await waitFor(() => expect(fetchLinkTitle).toHaveBeenCalledTimes(1))
  expect(link.textContent).toBe(label)
  expect(link.getAttribute('href')).toBe('https://example.com/private-page')
  expect(openExternal).toHaveBeenCalledTimes(1)
})

it('retains automatic local file previews through the existing bridge', async () => {
  fixture.content = 'MEDIA:/tmp/generated/local.png'
  const readFileDataUrl = vi.fn().mockResolvedValue('data:image/png;base64,TE9DQUw=')
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  const { container } = gallery()
  await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/generated/local.png'))
  await waitFor(() => expect(container.querySelector('img[src^="data:image/png"]')).not.toBeNull())
  expect(screen.queryByRole('button', { name: 'Preview' })).toBeNull()
})
