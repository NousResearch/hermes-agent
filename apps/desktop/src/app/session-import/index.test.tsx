import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setSessionOwnerHint } from '@/store/session'

import { type ForeignImportResult, foreignRequest } from './api'

import { SessionImportView } from './index'

vi.mock('./api', () => ({ foreignRequest: vi.fn() }))
vi.mock('@/store/session', () => ({ setSessionOwnerHint: vi.fn() }))
vi.mock('@/components/assistant-ui/markdown-text', () => ({
  MarkdownTextContent: ({ text }: { text: string }) => <p>{text}</p>
}))
vi.mock('../overlays/overlay-view', () => ({
  OverlayView: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const owner = { connectionId: 'workstation', profile: 'research' }

const session = {
  id: 'foreign-one',
  source: 'claude',
  label: 'Claude Code',
  title: 'Repair imports',
  project: 'Client Alpha',
  cwd: '/work/project',
  mtime: 1000,
  turn_count: 2,
  excerpt: 'Help with imports'
}

function mount(onOpenSession = vi.fn()) {
  return {
    onOpenSession,
    ...render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SessionImportView onClose={vi.fn()} onOpenSession={onOpenSession} owner={owner} />
      </QueryClientProvider>
    )
  }
}

it('browses without importing, then retries a failed import on the captured owner before opening', async () => {
  let attempts = 0
  vi.mocked(foreignRequest).mockImplementation(async (_owner, method) => {
    if (method === 'list') {
      return { sessions: [session], next_offset: null, host: 'studio', unreadable: 0 }
    }

    if (method === 'preview') {
      return { messages: [{ role: 'user', content: 'Please repair this' }], total: 1, already_imported: null }
    }

    if (++attempts === 1) {
      throw new Error('Connection interrupted')
    }

    return { session_id: 'durable-one', already_imported: false }
  })
  const { onOpenSession } = mount()
  fireEvent.click(await screen.findByRole('button', { name: /Repair imports/ }))
  expect(screen.getByText('Client Alpha')).toBeTruthy()
  await screen.findByText('Please repair this')
  expect(attempts).toBe(0)
  fireEvent.click(screen.getByRole('button', { name: 'Continue in Hermes' }))
  await screen.findByRole('alert')
  expect(onOpenSession).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Continue in Hermes' }))
  await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith('durable-one'))
  expect(setSessionOwnerHint).toHaveBeenCalledWith('durable-one', owner)
  expect(foreignRequest).toHaveBeenCalledWith(owner, 'import', { id: 'foreign-one' }, expect.any(AbortSignal))
})

it('does not navigate when an import finishes after the view has closed', async () => {
  let finish!: (result: ForeignImportResult) => void
  vi.mocked(foreignRequest).mockImplementation(async (_owner, method) => {
    if (method === 'list') {
      return { sessions: [session], next_offset: null, host: 'studio', unreadable: 0 }
    }

    if (method === 'preview') {
      return { messages: [], total: 0, already_imported: 'existing' }
    }

    return new Promise<ForeignImportResult>(resolve => {
      finish = resolve
    })
  })
  const { onOpenSession, unmount } = mount()
  fireEvent.click(await screen.findByRole('button', { name: /Repair imports/ }))
  fireEvent.click(await screen.findByRole('button', { name: 'Open in Hermes' }))
  await waitFor(() => expect(finish).toBeDefined())
  unmount()
  finish({ session_id: 'existing', already_imported: true })
  await waitFor(() => expect(setSessionOwnerHint).toHaveBeenCalledWith('existing', owner))
  expect(onOpenSession).not.toHaveBeenCalled()
})

it('shows installed local sources and explains that cloud sessions are excluded', async () => {
  vi.mocked(foreignRequest).mockImplementation(async (_owner, method) => {
    if (method === 'list') {
      return {
        sessions: [],
        sources: ['claude', 'cowork', 'codex'],
        next_offset: null,
        host: 'studio',
        unreadable: 0
      }
    }

    throw new Error(`Unexpected ${method}`)
  })

  mount()
  expect(await screen.findByRole('button', { name: 'Claude Cowork' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'ChatGPT Work / Codex' })).toBeTruthy()
  expect(screen.getByText(/Cloud-only sessions are not included/i)).toBeTruthy()
})
