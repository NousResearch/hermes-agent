import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setSessionOwnerHint } from '@/store/session'

import { type ForeignImportResult, foreignRequest, type ForeignSnapshot } from './api'

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
const sourceOwner = { connectionId: 'local', profile: 'default' }

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

const snapshot: ForeignSnapshot = {
  origin: { tool: 'claude-code', path: `desktop:${'a'.repeat(64)}`, foreign_session_id: 'foreign-one' },
  messages: [{ role: 'user', content: 'Please repair this' }],
  title: 'Repair imports'
}

function mount(onOpenSession = vi.fn(), viewOwner = owner) {
  return {
    onOpenSession,
    ...render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SessionImportView onClose={vi.fn()} onOpenSession={onOpenSession} owner={viewOwner} />
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

    if (method === 'export') {
      return snapshot
    }

    if (++attempts === 1) {
      throw new Error('Connection interrupted')
    }

    return { session_id: 'durable-one', already_imported: false }
  })
  const { onOpenSession } = mount()
  fireEvent.click(await screen.findByRole('button', { name: /Repair imports/ }))
  expect(screen.getByText('Client Alpha')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Grok Bot' })).toBeNull()
  await screen.findByText('Please repair this')
  expect(attempts).toBe(0)
  fireEvent.click(screen.getByRole('button', { name: 'Continue in Hermes' }))
  await screen.findByRole('alert')
  expect(onOpenSession).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Continue in Hermes' }))
  await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith('durable-one'))
  expect(setSessionOwnerHint).toHaveBeenCalledWith('durable-one', owner)
  expect(foreignRequest).toHaveBeenCalledWith(sourceOwner, 'export', { id: 'foreign-one' }, expect.any(AbortSignal))
  expect(foreignRequest).toHaveBeenCalledWith(owner, 'import', { snapshot }, expect.any(AbortSignal))
})

it('keeps a same-device import on its destination profile without exporting a snapshot', async () => {
  const localOwner = { connectionId: 'local', profile: 'research' }
  vi.mocked(foreignRequest).mockImplementation(async (_owner, method) => {
    if (method === 'list') {
      return { sessions: [session], next_offset: null, host: 'studio', unreadable: 0 }
    }

    if (method === 'preview') {
      return { messages: [], total: 0, already_imported: null }
    }

    return { session_id: 'local-one', already_imported: false }
  })

  const { onOpenSession } = mount(vi.fn(), localOwner)
  fireEvent.click(await screen.findByRole('button', { name: /Repair imports/ }))
  const continueButton = await screen.findByRole('button', { name: 'Continue in Hermes' })
  await waitFor(() => expect((continueButton as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(continueButton)
  await waitFor(() =>
    expect(foreignRequest).toHaveBeenCalledWith(localOwner, 'import', { id: 'foreign-one' }, expect.any(AbortSignal))
  )
  await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith('local-one'))
  expect(vi.mocked(foreignRequest).mock.calls.some(([, method]) => method === 'export')).toBe(false)
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

    if (method === 'export') {
      return snapshot
    }

    return new Promise<ForeignImportResult>(resolve => {
      finish = resolve
    })
  })
  const { onOpenSession, unmount } = mount()
  fireEvent.click(await screen.findByRole('button', { name: /Repair imports/ }))
  const continueButton = await screen.findByRole('button', { name: 'Continue in Hermes' })
  await waitFor(() => expect((continueButton as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(continueButton)
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
        sources: ['claude', 'cowork', 'codex', 'grok'],
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
  expect(screen.queryByRole('combobox')).toBeNull()
  const grok = screen.getByRole('button', { name: 'Grok Bot' })
  expect(grok.closest('aside')).toBeNull()
  expect(grok.closest('header')).toBeTruthy()
  expect(grok.querySelector('svg, i')).toBeNull()
  fireEvent.click(grok)
  await waitFor(() =>
    expect(foreignRequest).toHaveBeenCalledWith(
      sourceOwner,
      'list',
      { source: 'grok', offset: 0 },
      expect.any(AbortSignal)
    )
  )
  expect(grok.getAttribute('aria-pressed')).toBe('true')
  expect(screen.getByText(/History may be incomplete/)).toBeTruthy()
  expect(screen.getByText(/Cloud-only sessions are not included/i)).toBeTruthy()
})
