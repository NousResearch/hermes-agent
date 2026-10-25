import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { PreviewTilePane } from '@/app/chat/right-rail/preview'
import { SessionViewProvider } from '@/app/chat/session-view'
import { MarkdownImage } from '@/components/assistant-ui/markdown-text'
import { $previewTabs } from '@/store/preview'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { readDesktopFileText } from './desktop-fs'
import { resolveMediaPlaybackSrc } from './media'

// The pane's consent dialog needs a QueryClientProvider; these tests read files, not consent.
vi.mock('@/app/chat/right-rail/real-profile-consent-dialog', () => ({ RealProfileConsentDialog: () => null }))

// WA-5: a Docker agent's /workspace is its own session's sandbox, so a Webapp file request names
// the session it belongs to. The gateway half is tests/hermes_cli/test_web_server_fs_session_workspace.py.
const OWNER = { connectionId: LOCAL_CONNECTION_ID, profile: 'default' }
const CLIP = '/workspace/clip.mp4'
const win = window
let requests: { body: Record<string, unknown>; method: string; url: URL }[] = []

beforeEach(() => {
  requests = []
  seedWebappSession('served-token')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: URL, init?: RequestInit) => {
      requests.push({ body: JSON.parse(String(init?.body ?? '{}')), method: init?.method ?? 'GET', url: input })

      const answers: Record<string, unknown> = {
        '/api/files/ticket': { ticket: 'stream-ticket' },
        '/api/fs/read-text': { path: '/workspace/note.txt', text: 'session note' },
        '/api/fs/write-text': { ok: true, path: '/workspace/note.txt' }
      }

      return new Response(JSON.stringify(answers[input.pathname] ?? {}), { status: 200 })
    })
  )
  expect(installBrowserDesktopBridge()).toBe(true)
  setApiRequestConnection(LOCAL_CONNECTION_ID)
  setApiRequestProfile('default')
  $connection.set({ connectionId: LOCAL_CONNECTION_ID, mode: 'remote', profile: 'default' } as never)
})

afterEach(() => {
  cleanup()
  $previewTabs.set([])
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $connection.set(null)
  window.dispatchEvent(new Event('beforeunload'))
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('media in a session sandbox', () => {
  it('tickets a clip for the session it belongs to', async () => {
    const src = new URL(await resolveMediaPlaybackSrc(CLIP, OWNER, 'sess-1'))
    const [ticket] = requests

    expect(ticket.body).toEqual({ path: CLIP, profile: 'default', route: 'stream', session_id: 'sess-1' })
    expect(Object.fromEntries(src.searchParams)).toEqual({
      path: CLIP,
      profile: 'default',
      session_id: 'sess-1',
      ticket: 'stream-ticket'
    })
  })

  it('names no session when the caller has none', async () => {
    await resolveMediaPlaybackSrc(CLIP, OWNER)

    expect(requests[0].body).toEqual({ path: CLIP, profile: 'default', route: 'stream' })
  })

  it('plays a transcript clip from the transcript session', async () => {
    const view = { $runtimeId: atom<null | string>('runtime-1'), $storedId: atom<null | string>('stored-1') }

    const { container } = render(
      <SessionViewProvider value={view as never}>
        <MarkdownImage alt="clip" src={CLIP} />
      </SessionViewProvider>
    )

    await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('ticket='))
    expect(requests[0].body).toMatchObject({ path: CLIP, route: 'stream', session_id: 'stored-1' })
  })
})

const NOTE = '/workspace/note.txt'

const readsOf = (route: string) => requests.filter(request => request.url.pathname === route)

describe('text previews in a session sandbox', () => {
  it('reads a preview for the session it belongs to', async () => {
    await readDesktopFileText(NOTE, { sessionId: 'sess-1' })
    await readDesktopFileText(NOTE)

    const [scoped, bare] = readsOf('/api/fs/read-text')

    expect(scoped.url.searchParams.get('path')).toBe(NOTE)
    expect(scoped.url.searchParams.getAll('session_id')).toEqual(['sess-1'])
    expect(bare.url.searchParams.get('path')).toBe(NOTE)
    expect(bare.url.searchParams.has('session_id')).toBe(false)
  })

  it('previews a tab from the session that owns it', async () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(Date.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
    $previewTabs.set([
      {
        id: 'file:note',
        sessionId: 'stored-1',
        target: { kind: 'file', label: 'note.txt', path: NOTE, previewKind: 'text', source: NOTE, url: `file://${NOTE}` }
      }
    ])

    await act(async () => {
      render(<PreviewTilePane tabId="file:note" />)
    })

    await waitFor(() => expect(readsOf('/api/fs/read-text')).not.toHaveLength(0))
    expect(readsOf('/api/fs/read-text')[0].url.searchParams.get('session_id')).toBe('stored-1')
  })
})

describe('native Desktop reads', () => {
  it('keeps a local read to its path', async () => {
    document.documentElement.removeAttribute('data-hermes-desktop-host')
    const readFileText = vi.fn(async () => ({ path: NOTE, text: 'local' }))
    Object.defineProperty(win, 'hermesDesktop', { configurable: true, value: { readFileText } })
    $connection.set({ mode: 'local' } as never)

    await readDesktopFileText(NOTE, { sessionId: 'sess-1' })

    expect(readFileText).toHaveBeenCalledWith(NOTE)
    expect(requests).toEqual([])
  })
})
