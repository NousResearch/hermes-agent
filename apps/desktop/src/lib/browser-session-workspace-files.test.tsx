import '@/store/session-states'

import { EditorView } from '@codemirror/view'
import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { PreviewTilePane } from '@/app/chat/right-rail/preview'
import { LocalFilePreview } from '@/app/chat/right-rail/preview-file'
import { SessionViewProvider } from '@/app/chat/session-view'
import { MarkdownImage } from '@/components/assistant-ui/markdown-text'
import { $previewTabs } from '@/store/preview'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { readDesktopFileText, writeDesktopFileText } from './desktop-fs'
import { normalizeOrLocalPreviewTarget } from './local-preview'
import { resolveMediaPlaybackSrc } from './media'

// The pane's consent dialog needs a QueryClientProvider; these tests read files, not consent.
vi.mock('@/app/chat/right-rail/real-profile-consent-dialog', () => ({ RealProfileConsentDialog: () => null }))
vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

// WA-5: a Docker agent's /workspace is its own session's sandbox, so a Webapp file request names
// the session it belongs to. The gateway half is tests/hermes_cli/test_web_server_fs_session_workspace.py.
const OWNER = { connectionId: LOCAL_CONNECTION_ID, profile: 'default' }
const CLIP = '/workspace/clip.mp4'
const win = window
let requests: { body: Record<string, unknown>; method: string; url: URL }[] = []

// The profile each session belongs to. `ops-session` is a Bot's: the window stays on `default`.
const SESSION_OWNERS: Record<string, string> = { 'ops-session': 'ops', 'sess-1': 'default', 'stored-1': 'default' }

const profileOf = (request: (typeof requests)[number]) =>
  request.url.searchParams.get('profile') ?? (request.body.profile as string | undefined) ?? null

const sessionOf = (request: (typeof requests)[number]) =>
  request.url.searchParams.get('session_id') ?? (request.body.session_id as string | undefined) ?? null

beforeEach(() => {
  requests = []
  seedWebappSession('served-token')
  _resetSessionOwnerHintsForTests()

  for (const [sessionId, profile] of Object.entries(SESSION_OWNERS)) {
    setSessionOwnerHint(sessionId, { connectionId: LOCAL_CONNECTION_ID, profile })
  }

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: URL, init?: RequestInit) => {
      const request = { body: JSON.parse(String(init?.body ?? '{}')), method: init?.method ?? 'GET', url: input }

      requests.push(request)
      const sessionId = sessionOf(request)

      // The gateway looks a session up in the profile the request names, nowhere else.
      if (sessionId && SESSION_OWNERS[sessionId] !== (profileOf(request) ?? 'default')) {
        return new Response(JSON.stringify({ detail: 'Session not found' }), { status: 404 })
      }

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
  _resetSessionOwnerHintsForTests({ storage: true })
  $previewTabs.set([])
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  globalThis.document.documentElement.removeAttribute('data-hermes-desktop-host')
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

describe('Save in a session sandbox', () => {
  it('saves a file for the session it was read for', async () => {
    await writeDesktopFileText(NOTE, 'edited', { sessionId: 'sess-1' })
    await writeDesktopFileText(NOTE, 'edited')

    const [scoped, bare] = readsOf('/api/fs/write-text')

    expect(scoped).toMatchObject({ body: { content: 'edited', path: NOTE, session_id: 'sess-1' }, method: 'POST' })
    expect(bare.body).toEqual({ content: 'edited', path: NOTE })
  })

  it('saves an edited preview to the session the tab belongs to', async () => {
    const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)

    Object.defineProperties(Range.prototype, {
      getClientRects: { configurable: true, value: () => [] },
      getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
    })

    try {
      const rendered = render(
        <LocalFilePreview
          reloadKey={0}
          sessionId="stored-1"
          target={{ kind: 'file', label: 'note.txt', path: NOTE, previewKind: 'text', source: NOTE, url: `file://${NOTE}` }}
        />
      )

      fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
      const editor = EditorView.findFromDOM(rendered.container.querySelector('.cm-content')!)!
      act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'edited' } }))
      fireEvent.click(screen.getByRole('button', { name: /Save/ }))

      await waitFor(() => expect(readsOf('/api/fs/write-text')).toHaveLength(1))
      expect(readsOf('/api/fs/write-text')[0].body).toEqual({ content: 'edited', path: NOTE, session_id: 'stored-1' })
      // The first load, the stale-on-disk check before the write and the reload after it all
      // read the same session's copy.
      const sessions = readsOf('/api/fs/read-text').map(read => read.url.searchParams.get('session_id'))

      expect(sessions.length).toBeGreaterThanOrEqual(2)
      expect(new Set(sessions)).toEqual(new Set(['stored-1']))
    } finally {
      for (const key of ['getClientRects', 'getBoundingClientRect']) {
        if (rangeDescriptors[key]) {
          Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
        } else {
          Reflect.deleteProperty(Range.prototype, key)
        }
      }
    }
  })
})

describe('a session owned by another profile than the window', () => {
  const OPS = 'ops-session'

  it('reads its file with the profile that owns it', async () => {
    await expect(readDesktopFileText(NOTE, { sessionId: OPS })).resolves.toMatchObject({ text: 'session note' })

    const [read] = readsOf('/api/fs/read-text')

    expect([profileOf(read), sessionOf(read)]).toEqual(['ops', OPS])
  })

  it('saves its file with the profile that owns it', async () => {
    await expect(writeDesktopFileText(NOTE, 'edited', { sessionId: OPS })).resolves.toEqual({ path: NOTE })

    const [save] = readsOf('/api/fs/write-text')

    expect([profileOf(save), sessionOf(save)]).toEqual(['ops', OPS])
  })

  it('tickets its clip with the profile that owns it', async () => {
    const src = new URL(await resolveMediaPlaybackSrc(CLIP, OWNER, OPS))

    expect(requests[0].body).toEqual({ path: CLIP, profile: 'ops', route: 'stream', session_id: OPS })
    expect(Object.fromEntries(src.searchParams)).toMatchObject({ profile: 'ops', session_id: OPS })
  })

  it('enriches a preview of its file with the profile that owns it', async () => {
    await normalizeOrLocalPreviewTarget(NOTE, '/workspace', { sessionId: OPS })

    const [read] = readsOf('/api/fs/read-text')

    expect([profileOf(read), sessionOf(read)]).toEqual(['ops', OPS])
  })

  it('names no session when its owner is unknown, as a sessionless read does', async () => {
    await readDesktopFileText(NOTE, { sessionId: 'unknown-session' })
    await writeDesktopFileText(NOTE, 'edited', { sessionId: 'unknown-session' })
    await resolveMediaPlaybackSrc(CLIP, OWNER, 'unknown-session')

    expect(requests.map(request => [request.url.pathname, profileOf(request), sessionOf(request)])).toEqual([
      ['/api/fs/read-text', 'default', null],
      ['/api/fs/write-text', 'default', null],
      ['/api/files/ticket', 'default', null]
    ])
  })
})

describe('native Desktop reads', () => {
  it('keeps a local save to its path', async () => {
    globalThis.document.documentElement.removeAttribute('data-hermes-desktop-host')
    const writeTextFile = vi.fn(async () => ({ path: NOTE }))
    Object.defineProperty(win, 'hermesDesktop', { configurable: true, value: { writeTextFile } })
    $connection.set({ mode: 'local' } as never)

    await writeDesktopFileText(NOTE, 'edited', { sessionId: 'sess-1' })

    expect(writeTextFile).toHaveBeenCalledWith(NOTE, 'edited')
    expect(requests).toEqual([])
  })

  it('keeps a local read to its path', async () => {
    globalThis.document.documentElement.removeAttribute('data-hermes-desktop-host')
    const readFileText = vi.fn(async () => ({ path: NOTE, text: 'local' }))
    Object.defineProperty(win, 'hermesDesktop', { configurable: true, value: { readFileText } })
    $connection.set({ mode: 'local' } as never)

    await readDesktopFileText(NOTE, { sessionId: 'sess-1' })

    expect(readFileText).toHaveBeenCalledWith(NOTE)
    expect(requests).toEqual([])
  })
})
