import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { SessionViewProvider } from '@/app/chat/session-view'
import { MarkdownImage } from '@/components/assistant-ui/markdown-text'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { resolveMediaPlaybackSrc } from './media'

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
