import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { localPreviewTarget } from './local-preview'
import { resolveMediaDisplaySrc, resolveMediaPlaybackSrc } from './media'

// The Webapp asks its gateway for a Windows drive-letter or UNC file in the forms the gateway
// resolves: reads by the native path, streams by the file: URL itself. The gateway half,
// tests/hermes_cli/test_webapp_file_urls.py, answers these exact requests.
const FILES = [
  { dir: 'file:///C:/Users/me/renders', native: 'C:\\Users\\me\\renders' },
  { dir: 'file://nas/share/renders', native: '\\\\nas\\share\\renders' }
]

const OWNER = { connectionId: LOCAL_CONNECTION_ID, profile: 'default' }
const win = window
let requests: { body: Record<string, unknown>; url: URL }[] = []

beforeEach(() => {
  requests = []
  seedWebappSession('served-token')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: URL, init?: RequestInit) => {
      requests.push({ body: JSON.parse(String(init?.body ?? '{}')), url: input })

      const answers: Record<string, unknown> = {
        '/api/files/ticket': { ticket: 'stream-ticket' },
        '/api/fs/read-data-url': { dataUrl: 'data:image/png;base64,UE5H' },
        '/api/fs/read-text': { text: '<p>report</p>' }
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

describe.each(FILES)('a Webapp file under $dir', ({ dir, native }) => {
  it('reads an image by its native path', async () => {
    await expect(resolveMediaDisplaySrc(`${dir}/shot%20one.png`, OWNER)).resolves.toBe('data:image/png;base64,UE5H')

    const [read] = requests

    expect(read.url.pathname).toBe('/api/fs/read-data-url')
    expect(read.url.searchParams.get('path')).toBe(`${native}\\shot one.png`)
  })

  it('previews a page by its native path', async () => {
    const target = localPreviewTarget(`${dir}/report%20%231.html`)

    await expect(window.hermesDesktop.readFileText(target?.path ?? '')).resolves.toMatchObject({
      text: '<p>report</p>'
    })

    const [read] = requests

    expect(read.url.pathname).toBe('/api/fs/read-text')
    expect(read.url.searchParams.get('path')).toBe(`${native}\\report #1.html`)
  })

  it('streams a clip by its file URL, ticketed for that URL', async () => {
    const clip = `${dir}/clip%20one.mp4`
    const src = new URL(await resolveMediaPlaybackSrc(clip, OWNER))
    const [ticket] = requests

    expect(ticket.url.pathname).toBe('/api/files/ticket')
    expect(ticket.body).toEqual({ path: clip, profile: 'default', route: 'stream' })
    expect(src.pathname).toBe('/api/files/stream')
    expect(Object.fromEntries(src.searchParams)).toEqual({ path: clip, profile: 'default', ticket: 'stream-ticket' })
  })
})

it('refuses to stream a file URL that encodes a separator', async () => {
  // The gateway would decode the separator into another path segment, so the clip could be
  // a different file than the one its label names.
  for (const clip of ['file:///C:/Users/me/a%2F..%2Fclip.mp4', 'file:///C:/Users/me/a%5Cclip.mp4']) {
    await expect(window.hermesDesktop.getGatewayFileStreamUrl!({ ...OWNER, path: clip })).rejects.toThrow(
      'does not name one file'
    )
  }

  expect(requests).toEqual([])
})
