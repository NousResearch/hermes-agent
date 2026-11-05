import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { resolveMediaDisplaySrc, resolveMediaPlaybackSrc } from './media'

const win = window

// Each served profile's backend answers for its own roots: the shared path holds
// different bytes per profile, and only the tile owner's backend admits its outputs dir.
const BACKENDS: Record<string, Record<string, string>> = {
  foreground: { '/srv/shared/shot.png': 'data:image/png;base64,Rk9SRUdST1VORA==' },
  'tile-owner': {
    '/srv/shared/shot.png': 'data:image/png;base64,T1dORVI=',
    '/srv/tile-owner/outputs/plot.png': 'data:image/png;base64,UExPVA=='
  }
}

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

it.each([
  { name: 'foreground tile (control)', owner: 'foreground', path: '/srv/shared/shot.png' },
  { name: 'background tile, same path with other bytes', owner: 'tile-owner', path: '/srv/shared/shot.png' },
  {
    name: 'background tile, path only its backend admits',
    owner: 'tile-owner',
    path: '/srv/tile-owner/outputs/plot.png'
  }
])('reads local-connection media on the owning browser profile: $name', async ({ owner, path }) => {
  seedWebappSession('served-token')

  const fetchMock = vi.fn(async (input: URL, _init?: RequestInit) => {
    const dataUrl = BACKENDS[input.searchParams.get('profile') || '']?.[input.searchParams.get('path') || '']

    return dataUrl
      ? new Response(JSON.stringify({ dataUrl }), { status: 200 })
      : new Response(JSON.stringify({ detail: 'Path is outside the allowed roots' }), { status: 403 })
  })

  vi.stubGlobal('fetch', fetchMock)
  expect(installBrowserDesktopBridge()).toBe(true)
  // The user switched the page to another profile; the tile kept its captured owner.
  setApiRequestConnection(LOCAL_CONNECTION_ID)
  setApiRequestProfile('foreground')
  $connection.set({ connectionId: LOCAL_CONNECTION_ID, mode: 'remote', profile: 'foreground' } as never)

  await expect(
    resolveMediaDisplaySrc(`file://${path}`, { connectionId: LOCAL_CONNECTION_ID, profile: owner })
  ).resolves.toBe(BACKENDS[owner][path])
  const [requestUrl] = fetchMock.mock.calls.at(-1) as [URL, RequestInit]
  expect(requestUrl.pathname).toBe('/api/fs/read-data-url')
  expect(requestUrl.searchParams.get('profile')).toBe(owner)
})

it.each(['clip.mp3', 'clip.mp4'])('streams background-tile %s through the owning browser profile', async file => {
  seedWebappSession('served-token')

  const ticketProfiles: unknown[] = []

  const fetchMock = vi.fn(async (input: URL, init?: RequestInit) => {
    if (input.pathname === '/api/files/ticket') {
      ticketProfiles.push(JSON.parse(String(init?.body ?? '{}')).profile)

      return new Response(JSON.stringify({ ticket: 'owner-ticket' }), { status: 200 })
    }

    return new Response('{}', { status: 404 })
  })

  vi.stubGlobal('fetch', fetchMock)
  expect(installBrowserDesktopBridge()).toBe(true)
  setApiRequestConnection(LOCAL_CONNECTION_ID)
  setApiRequestProfile('foreground')
  $connection.set({ connectionId: LOCAL_CONNECTION_ID, mode: 'remote', profile: 'foreground' } as never)

  const src = await resolveMediaPlaybackSrc(`/srv/tile-owner/${file}`, {
    connectionId: LOCAL_CONNECTION_ID,
    profile: 'tile-owner'
  })

  expect(ticketProfiles).toEqual(['tile-owner'])
  expect(new URL(src).searchParams.get('profile')).toBe('tile-owner')
})
