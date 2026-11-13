import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { useMediaPlaybackSrc } from './use-media-playback-src'

const win = window

afterEach(() => {
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $connection.set(null)
  window.dispatchEvent(new Event('beforeunload'))
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it.each([
  { expected: 'foreground', name: 'main thread (control)', scope: MAIN_COMPOSER_SCOPE },
  {
    expected: 'tile-owner',
    name: 'background tile',
    scope: { ...MAIN_COMPOSER_SCOPE, connectionId: LOCAL_CONNECTION_ID, profile: 'tile-owner', target: 'tile:t' as const }
  }
])('a transcript video streams on its scope owner: $name', async ({ expected, scope }) => {
  seedWebappSession('served-token')

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: URL) =>
      input.pathname === '/api/files/ticket'
        ? new Response(JSON.stringify({ ticket: 'scope-ticket' }), { status: 200 })
        : new Response('{}', { status: 404 })
    )
  )
  expect(installBrowserDesktopBridge()).toBe(true)
  setApiRequestConnection(LOCAL_CONNECTION_ID)
  setApiRequestProfile('foreground')
  $connection.set({ connectionId: LOCAL_CONNECTION_ID, mode: 'remote', profile: 'foreground' } as never)

  const wrapper = ({ children }: { children: ReactNode }) => (
    <ComposerScopeProvider value={scope}>{children}</ComposerScopeProvider>
  )

  const { result } = renderHook(() => useMediaPlaybackSrc('/srv/tile-owner/clip.mp4'), { wrapper })

  await waitFor(() => expect(result.current.src).not.toBe(''))
  expect(new URL(result.current.src).searchParams.get('profile')).toBe(expected)
})
