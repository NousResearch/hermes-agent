import { act, renderHook } from '@testing-library/react'
import { batch } from 'nanostores'
import { useEffect, useState } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $diskPluginsScanPending } from '@/contrib/runtime-loader'
import type * as Hermes from '@/hermes'
import { getSession } from '@/hermes'
import { createBrowserProfileBridge } from '@/lib/browser-profile'
import { activeConnectionScopeSuffix, rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import {
  $freshDraftKey,
  adoptNewSessionDraft,
  announceNewSessionDraftKey,
  requestVoiceConversationStart,
  rotateFreshDraftKey,
  stashSessionDraft
} from '@/store/composer'
import { gatewayActivationEpoch } from '@/store/gateway'
import { $activeGatewayProfile, requestFreshSession } from '@/store/profile'
import {
  $profileNavigationRequest,
  cancelProfileNavigationRestore,
  PROFILE_NAVIGATION_RESTORE_TIMEOUT_MS,
  requestProfileNavigationRestore
} from '@/store/profile-navigation'
import {
  $connection,
  getRememberedRoute,
  getRememberedSessionId,
  getSessionOwnerHint,
  setRememberedRoute,
  setRememberedSessionId,
  setSessionOwnerHint
} from '@/store/session'
import { $sessionListAuthority } from '@/store/session-list-authority'
import { $sessionTiles } from '@/store/session-states'
import type * as Windows from '@/store/windows'
import { makeSessionInfo } from '@/test/session-info'

import { requestComposerInsert } from '../../chat/composer/focus'
import { useFreshSessionRequest } from '../wiring-effects'

import { useDesktopIntegrations } from './use-desktop-integrations'
import { useProfileNavigationRestore } from './use-profile-navigation-restore'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  getSession: vi.fn()
}))

const { peer } = vi.hoisted(() => ({ peer: vi.fn(() => false) }))
vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<typeof Windows>()),
  isPeerInstanceWindow: peer
}))

const navigate = vi.fn()
const sessions = [makeSessionInfo({ id: 'saved', profile: 'coder' })]

function pick(profile = 'coder') {
  requestFreshSession()
  requestProfileNavigationRestore(profile)
}

function activate(profile = 'coder') {
  $activeGatewayProfile.set(profile)
  const request = $profileNavigationRequest.get()

  if (!request) {
    return
  }

  $profileNavigationRequest.set({ ...request, scope: activeConnectionScopeSuffix(), activation: 'ready' })
}

function loaded(profile = 'coder', status: 'loaded' | 'failed' = 'loaded') {
  $sessionListAuthority.set({ profile, scope: activeConnectionScopeSuffix(), epoch: gatewayActivationEpoch(), status })
}

// jsdom cannot manufacture trusted DOM events. Invoke the registered capture
// callback with the browser's trust bit; the separate synthetic-event test goes
// through real dispatchEvent and verifies its false bit is respected.
const captureListeners = new Map<string, EventListener>()

function trusted(type: string) {
  captureListeners.get(type)?.({ isTrusted: true, type } as Event)
}

function render(options: { profileReady?: boolean; rows?: typeof sessions } = {}) {
  return renderHook(() => {
    const [pathname, setPathname] = useState('/previous')
    useFreshSessionRequest(() => {
      rotateFreshDraftKey()
      window.history.replaceState(null, '', '#/')
      setPathname('/')
    })
    useProfileNavigationRestore({
      activeProfile: $activeGatewayProfile.get(),
      locationPathname: pathname,
      profileReady: options.profileReady ?? true,
      navigate,
      sessions: options.rows ?? sessions,
      resumeExhaustedSessionId: null
    })

    return { pathname, setPathname }
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  captureListeners.clear()
  const add = window.addEventListener.bind(window)
  vi.spyOn(window, 'addEventListener').mockImplementation((type, listener, options) => {
    if (options === true) {
      captureListeners.set(type, listener as EventListener)
    }

    add(type, listener, options)
  })
  peer.mockReturnValue(false)
  announceNewSessionDraftKey(null)
  window.localStorage.clear()
  window.history.replaceState(null, '', '#/previous')
  rescopeConnectionScopedStores({ mode: 'local' })
  $connection.set(null)
  $profileNavigationRequest.set(null)
  $sessionListAuthority.set(null)
  $sessionTiles.set([])
  $diskPluginsScanPending.set(false)
})

afterEach(() => {
  $profileNavigationRequest.set(null)
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it('voice conversation start cancels pending restore', () => {
  pick()
  requestVoiceConversationStart()
  expect($profileNavigationRequest.get()).toBeNull()
})

it('composer insert drops a pending restore before deferred dispatch', () => {
  pick()
  requestComposerInsert('from deep link', { target: 'main' })
  expect($profileNavigationRequest.get()).toBeNull()
})

it('starts a fresh draft immediately, then replaces it with the remembered session', () => {
  setRememberedRoute('/saved', 'coder')
  const oldKey = $freshDraftKey.get()
  const hook = render()
  act(() => pick())
  expect(hook.result.current.pathname).toBe('/')
  expect($freshDraftKey.get()).not.toBe(oldKey)
  expect(navigate).not.toHaveBeenCalled()
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).toHaveBeenCalledWith('/saved', { replace: true })
  expect($profileNavigationRequest.get()).toBeNull()
})

it('restores a remembered non-session page', () => {
  setRememberedRoute('/capabilities', 'coder')
  render()
  act(() => pick())
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).toHaveBeenCalledWith('/capabilities', { replace: true })
})

it('nothing remembered leaves main on the switch draft', () => {
  const hook = render()
  act(() => pick())
  const key = $freshDraftKey.get()
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).not.toHaveBeenCalled()
  expect(hook.result.current.pathname).toBe('/')
  expect($freshDraftKey.get()).toBe(key)
  expect($profileNavigationRequest.get()).toBeNull()
})

it('typing then sending during activation never restores even after the composer empties', () => {
  setRememberedRoute('/saved', 'coder')
  const hook = render()
  act(() => pick())
  act(() => {
    trusted('keydown')
    stashSessionDraft($freshDraftKey.get(), 'my first message', [])
    trusted('input')
    trusted('keydown') // Enter sends and clears the editor before session.create finishes.
    stashSessionDraft($freshDraftKey.get(), '', [])
  })
  expect($profileNavigationRequest.get()).toBeNull()
  act(() => {
    activate()
    loaded()
  })
  expect(hook.result.current.pathname).toBe('/')
  expect(navigate).not.toHaveBeenCalled()
})

it.each(['pointerdown', 'paste', 'drop', 'input', 'compositionstart'])(
  'a trusted %s cancels even before a restore effect mounts',
  type => {
    setRememberedRoute('/saved', 'coder')
    pick()
    trusted(type)
    expect($profileNavigationRequest.get()).toBeNull()
    render()
    act(() => {
      activate()
      loaded()
    })
    expect(navigate).not.toHaveBeenCalled()
  }
)

it('replacing a request cleans up its listeners and timer exactly once inside batch', () => {
  vi.useFakeTimers()
  const remove = vi.spyOn(window, 'removeEventListener')
  const clear = vi.spyOn(globalThis, 'clearTimeout')
  pick()
  const listeners = new Map(captureListeners)
  const timers = vi.getTimerCount()
  batch(() => {
    requestProfileNavigationRestore('other')

    for (const [type, listener] of listeners) {
      expect(remove.mock.calls.filter(call => call[0] === type && call[1] === listener)).toHaveLength(1)
    }

    expect(clear).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(timers)
  })
  expect(remove).toHaveBeenCalledTimes(listeners.size)
  expect(clear).toHaveBeenCalledTimes(1)
  expect($profileNavigationRequest.get()?.profile).toBe('other')
})

it.each([
  { key: '2', repeat: true },
  ...[
    'Shift',
    'Control',
    'Alt',
    'Meta',
    'CapsLock',
    'AltGraph',
    'NumLock',
    'ScrollLock',
    'Fn',
    'FnLock',
    'Symbol',
    'SymbolLock'
  ].map(key => ({ key }))
])('ignores a repeat or modifier-only keydown: %j', detail => {
  pick()
  const request = $profileNavigationRequest.get()
  captureListeners.get('keydown')?.({ isTrusted: true, type: 'keydown', ...detail } as KeyboardEvent)
  expect($profileNavigationRequest.get()).toBe(request)
  captureListeners.get('keydown')?.({ isTrusted: true, type: 'keydown', key: 'x' } as KeyboardEvent)
  expect($profileNavigationRequest.get()).toBeNull()
})

it('ignores pointerdown inside the profile rail', () => {
  const rail = document.createElement('div')
  rail.dataset.slot = 'profile-rail'
  const button = document.createElement('button')
  rail.append(button)
  pick()
  const request = $profileNavigationRequest.get()
  captureListeners.get('pointerdown')?.({ isTrusted: true, type: 'pointerdown', target: button } as unknown as Event)
  expect($profileNavigationRequest.get()).toBe(request)
  trusted('keydown')
  expect($profileNavigationRequest.get()).toBeNull()
})

it('untrusted synthetic events do not cancel a pending restore', () => {
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())

  for (const type of ['keydown', 'pointerdown', 'paste', 'drop', 'input', 'compositionstart']) {
    window.dispatchEvent(new Event(type))
  }

  expect($profileNavigationRequest.get()).not.toBeNull()
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).toHaveBeenCalledWith('/saved', { replace: true })
})

it.each(['consumed', 'cancelled', 'touched', 'timeout'])('removes every capture listener when %s', end => {
  vi.useFakeTimers()
  const remove = vi.spyOn(window, 'removeEventListener')
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())
  const listeners = new Map(captureListeners)
  act(() => {
    if (end === 'consumed') {
      activate()
      loaded()
    }

    if (end === 'cancelled') {
      cancelProfileNavigationRestore()
    }

    if (end === 'touched') {
      trusted('pointerdown')
    }

    if (end === 'timeout') {
      vi.advanceTimersByTime(PROFILE_NAVIGATION_RESTORE_TIMEOUT_MS)
    }
  })
  expect($profileNavigationRequest.get()).toBeNull()

  for (const [type, listener] of listeners) {
    expect(remove).toHaveBeenCalledWith(type, listener, true)
  }
})

it.each(['navigation', 'new-session'] as const)('yields to a later %s', action => {
  setRememberedRoute('/saved', 'coder')
  const hook = render()
  act(() => pick())
  act(() => {
    if (action === 'navigation') {
      window.history.pushState(null, '', '#/capabilities')
      hook.result.current.setPathname('/capabilities')
    }

    if (action === 'new-session') {
      requestFreshSession()
    }
  })
  act(() => {
    if ($profileNavigationRequest.get()) {
      activate()
    }

    loaded()
  })
  expect(navigate).not.toHaveBeenCalled()
  expect($profileNavigationRequest.get()).toBeNull()
})

it('the real Webapp remember bridge preserves the pending restore between activation and list authority', async () => {
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())
  act(() => activate())
  window.history.replaceState({ idx: 1, key: 'draft', usr: null }, '', '#/')
  const bridge = createBrowserProfileBridge({ basePath: '/', currentProfile: () => 'coder', openWindow: vi.fn() })
  await act(async () => {
    await bridge.profile!.remember('coder')
  })
  expect(window.location.search).toContain('profile=coder')
  expect(window.history.state).toEqual({ idx: 1, key: 'draft', usr: null })
  act(() => loaded())
  expect(navigate).toHaveBeenCalledWith('/saved', { replace: true })
})

it('does not restore in a peer window', () => {
  peer.mockReturnValue(true)
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())
  expect($profileNavigationRequest.get()).toBeNull()
  expect(navigate).not.toHaveBeenCalled()
})

it('waits for profileReady', () => {
  setRememberedRoute('/saved', 'coder')
  render({ profileReady: false })
  act(() => pick())
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).not.toHaveBeenCalled()
  expect($profileNavigationRequest.get()).not.toBeNull()
})

it('leaves main on its draft when the remembered session is already tiled', () => {
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())
  setSessionOwnerHint('saved', { connectionId: 'other', profile: 'other' })
  $sessionTiles.set([{ storedSessionId: 'saved', workspaceMode: 'sessions' } as never])
  const tiles = $sessionTiles.get()
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).not.toHaveBeenCalled()
  expect($sessionTiles.get()).toBe(tiles)
  expect(getSessionOwnerHint('saved')).toEqual({ connectionId: 'other', profile: 'other' })
  expect($profileNavigationRequest.get()).toBeNull()
  expect(adoptNewSessionDraft('saved')).toBe(false)
})

it('does not announce a draft key on live restore', () => {
  setRememberedRoute('/saved', 'coder')
  render()
  act(() => pick())
  act(() => {
    activate()
    loaded()
  })
  expect(navigate).toHaveBeenCalledWith('/saved', { replace: true })
  expect(adoptNewSessionDraft('saved')).toBe(false)
})

it('drops failed activation and leaves the draft', () => {
  const hook = render()
  act(() => pick())
  act(() => $profileNavigationRequest.set({ ...$profileNavigationRequest.get()!, activation: 'failed' }))
  expect(hook.result.current.pathname).toBe('/')
  expect($profileNavigationRequest.get()).toBeNull()
})

it('drops a failed list read and leaves the draft', () => {
  const hook = render()
  act(() => pick())
  act(() => {
    activate()
    loaded('coder', 'failed')
  })
  expect(hook.result.current.pathname).toBe('/')
  expect($profileNavigationRequest.get()).toBeNull()
})

it('expires a stalled restore without replacing the draft', () => {
  vi.useFakeTimers()
  const hook = render()
  act(() => pick())
  act(() => vi.advanceTimersByTime(PROFILE_NAVIGATION_RESTORE_TIMEOUT_MS))
  expect(hook.result.current.pathname).toBe('/')
  expect($profileNavigationRequest.get()).toBeNull()
  expect(navigate).not.toHaveBeenCalled()
})

it('drops a connection-scope change', () => {
  render()
  act(() => pick())
  act(() => activate())
  act(() => {
    rescopeConnectionScopedStores({ mode: 'remote', baseUrl: 'https://other', profile: 'coder' })
    loaded()
  })
  expect($profileNavigationRequest.get()).toBeNull()
  expect(navigate).not.toHaveBeenCalled()
})

it('a late B effect cannot clear C, which replaces only its own draft', () => {
  setRememberedRoute('/capabilities', 'c')
  let supersede = false
  renderHook(() => {
    // Runs before the restore effect but after it captured B during render.
    useEffect(() => {
      if (supersede && $profileNavigationRequest.get()?.profile === 'b') {
        supersede = false
        pick('c')
      }
    })
    const [pathname, setPathname] = useState('/previous')
    useFreshSessionRequest(() => {
      rotateFreshDraftKey()
      window.history.replaceState(null, '', '#/')
      setPathname('/')
    })
    useProfileNavigationRestore({
      activeProfile: $activeGatewayProfile.get(),
      locationPathname: pathname,
      profileReady: true,
      navigate,
      sessions,
      resumeExhaustedSessionId: null
    })
  })
  act(() => pick('a'))
  supersede = true
  act(() => pick('b'))
  expect($profileNavigationRequest.get()?.profile).toBe('c')
  act(() => {
    activate('c')
    loaded('c')
  })
  expect(navigate.mock.calls).toEqual([['/capabilities', { replace: true }]])
})

it('a boot-time switch followed by typing cannot be hijacked by cold restore after live restore drops', () => {
  setRememberedRoute('/saved', 'coder')
  let ready = false

  const hook = renderHook(() => {
    useFreshSessionRequest(() => {
      rotateFreshDraftKey()
      window.history.replaceState(null, '', '#/')
    })

    const common = {
      activeProfile: 'coder',
      locationPathname: '/',
      profileReady: ready,
      navigate,
      sessions,
      resumeExhaustedSessionId: null
    }

    useProfileNavigationRestore(common)
    useDesktopIntegrations({
      ...common,
      chatOpen: false,
      hasPreview: false,
      refreshSessions: vi.fn(),
      resumeLastSession: true,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map() }
    })
  })

  act(() => pick())
  act(() => trusted('keydown'))
  stashSessionDraft($freshDraftKey.get(), 'keep this', [])
  ready = true
  act(() => {
    activate()
    loaded()
  })
  hook.rerender()
  expect(navigate).not.toHaveBeenCalled()
  expect($profileNavigationRequest.get()).toBeNull()
})

it('an unlisted remembered session stays on the draft without a lookup or history writes', async () => {
  setRememberedRoute('/outside-page', 'coder')
  setRememberedSessionId('outside-page', 'coder')
  vi.mocked(getSession).mockResolvedValue(makeSessionInfo({ id: 'outside-page', profile: 'coder' }))
  const hook = render()
  act(() => pick())
  await act(async () => {
    activate()
    loaded()
  })
  expect(navigate).not.toHaveBeenCalled()
  expect(getSession).not.toHaveBeenCalled()
  expect(hook.result.current.pathname).toBe('/')
  expect($profileNavigationRequest.get()).toBeNull()
  expect(getRememberedRoute('coder')).toBe('/outside-page')
  expect(getRememberedSessionId('coder')).toBe('outside-page')
})
