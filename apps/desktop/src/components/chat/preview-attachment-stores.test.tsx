import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { localPreviewTarget } from '@/lib/local-preview'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  _resetSessionOwnerHintsForTests,
  forgetSessionOwnerHintsForConnection,
  forgetSessionOwnerHintsForSession,
  setSessionOwnerHint
} from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  knownOwnerForSession,
  storedSessionIdForRuntimeId
} from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

// Runtime import keeps Electron's separately configured TS source graph out of
// the renderer typecheck; Vitest still executes the real canonical normalizer.
const registryModulePath = '../../../electron/connection-registry'

const { normalizeRegistry } = (await import(registryModulePath)) as {
  normalizeRegistry(raw: unknown): {
    version: number
    primary: string
    connections: Array<{ id: string; kind: 'local' | 'remote' | 'cloud' | 'ssh'; label: string }>
  }
}

const storedId = '20260915_025755_fa5997'
const runtimeId = 'runtime-local-resume'
const cwd = '/tmp/文件卡片验收'
const target = `${cwd}/中文 空格 + # 验收.txt`
const owner = { connectionId: 'local', profile: 'lr' }
// Use main's canonical schema/normalizer, not a made-up local cache entry.
const normalizedRegistry = normalizeRegistry({})

const registry = {
  ...normalizedRegistry,
  secureTokenStorage: false,
  connections: normalizedRegistry.connections.map(entry => ({ ...entry, tokenSet: false, tokenPreview: null }))
}

const originalDesktop = window.hermesDesktop
const openExternal = vi.fn(async () => undefined)
const revealPath = vi.fn(async () => true)

beforeEach(() => {
  _resetSessionOwnerHintsForTests({ storage: true })
  $sessionTiles.set([])
  $sessionStates.set({})
  $sessions.set([])
  $connection.set({ mode: 'local' } as never)
  $connectionsRegistry.set(registry)
  $currentCwd.set(cwd)
  $selectedStoredSessionId.set(storedId)
  $activeSessionId.set(runtimeId)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      normalizePreviewTarget: vi.fn(async (path: string, base?: string) => localPreviewTarget(path, base)),
      openExternal,
      revealPath
    }
  })
})

afterEach(() => {
  cleanup()
  _resetSessionOwnerHintsForTests({ storage: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  $connection.set(null)
  $connectionsRegistry.set(null)
  $currentCwd.set('')
  vi.clearAllMocks()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
})

it('loads authoritative ownership for a resumed card without mounting the connection switcher', async () => {
  $connectionsRegistry.set(null)
  setSessionOwnerHint(storedId, owner)
  const list = vi.fn(async () => registry)
  Object.assign(window.hermesDesktop!, { connections: { list } })
  expect(knownOwnerForSession(runtimeId)).toEqual(owner)
  expect($connectionsRegistry.get()?.connections.find(entry => entry.id === owner.connectionId)?.kind).toBeUndefined()
  render(
    <>
      <PreviewAttachment target={target} />
      <PreviewAttachment target={target} />
    </>
  )
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  fireEvent.click((await screen.findAllByRole('button', { name: 'Open file' }))[0])
  expect(list).toHaveBeenCalledTimes(1)
  expect($connectionsRegistry.get()).toEqual(registry)
  expect(openExternal).toHaveBeenCalledWith(localPreviewTarget(target)?.url)
})

it('keeps registry hydration fail-closed and preserves newer authority and owner precedence', async () => {
  const remoteRegistry = {
    ...registry,
    // Deliberately hostile response: spelling an id "local" is not proof.
    connections: registry.connections.map(entry => ({ ...entry, kind: 'remote' as const }))
  }

  for (const scenario of [
    'missing bridge',
    'rejected read',
    'missing entry',
    'remote id local',
    'unknown owner',
    'ambiguous owner',
    'remote mode',
    'local mode',
    'native null',
    'newer registry'
  ]) {
    cleanup()
    _resetSessionOwnerHintsForTests({ storage: true })
    $connectionsRegistry.set(null)
    openExternal.mockClear()
    revealPath.mockClear()

    const normalize = vi.fn(async (path: string, base?: string) =>
      scenario === 'native null' ? null : localPreviewTarget(path, base)
    )

    const list = vi.fn(async () => {
      if (scenario === 'rejected read') {
        throw new Error('IPC unavailable')
      }

      if (scenario === 'missing entry') {
        return { ...registry, connections: [] }
      }

      if (scenario === 'remote id local' || scenario === 'local mode') {
        return remoteRegistry
      }

      if (scenario === 'newer registry') {
        $connectionsRegistry.set(remoteRegistry)
      }

      return registry
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: normalize,
        openExternal,
        revealPath,
        ...(scenario === 'missing bridge' ? {} : { connections: { list } })
      }
    })

    if (scenario !== 'unknown owner') {
      setSessionOwnerHint(storedId, {
        ...owner,
        ...(scenario === 'remote mode' ? { mode: 'remote' as const } : {}),
        ...(scenario === 'local mode' ? { mode: 'local' as const } : {})
      })
    }

    if (scenario === 'ambiguous owner') {
      setSessionOwnerHint(storedId, { connectionId: 'other', profile: 'lr', mode: 'remote' })
      expect(knownOwnerForSession(runtimeId)).toBeUndefined()
    }

    await act(async () => {
      render(<PreviewAttachment target={target} />)
    })

    if (scenario === 'local mode') {
      // Explicit owner mode retains its existing precedence over registry kind.
      expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()
    } else {
      expect(screen.queryByRole('button', { name: 'Open file' }), scenario).toBeNull()
      expect(
        screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ }),
        scenario
      ).toBeNull()
    }

    expect(openExternal, scenario).not.toHaveBeenCalled()
    expect(revealPath, scenario).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()

    if (scenario === 'newer registry') {
      expect($connectionsRegistry.get()).toEqual(remoteRegistry)
    }
  }
})

it('resolves a resumed primary view through the real stores and exact local owner hint', async () => {
  setSessionOwnerHint(storedId, owner)
  expect(PRIMARY_SESSION_VIEW.$runtimeId.get()).toBe(runtimeId)
  expect(PRIMARY_SESSION_VIEW.$storedId.get()).toBe(storedId)
  expect(storedSessionIdForRuntimeId(runtimeId)).toBe(storedId)
  expect(knownOwnerForSession(runtimeId)).toEqual(owner)
  render(<PreviewAttachment target={target} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Open file' }))
  expect(openExternal).toHaveBeenCalledWith(localPreviewTarget(target)?.url)
  fireEvent.click(
    screen.getByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
  )
  expect(revealPath).toHaveBeenCalledWith(target)
})

it('reacts when a resumed primary session owner becomes known without changing its ids or cwd', async () => {
  render(<PreviewAttachment target={target} />)
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  expect(knownOwnerForSession(runtimeId)).toBeUndefined()
  act(() => setSessionOwnerHint(storedId, owner))
  expect(knownOwnerForSession(runtimeId)).toEqual(owner)
  expect(await screen.findByRole('button', { name: 'Open file' }, { timeout: 1500 })).toBeTruthy()

  // Revocation, remote ownership and ambiguous owners must also repaint,
  // without borrowing the ambient local connection as proof of ownership.
  act(() => forgetSessionOwnerHintsForSession(storedId))
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  act(() => setSessionOwnerHint(storedId, { ...owner, mode: 'remote' }))
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  act(() => setSessionOwnerHint(storedId, owner))
  await screen.findByRole('button', { name: 'Open file' })
  act(() => setSessionOwnerHint(storedId, { connectionId: 'remote', profile: 'lr', mode: 'remote' }))
  expect(knownOwnerForSession(runtimeId)).toBeUndefined()
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  act(() => forgetSessionOwnerHintsForConnection('remote'))
  await screen.findByRole('button', { name: 'Open file' })
  act(() => forgetSessionOwnerHintsForConnection('local'))
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()

  // A newly proven owner must not turn an authoritative native null into a
  // renderer-classified file; preview and download remain available.
  vi.mocked(window.hermesDesktop!.normalizePreviewTarget).mockResolvedValue(null)
  await act(async () => setSessionOwnerHint(storedId, owner))
  expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
  expect(openExternal).not.toHaveBeenCalled()
  expect(revealPath).not.toHaveBeenCalled()
})
