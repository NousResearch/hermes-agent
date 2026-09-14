import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { jsx } from 'react/jsx-runtime'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as SessionStore from '@/store/session'

import type { NativeChatBinding } from './native-chat-panel'

const {
  bindCreatedSession,
  requestComposerFocus,
  requestGatewayForAgent,
  retainForegroundSessionSurface,
  retainGatewayForAgent,
  resumeTile,
  sessionTileDelegate,
  setSessionOwnerHint,
  updateSession
} = vi.hoisted(() => ({
  bindCreatedSession: vi.fn(),
  requestComposerFocus: vi.fn(),
  requestGatewayForAgent: vi.fn(),
  retainForegroundSessionSurface: vi.fn(),
  retainGatewayForAgent: vi.fn(),
  resumeTile: vi.fn(),
  sessionTileDelegate: vi.fn(),
  setSessionOwnerHint: vi.fn(),
  updateSession: vi.fn()
}))

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent, retainGatewayForAgent }))
vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<typeof SessionStore>()),
  setSessionOwnerHint
}))
vi.mock('@/store/session-states', async () => {
  const { atom: createAtom } = await import('nanostores')

  return {
    $sessionStates: createAtom({}),
    $sessionTileDelegateRevision: createAtom(1),
    retainForegroundSessionSurface,
    sessionTileDelegate
  }
})
vi.mock('./composer/focus', () => ({ requestComposerFocus }))
// The panel lazy-loads the real transcript renderer; the stub keeps this test
// on the binding/controller behavior it qualifies (resume ownership, leases,
// focus) instead of the whole ChatView tree.
vi.mock('./session-tile', async () => {
  const { jsx } = await import('react/jsx-runtime')

  return {
    SessionChatSurface: (props: { runtimeId: string }) =>
      jsx('div', { 'data-runtime-id': props.runtimeId, 'data-testid': 'native-surface' })
  }
})

import { $sessionStates } from '@/store/session-states'

import { NativeChatPanel } from './native-chat-panel'

const route = {
  connectionId: 'scope-internal',
  mode: 'remote' as const,
  profile: 'Internal',
  targetProfile: 'internal-workspace'
}

function bindingFor(storedSessionId: string): NativeChatBinding {
  // A fresh object with the SAME route fields: what a plugin's inline
  // render produces on every render.
  return { route: { ...route }, storedSessionId }
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

describe('NativeChatPanel resume ownership', () => {
  beforeEach(() => {
    requestComposerFocus.mockReset()
    resumeTile.mockReset()
    retainForegroundSessionSurface.mockReset()
    retainForegroundSessionSurface.mockImplementation(() => vi.fn())
    sessionTileDelegate.mockReset()
    sessionTileDelegate.mockReturnValue({ bindCreatedSession, resumeTile, updateSession })
    setSessionOwnerHint.mockReset()
    updateSession.mockReset()
    $sessionStates.set({})
  })

  afterEach(cleanup)

  it('resumes the new binding when it changes while an earlier resume is still in flight', async () => {
    const first = deferred<string>()
    const second = deferred<string>()

    resumeTile.mockImplementation((storedSessionId: string) =>
      storedSessionId === 'stored-first' ? first.promise : second.promise
    )

    const view = render(jsx(NativeChatPanel, { binding: bindingFor('stored-first') }))

    await waitFor(() => expect(resumeTile).toHaveBeenCalledWith('stored-first', { refreshTranscript: true }))

    view.rerender(jsx(NativeChatPanel, { binding: bindingFor('stored-second') }))

    await waitFor(() => expect(resumeTile).toHaveBeenCalledWith('stored-second', { refreshTranscript: true }))

    await act(async () => {
      second.resolve('runtime-second')
    })

    const surface = await screen.findByTestId('native-surface')

    expect(surface.getAttribute('data-runtime-id')).toBe('runtime-second')

    // The stale resume belongs to the replaced binding: settling later must not
    // publish its runtime over the binding the panel is actually showing.
    await act(async () => {
      first.resolve('runtime-first')
    })

    expect(screen.getByTestId('native-surface').getAttribute('data-runtime-id')).toBe('runtime-second')
  })

  it('keeps its foreground lease and resume when an equivalent binding object is re-created inline', async () => {
    resumeTile.mockResolvedValue('runtime-stable')

    const view = render(jsx(NativeChatPanel, { binding: bindingFor('stored-stable') }))

    await screen.findByTestId('native-surface')

    const leases = retainForegroundSessionSurface.mock.calls.length
    const resumes = resumeTile.mock.calls.length

    view.rerender(jsx(NativeChatPanel, { binding: bindingFor('stored-stable') }))
    view.rerender(jsx(NativeChatPanel, { binding: bindingFor('stored-stable') }))

    expect(retainForegroundSessionSurface).toHaveBeenCalledTimes(leases)
    expect(resumeTile).toHaveBeenCalledTimes(resumes)
  })

  it('focuses only its own native composer when the surface asks', async () => {
    resumeTile.mockResolvedValue('runtime-focus')

    const view = render(jsx(NativeChatPanel, { binding: bindingFor('stored-focus') }))

    await screen.findByTestId('native-surface')

    expect(requestComposerFocus).not.toHaveBeenCalled()

    view.rerender(jsx(NativeChatPanel, { binding: bindingFor('stored-focus'), focusRequest: 1 }))

    await waitFor(() => expect(requestComposerFocus).toHaveBeenCalledWith('native:stored-focus'))
  })

  it('releases its foreground lease and drops a late resume when it unmounts', async () => {
    const pending = deferred<string>()
    const disposers: Array<ReturnType<typeof vi.fn>> = []

    resumeTile.mockReturnValue(pending.promise)
    retainForegroundSessionSurface.mockImplementation(() => {
      const dispose = vi.fn()

      disposers.push(dispose)

      return dispose
    })

    const view = render(jsx(NativeChatPanel, { binding: bindingFor('stored-close') }))

    await waitFor(() => expect(resumeTile).toHaveBeenCalledTimes(1))

    view.unmount()

    expect(disposers.at(-1)).toHaveBeenCalled()

    await act(async () => {
      pending.resolve('runtime-late')
    })

    expect(updateSession).not.toHaveBeenCalled()
  })
})
