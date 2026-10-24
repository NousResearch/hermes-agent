import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { usePlayerAgentBridge } from '@/lib/player-agent-bridge'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'
import { $focusedRuntimeId, $sessionTiles } from '@/store/session-states'

describe('player linking to the visible chat', () => {
  const originalDesktop = window.hermesDesktop
  const linkAwareness = vi.fn(async scope => ({ linked: true, scope }))
  const unlinkAwareness = vi.fn(async () => ({ linked: false, scope: null }))
  const agentRequest = vi.fn(async request => ({ operation: request.operation.type }))
  const owner = { connectionId: 'preview-remote', profile: 'default' }

  beforeEach(() => {
    vi.clearAllMocks()
    setWorkspaceScope('sessions')
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $connection.set({ mode: 'local', connectionId: 'ambient-other', baseUrl: 'http://127.0.0.1:1' } as never)
    $sessionTiles.set([
      { storedSessionId: 'chat-a', runtimeId: 'runtime-a', ownerRoute: owner },
      { storedSessionId: 'chat-b', runtimeId: 'runtime-b', ownerRoute: owner }
    ])
    $layoutTree.set(
      split('row', [
        group(['workspace', 'session-tile:chat-a'], { active: 'session-tile:chat-a', id: 'chat-a-group' }),
        group(['session-tile:chat-b'], { active: 'session-tile:chat-b', id: 'chat-b-group' }),
        group(['hermestv:pane'], { active: 'hermestv:pane', id: 'player-group' })
      ])
    )
    noteActiveTreeGroup('chat-a-group')
    window.hermesDesktop = {
      ...originalDesktop,
      player: { linkAwareness, unlinkAwareness, agentRequest }
    } as unknown as Window['hermesDesktop']
  })

  afterEach(async () => {
    const { result, unmount } = renderHook(() => usePlayerAgentBridge())
    await act(async () => {
      await result.current.unlink()
    })
    unmount()
    cleanup()
    window.hermesDesktop = originalDesktop
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
    $sessionTiles.set([])
    $connection.set(null)
  })

  it('links a real tile runtime and retains its exact owner while clicking the player', async () => {
    const { result } = renderHook(() => usePlayerAgentBridge())
    act(() => noteActiveTreeGroup('player-group'))
    expect(result.current.scope).toEqual({
      connectionId: owner.connectionId,
      profileId: owner.profile,
      sessionId: 'runtime-a'
    })
    await act(async () => {
      await result.current.link()
    })
    expect(linkAwareness).toHaveBeenCalledWith({
      connectionId: owner.connectionId,
      profileId: owner.profile,
      sessionId: 'runtime-a'
    })
    expect(result.current.linked).toBe(true)
    await result.current.request({ type: 'context' })
    expect(agentRequest).toHaveBeenCalledWith({ scope: result.current.scope, operation: { type: 'context' } })
  })

  it('never reuses another chat link or invents an owner for an unresolved tile', async () => {
    const { result } = renderHook(() => usePlayerAgentBridge())
    await act(async () => {
      await result.current.link()
    })
    act(() => noteActiveTreeGroup('chat-b-group'))
    expect($focusedRuntimeId.get()).toBe('runtime-b')
    await waitFor(() => expect(result.current.linked).toBe(false))
    await expect(result.current.request({ type: 'context' })).rejects.toThrow('Link the active Hermes session')
    act(() => $sessionTiles.set([{ storedSessionId: 'chat-b', runtimeId: 'unknown-runtime' }]))
    await waitFor(() => expect(result.current.scope).toBeNull())
    await expect(result.current.link()).rejects.toThrow('No active Hermes session')
    expect(linkAwareness).toHaveBeenCalledTimes(1)

    // A URL-only legacy PRIMARY has no registry owner. It still uses the
    // transport's legacy local identity, without granting this to unknown tiles.
    act(() => {
      $connection.set({ mode: 'remote', baseUrl: 'http://127.0.0.1:43177' } as never)
      $activeSessionId.set('legacy-primary')
      noteActiveTreeGroup('chat-a-group')
      $layoutTree.set(group(['workspace'], { active: 'workspace', id: 'chat-a-group' }))
    })
    await waitFor(() =>
      expect(result.current.scope).toEqual({ connectionId: 'local', profileId: 'default', sessionId: 'legacy-primary' })
    )
  })
})
