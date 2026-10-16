import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { makeSessionInfo } from '@/test/session-info'

import { $backgroundStatusBySession } from './composer-status'
import { $activeSessionId, $sessions } from './session'
import {
  $sessionStates,
  $sessionTiles,
  clearAllSessionStates,
  closeSessionTile,
  publishSessionState
} from './session-states'
import { $sidebarActivityById } from './sidebar-activity'
import { $subagentsBySession } from './subagents'
import { $todosBySession } from './todos'

const watching = {
  type: 'tool-call' as const,
  toolName: 'terminal',
  toolCallId: 'watch',
  args: { command: 'gh pr checks --watch', context: 'Watching CI' }
}

function runningState(storedId = 'stored') {
  return {
    ...createClientSessionState(storedId),
    busy: true,
    turnLive: true,
    messages: [{ id: 'reply', role: 'assistant' as const, pending: true, parts: [watching] }]
  }
}

beforeEach(() => {
  clearAllSessionStates()
  $activeSessionId.set(null)
  $sessionTiles.set([])
  $sessions.set([])
  $backgroundStatusBySession.set({})
  $subagentsBySession.set({})
  $todosBySession.set({})
})
afterEach(() => {
  clearAllSessionStates()
  $sessionTiles.set([])
  $backgroundStatusBySession.set({})
  $subagentsBySession.set({})
  $todosBySession.set({})
  $sessions.set([])
})

describe('sidebar activity', () => {
  it('does not project active tools as monitoring', () => {
    publishSessionState('runtime', runningState())
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('does not mistake an old unresolved tool for current work', () => {
    publishSessionState('runtime', { ...runningState(), busy: false, turnLive: false })
    expect($sidebarActivityById.get().stored).toBeUndefined()
    publishSessionState('runtime', {
      ...runningState(),
      messages: [
        { id: 'old', role: 'assistant', parts: [watching] },
        { id: 'new-request', role: 'user', parts: [{ type: 'text', text: 'Next task' }] }
      ]
    })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('retires a completed tool immediately, including an empty result', () => {
    publishSessionState('runtime', {
      ...runningState(),
      messages: [{ id: 'reply', role: 'assistant', pending: true, parts: [{ ...watching, result: '' }] }]
    })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('does not keep an interrupted tool alive from its pending part', () => {
    publishSessionState('runtime', { ...runningState(), interrupted: true })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('does not treat active delegation or leftover todos as monitoring', () => {
    publishSessionState('runtime', createClientSessionState('stored'))

    const child = {
      id: 'child',
      parentId: null,
      goal: 'Watching CI',
      status: 'running' as const,
      taskCount: 1,
      taskIndex: 0,
      startedAt: 1,
      updatedAt: 1,
      filesRead: [],
      filesWritten: [],
      stream: []
    }

    $subagentsBySession.set({ runtime: [child] })
    $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime' }])
    closeSessionTile('stored')
    expect($sidebarActivityById.get().stored).toBeUndefined()
    $subagentsBySession.set({ runtime: [{ ...child, status: 'completed' }] })
    $todosBySession.set({ runtime: [{ id: 'todo', content: 'Watch CI', status: 'in_progress' }] })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('keeps real background work after the turn settles and drops it on exit', () => {
    const process = {
      type: 'background' as const,
      state: 'running' as const,
      title: 'Watching CI',
      id: 'proc',
      awaitingNotification: true
    }

    publishSessionState('runtime', createClientSessionState('stored'))
    $backgroundStatusBySession.set({ runtime: [process] })
    $sessionTiles.set([{ storedSessionId: 'stored', runtimeId: 'runtime' }])
    closeSessionTile('stored')
    expect($sessionStates.get().runtime?.messages).toEqual([])
    expect($sidebarActivityById.get().stored).toBe(process)
    $backgroundStatusBySession.set({ runtime: [{ ...process, state: 'done' }] })
    expect($sidebarActivityById.get().stored).toBeUndefined()
    $backgroundStatusBySession.set({ runtime: [{ ...process, state: 'failed' }] })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('suppresses monitoring while the parent is working', () => {
    publishSessionState('runtime', runningState())
    $backgroundStatusBySession.set({
      runtime: [
        { type: 'background', state: 'running', title: 'Watching checks', id: 'proc', awaitingNotification: true }
      ]
    })
    expect($sidebarActivityById.get().stored).toBeUndefined()
  })

  it('resolves compression aliases and draft runtime ids', () => {
    $sessions.set([makeSessionInfo({ id: 'tip', _lineage_root_id: 'stored' })])

    const process = {
      type: 'background' as const,
      state: 'running' as const,
      title: 'Watching checks',
      id: 'proc',
      awaitingNotification: true
    }

    publishSessionState('runtime', createClientSessionState('stored'))
    $backgroundStatusBySession.set({ runtime: [process], draft: [process] })
    expect($sidebarActivityById.get().tip).toBe(process)
    publishSessionState('draft', { ...createClientSessionState(), storedSessionId: null })
    expect($sidebarActivityById.get().draft).toBe(process)
  })

  it('keeps the map reference stable across prose deltas and unrelated idle writes', () => {
    const idle = { ...runningState(), busy: false, turnLive: false }
    publishSessionState('runtime', idle)
    $backgroundStatusBySession.set({
      runtime: [
        { type: 'background', state: 'running', title: 'Watching checks', id: 'proc', awaitingNotification: true }
      ]
    })
    const previous = $sidebarActivityById.get()
    publishSessionState('other', createClientSessionState('other-stored'))
    expect($sidebarActivityById.get()).toBe(previous)
    publishSessionState('runtime', {
      ...idle,
      messages: [
        { id: 'reply', role: 'assistant', pending: true, parts: [watching, { type: 'text', text: 'Progress' }] }
      ]
    })
    expect($sidebarActivityById.get()).toBe(previous)
  })
})
