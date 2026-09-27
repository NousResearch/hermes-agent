import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'
import { setActiveSessionId, setSelectedStoredSessionId } from '@/store/session'
import { $focusedRuntimeId, $sessionTiles } from '@/store/session-states'

import { handleServerRequest, type ServerRequestContext } from './server-requests'

const runPage = vi.fn(async () => JSON.stringify({ success: true, elements: [] }))
let unregister: (() => void) | undefined

const deps: ServerRequestContext['deps'] = {
  activeSessionIdRef: { current: 'primary-runtime' },
  sessionInterrupted: () => false,
  updateSessionState: vi.fn(),
  upsertToolCall: vi.fn()
}

function focusTile() {
  setActiveSessionId('primary-runtime')
  setSelectedStoredSessionId('primary-stored')
  $sessionTiles.set([{ dir: 'right', runtimeId: 'tile-runtime', storedSessionId: 'tile-stored' }])
  $layoutTree.set({
    type: 'group',
    id: 'chat-zone',
    panes: ['workspace', 'session-tile:tile-stored'],
    active: 'session-tile:tile-stored'
  })
  $activeTreeGroup.set('chat-zone')
  expect($focusedRuntimeId.get()).toBe('tile-runtime')
  $previewTabs.set([
    {
      id: 'url:test',
      target: { kind: 'url', url: 'https://example.invalid', source: 'https://example.invalid', label: 'Test page' }
    }
  ])
  $rightRailActiveTabId.set('url:test')
  unregister = registerPreviewScriptRunner('url:test', runPage)
}

async function deliver(sessionId: string) {
  const respond = vi.fn()
  handleServerRequest(
    {
      id: 'focused-preview-test',
      method: 'preview.act',
      profile: 'default',
      params: { action: 'elements', session_id: sessionId },
      respond,
      fail: vi.fn()
    },
    deps,
    'primary-runtime'
  )
  await vi.waitFor(() => expect(respond).toHaveBeenCalledOnce())

  return JSON.parse(respond.mock.calls[0][0].value)
}

afterEach(() => {
  $sessionTiles.set([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  unregister?.()
  unregister = undefined
  $previewTabs.set([])
  vi.clearAllMocks()
})

describe('focused tile preview commands', () => {
  it('follows actual chat focus, retains it in the browser, and revokes it on another chat or tile close', async () => {
    focusTile()
    expect(await deliver('tile-runtime')).toEqual({ success: true, elements: [] })
    expect(runPage).toHaveBeenCalledOnce()
    runPage.mockClear()
    expect(await deliver('primary-runtime')).toMatchObject({ success: false })
    expect(runPage).not.toHaveBeenCalled()

    $layoutTree.set({
      type: 'group',
      id: 'chat-zone',
      panes: ['workspace', 'session-tile:tile-stored', 'preview-tile:url:test'],
      active: 'preview-tile:url:test'
    })
    expect(await deliver('tile-runtime')).toEqual({ success: true, elements: [] })
    expect(runPage).toHaveBeenCalledOnce()
    runPage.mockClear()
    expect(await deliver('primary-runtime')).toMatchObject({ success: false })
    expect(runPage).not.toHaveBeenCalled()

    $layoutTree.set({
      type: 'group',
      id: 'chat-zone',
      panes: ['workspace', 'session-tile:tile-stored', 'preview-tile:url:test'],
      active: 'workspace'
    })
    $layoutTree.set({
      type: 'group',
      id: 'chat-zone',
      panes: ['workspace', 'session-tile:tile-stored', 'preview-tile:url:test'],
      active: 'preview-tile:url:test'
    })
    expect(await deliver('tile-runtime')).toMatchObject({ success: false })
    expect(runPage).not.toHaveBeenCalled()
    expect(await deliver('primary-runtime')).toMatchObject({ success: true })
    expect(runPage).toHaveBeenCalledOnce()
    runPage.mockClear()

    $layoutTree.set({
      type: 'group',
      id: 'chat-zone',
      panes: ['workspace', 'session-tile:tile-stored', 'preview-tile:url:test'],
      active: 'session-tile:tile-stored'
    })
    $layoutTree.set({
      type: 'group',
      id: 'chat-zone',
      panes: ['workspace', 'preview-tile:url:test'],
      active: 'preview-tile:url:test'
    })
    $sessionTiles.set([])
    const respond = vi.fn()
    handleServerRequest(
      {
        id: 'closed-tile',
        method: 'preview.act',
        profile: 'default',
        params: { action: 'elements', session_id: 'tile-runtime' },
        respond,
        fail: vi.fn()
      },
      deps,
      'primary-runtime'
    )
    expect(respond).not.toHaveBeenCalled()
    expect(runPage).not.toHaveBeenCalled()
  })

  it('does not grant a replayed hidden-primary request control during reconnect', async () => {
    focusTile()
    const respond = vi.fn()
    handleServerRequest(
      {
        id: 'replayed-preview-test',
        method: 'preview.act',
        profile: 'default',
        replayed: true,
        params: { action: 'elements', session_id: 'primary-runtime' },
        respond,
        fail: vi.fn()
      },
      deps,
      null
    )
    await vi.waitFor(() => expect(respond).toHaveBeenCalledOnce())
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
    expect(runPage).not.toHaveBeenCalled()
  })
})
