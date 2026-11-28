import { cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { dropSessionState, publishSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'

// A slash command typed in a split-view session TILE runs against the tile's own
// runtime (the tile passes it as `sessionId`). The notice it appends to that
// runtime must not carry the MAIN pane's selected stored id: the session-state
// cache reads a different stored id as a compression rotation and re-keys the
// tile onto the main chat, which rebuilds the split layout and retitles the tab.

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

const MAIN_RUNTIME = 'rt-main'
const MAIN_STORED = 'stored-main'
const TILE_RUNTIME = 'rt-tile'
const TILE_STORED = 'stored-tile'

afterEach(() => {
  cleanup()
  dropSessionState(TILE_RUNTIME)
})

async function mountWithTile(updates: Array<{ sessionId: string; storedSessionId: null | string | undefined }>) {
  publishSessionState(TILE_RUNTIME, createClientSessionState(TILE_STORED))

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) =>
    (method === 'config.set' && params?.key === 'yolo' ? { key: 'yolo', value: String(params.value) } : {}) as never
  )

  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      activeSessionId={MAIN_RUNTIME}
      onReady={h => (handle = h)}
      onUpdateState={(sessionId, storedSessionId) => updates.push({ sessionId, storedSessionId })}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
      storedSessionId={MAIN_STORED}
    />
  )

  return { handle: handle!, requestGateway }
}

describe('slash notice appended to a tile runtime', () => {
  it('/yolo from a tile toggles the tile session and never relabels it with the main chat id', async () => {
    const updates: Array<{ sessionId: string; storedSessionId: null | string | undefined }> = []
    const { handle, requestGateway } = await mountWithTile(updates)

    await handle.submitText('/yolo', { sessionId: TILE_RUNTIME })

    const toggles = requestGateway.mock.calls.filter(([method]) => method === 'config.set')
    expect(toggles).toEqual([['config.set', { key: 'yolo', session_id: TILE_RUNTIME, value: '1' }]])

    const tileUpdates = updates.filter(update => update.sessionId === TILE_RUNTIME)
    expect(tileUpdates.length).toBeGreaterThan(0)
    expect(tileUpdates.map(update => update.storedSessionId)).not.toContain(MAIN_STORED)
  })

  it('/yolo in the main pane still binds the notice to the selected main conversation', async () => {
    const updates: Array<{ sessionId: string; storedSessionId: null | string | undefined }> = []
    const { handle } = await mountWithTile(updates)

    await handle.submitText('/yolo')

    const mainUpdates = updates.filter(update => update.sessionId === MAIN_RUNTIME)
    expect(mainUpdates.length).toBeGreaterThan(0)
    expect(mainUpdates.every(update => update.storedSessionId === MAIN_STORED)).toBe(true)
  })
})
