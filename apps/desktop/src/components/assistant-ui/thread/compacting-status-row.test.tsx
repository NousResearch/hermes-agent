// A manual /compress runs with no live turn, so every busy-gated row stays
// silent and the compression was invisible in the transcript. These contracts
// pin the thread-level row that claims the whole phase — and its two
// deferrals: to the busy chrome mid-turn, and to a running tail's own loading
// row so the compaction is never narrated twice (#68634 pins one row).
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { setSessionCompacting, setSessionCompactionProgress } from '@/store/compaction'

import { CompactingStatusRow } from './status'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function anchorMessage(): ThreadMessage {
  return { id: 'user-1', role: 'user', content: [{ type: 'text', text: 'hi' }], createdAt, metadata: {} } as unknown as ThreadMessage
}

function renderRow({ isRunning = false, ownerBusy = false }: { isRunning?: boolean; ownerBusy?: boolean } = {}) {
  const $ownerBusy = atom(ownerBusy)

  function Harness() {
    const runtime = useExternalStoreRuntime<ThreadMessage>({ messages: [anchorMessage()], isRunning, onNew: async () => {} })
    const view = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom<string | null>('owner'), $busy: $ownerBusy }

    return (
      <AssistantRuntimeProvider runtime={runtime}>
        <SessionViewProvider value={view}>
          <CompactingStatusRow />
        </SessionViewProvider>
      </AssistantRuntimeProvider>
    )
  }

  return { $ownerBusy, ...render(<Harness />) }
}

afterEach(() => {
  cleanup()
  setSessionCompacting('owner', false)
})

it('claims the transcript for a standalone compression — the case no busy-gated row covers', () => {
  setSessionCompacting('owner', true)

  const { container } = renderRow()

  const row = container.querySelector('[data-slot="aui_compacting"]')
  expect(row).toBeTruthy()
  expect(row?.getAttribute('aria-label')).toBe('Summarizing thread')
})

it('prefers the gateway progress text over the static label', () => {
  setSessionCompacting('owner', true)
  setSessionCompactionProgress('owner', 'compressing 42 messages (~120,000 tok)…')

  const { container } = renderRow()

  const row = container.querySelector('[data-slot="aui_compacting"]')
  expect(row?.getAttribute('aria-label')).toBe('compressing 42 messages (~120,000 tok)…')
  expect(container.textContent).toContain('compressing 42 messages')
})

it('defers to the busy chrome mid-turn and disappears when the compaction ends', () => {
  setSessionCompacting('owner', true)

  const { container, $ownerBusy } = renderRow({ ownerBusy: true })
  expect(container.querySelector('[data-slot="aui_compacting"]')).toBeNull()

  act(() => $ownerBusy.set(false))
  expect(container.querySelector('[data-slot="aui_compacting"]')).toBeTruthy()

  act(() => setSessionCompacting('owner', false))
  expect(container.querySelector('[data-slot="aui_compacting"]')).toBeNull()
})

it('defers to a running tail\'s own loading row so the wait is narrated once', () => {
  setSessionCompacting('owner', true)

  const { container } = renderRow({ isRunning: true })

  expect(container.querySelector('[data-slot="aui_compacting"]')).toBeNull()
})

it('stays silent for another session\'s compaction', () => {
  setSessionCompacting('someone-else', true)

  const { container } = renderRow()

  expect(container.querySelector('[data-slot="aui_compacting"]')).toBeNull()
})
