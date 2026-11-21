// Regression test for #121096: switching sessions replaces thread messages in
// place (same indices, new ids). ResponseMessages keyed assistant rows by array
// index, so React reused a row instance across different logical messages, and
// the row's cached parts (PartByIndexProvider lastPartRef) leaked the previous
// message's blocks into the new one — duplicated progress blocks, scrambled
// part order. Rows must be keyed by message id so a replacement remounts.
import type * as AssistantUi from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const auiState = vi.hoisted(() => ({
  thread: { messages: [] as Array<{ id: string; role: string; content: string }> }
}))

const mountSeq = vi.hoisted(() => ({ current: 0 }))

vi.mock('@assistant-ui/react', async importOriginal => {
  const actual = await importOriginal<typeof AssistantUi>()
  const { useRef: useRowRef } = await import('react')

  // Mimics the PartByIndexProvider cache at the heart of #121096: a row
  // snapshots the parts of the frame it mounts with and keeps serving them for
  // as long as the instance lives. Only a remount (key change) picks up the
  // replacement message's parts.
  function MockMessageByIndex({ index }: { index: number }) {
    const message = auiState.thread.messages[index]
    const cached = useRowRef<{ id: string; text: string } | null>(null)

    if (cached.current === null && message) {
      cached.current = { id: message.id, text: message.content }
    }

    const seq = useRowRef(0)

    if (seq.current === 0) {
      mountSeq.current += 1
      seq.current = mountSeq.current
    }

    return (
      <div
        data-cached-text={cached.current?.text ?? ''}
        data-index={index}
        data-message-id={cached.current?.id ?? ''}
        data-mount-seq={seq.current}
        data-testid="mock-row"
      />
    )
  }

  return {
    ...actual,
    useAuiState: <T,>(selector: (state: typeof auiState) => T): T => selector(auiState),
    ThreadPrimitive: { ...actual.ThreadPrimitive, MessageByIndex: MockMessageByIndex }
  }
})

const { ResponseMessages } = await import('./response-group')

interface Row {
  index: string
  messageId: string
  cachedText: string
  mountSeq: string
}

const rows = (): Row[] =>
  screen.getAllByTestId('mock-row').map(node => ({
    index: node.dataset.index ?? '',
    messageId: node.dataset.messageId ?? '',
    cachedText: node.dataset.cachedText ?? '',
    mountSeq: node.dataset.mountSeq ?? ''
  }))

const assistant = (id: string, text: string) => ({ id, role: 'assistant', content: text })

function mountRows(indices: number[]) {
  let bump!: () => void

  function Harness() {
    const [, setVersion] = useState(0)
    bump = () => act(() => setVersion(version => version + 1))

    return <ResponseMessages components={{} as never} indices={indices} />
  }

  const view = render(<Harness />)

  return { view, rerender: () => bump() }
}

afterEach(() => {
  cleanup()
  auiState.thread.messages = []
})

describe('ResponseMessages row keys (#121096)', () => {
  it('remounts a replaced row inside a surviving section, so cached parts cannot leak', () => {
    // The section container is keyed by its first row's id, so a store
    // replacement (session switch, history backfill, compaction) that keeps
    // the first message but swaps a later one reuses the section — and with
    // index keys, reused the swapped row's instance too.
    auiState.thread.messages = [assistant('msg-a', 'alpha'), assistant('msg-b', 'beta')]

    const { view, rerender } = mountRows([0, 1])

    expect(rows()).toEqual([
      { index: '0', messageId: 'msg-a', cachedText: 'alpha', mountSeq: expect.any(String) },
      { index: '1', messageId: 'msg-b', cachedText: 'beta', mountSeq: expect.any(String) }
    ])
    const firstFrameSeq = rows().map(row => row.mountSeq)

    // The store replaces the second message in place; the section survives.
    auiState.thread.messages = [assistant('msg-a', 'alpha'), assistant('msg-d', 'delta')]
    rerender()

    const replaced = rows()

    // Old content is gone and each row reflects its new logical message.
    expect(replaced.map(row => [row.index, row.messageId, row.cachedText])).toEqual([
      ['0', 'msg-a', 'alpha'],
      ['1', 'msg-d', 'delta']
    ])

    // The untouched row keeps its instance; the replaced row is a fresh one.
    expect(replaced[0]!.mountSeq).toBe(firstFrameSeq[0])
    expect(replaced[1]!.mountSeq).not.toBe(firstFrameSeq[1])

    view.unmount()
  })

  it('keeps row instances across content flushes of the same logical message', () => {
    auiState.thread.messages = [assistant('msg-a', 'alpha')]

    const { view, rerender } = mountRows([0])

    const before = rows()
    expect(before).toEqual([{ index: '0', messageId: 'msg-a', cachedText: 'alpha', mountSeq: expect.any(String) }])

    // Same message id (streaming token flush): the row instance must survive,
    // same as the index key behaved.
    auiState.thread.messages = [assistant('msg-a', 'alpha beta')]
    rerender()

    const after = rows()

    expect(after.map(row => [row.index, row.messageId, row.mountSeq])).toEqual([['0', 'msg-a', before[0]!.mountSeq]])

    view.unmount()
  })
})
