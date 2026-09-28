import { readFileSync } from 'node:fs'

import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { parseAnsi } from '@/lib/ansi'
import { type ChatMessage, type GatewayEventPayload, toChatMessages, upsertToolPart } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { getToolDiff } from '@/store/tool-diffs'
import { $toolDisclosureStates, setHideCodeDiffs, setToolViewMode } from '@/store/tool-view'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

interface Receipt {
  rows: SessionMessage[]
  projected: SessionMessage[]
  completion: GatewayEventPayload
  changed: boolean
}

// A self-contained wire-contract fixture. Optionally run this same invariant
// against receipts produced by test_inline_edit_persistence.py's real tools/DB.
const diff = "--- a/publish.py\n+++ b/publish.py\n@@ -1 +1 @@\n-print('before')\n+print('after')"
const result = { success: true, path: 'publish.py', verified: true }
const metadata = { tool_result_metadata: { inline_diff: diff } }

const tool: SessionMessage = {
  role: 'tool',
  name: 'write_file',
  tool_call_id: 'edit-call',
  timestamp: 20,
  content: JSON.stringify(result),
  display_metadata: metadata
}

const fixtures: Record<string, Receipt> = {
  write: {
    rows: [
      { role: 'user', content: 'Update the file' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: tool.tool_call_id,
            type: 'function',
            function: { name: tool.name, arguments: '{"path":"publish.py"}' }
          }
        ]
      },
      tool
    ],
    projected: [{ ...tool, args: { path: 'publish.py' }, context: 'publish.py' }],
    completion: {
      name: tool.name,
      tool_id: tool.tool_call_id!,
      args: { path: 'publish.py' },
      result,
      inline_diff: diff
    },
    changed: true
  }
}

for (const [name, raw] of Object.entries({
  legacyPatch: { ...result, diff },
  noop: result,
  failed: { error: 'Write refused' },
  falseResult: false,
  nullResult: null,
  emptyResult: ''
})) {
  const seed = fixtures.write

  fixtures[name] = {
    rows: seed.rows.map(row =>
      row.role === 'tool' ? { ...row, content: JSON.stringify(raw), display_metadata: undefined } : row
    ),
    projected: seed.projected.map(row => ({ ...row, content: JSON.stringify(raw), display_metadata: undefined })),
    completion: { ...seed.completion, result: raw, inline_diff: undefined },
    changed: name === 'legacyPatch'
  }
}

const receipts: Record<string, Receipt> = process.env.INLINE_EDIT_RECEIPTS
  ? JSON.parse(readFileSync(process.env.INLINE_EDIT_RECEIPTS, 'utf8'))
  : fixtures

beforeEach(() => {
  expect(getToolDiff('edit-call')).toBe('')
  $toolDisclosureStates.set({})
  setHideCodeDiffs(false)
  setToolViewMode('product')
})
afterEach(cleanup)

it.each(Object.entries(receipts))(
  '%s: cold tool hydration preserves raw results and edit visibility',
  async (_name, receipt) => {
    const live: ChatMessage = {
      id: 'live',
      role: 'assistant',
      pending: false,
      parts: upsertToolPart([], receipt.completion, 'complete')
    }

    const liveUi = render(
      <ThreadRuntime messages={[toRuntimeMessage(live)]}>
        <Thread />
      </ThreadRuntime>
    )

    if (receipt.changed) {
      await waitFor(() => expect(liveUi.container.querySelector('[data-tool-row][data-file-edit]')).not.toBeNull())
    }

    cleanup()

    // Pending call, already-rendered assistant call, and orphan/RPC projection.
    const withCommentary = receipt.rows.map(row =>
      row.role === 'assistant' ? { ...row, content: 'Editing now' } : row
    )

    for (const rows of [receipt.rows, withCommentary, receipt.projected]) {
      expect(getToolDiff(receipt.completion.tool_id!)).toBe('')

      for (const serialized of [false, true]) {
        const history = rows.map(row =>
          serialized && row.display_metadata ? { ...row, display_metadata: JSON.stringify(row.display_metadata) } : row
        )

        const cold = toChatMessages(history)
        const parts = cold.flatMap(message => message.parts).filter(part => part.type === 'tool-call')

        expect(parts).toHaveLength(1)
        expect(parts[0].toolCallId).toBe(receipt.completion.tool_id)
        expect(parts[0].result).toEqual(receipt.completion.result)
        expect(parts[0].toolResultMetadata?.inline_diff).toBe(receipt.completion.inline_diff)

        const ui = render(
          <ThreadRuntime messages={cold.map(toRuntimeMessage)}>
            <Thread />
          </ThreadRuntime>
        )

        await waitFor(() => {
          expect(Boolean(ui.container.querySelector('[data-tool-row][data-file-edit]'))).toBe(receipt.changed)
        })

        if (receipt.changed) {
          expect(ui.container.textContent).toContain('publish.py')
          expect(ui.container.textContent).not.toContain('unrelated later edit')

          const diffLines = receipt.completion.inline_diff ?? (receipt.completion.result as { diff: string }).diff

          const addition = parseAnsi(diffLines)
            .map(segment => segment.text)
            .join('')
            .split('\n')
            .find(line => line.includes('+print('))

          expect(addition).toBeDefined()
          expect(ui.container.textContent).toContain(addition!.slice(1))
        }

        cleanup()
      }
    }
  }
)

it.each([
  ['object', JSON.stringify({ ok: true })],
  ['false', 'false'],
  ['null', 'null'],
  ['empty-string', '""'],
  ['zero', '0']
])('does not let a completed %s result claim a later orphan with a reused id', (_name, firstContent) => {
  const rows: SessionMessage[] = [
    { role: 'user', content: 'Read first' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'reused-call',
          type: 'function',
          function: { name: 'read_file', arguments: '{"path":"read.py"}' }
        }
      ]
    },
    {
      role: 'tool',
      tool_call_id: 'reused-call',
      tool_name: 'read_file',
      content: firstContent,
      timestamp: 2
    },
    { role: 'user', content: 'Write next' },
    {
      role: 'tool',
      name: 'write_file',
      tool_call_id: 'reused-call',
      args: { path: 'write.py' },
      content: JSON.stringify({ success: true, path: 'write.py' }),
      display_metadata: { tool_result_metadata: { inline_diff: diff } },
      timestamp: 4
    }
  ]

  const parts = toChatMessages(rows)
    .flatMap(message => message.parts)
    .filter(part => part.type === 'tool-call')

  expect(parts.map(part => part.toolName)).toEqual(['read_file', 'write_file'])
  expect(parts[0].result).toEqual(JSON.parse(firstContent))
  expect(parts[1].result).toEqual({ success: true, path: 'write.py' })
  expect(parts[1].toolResultMetadata?.inline_diff).toBe(diff)
})
