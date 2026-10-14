import { describe, expect, it } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages/hydration'
import { toRuntimeMessage } from '@/lib/chat-runtime'

const report = '# Report\n\n**Conclusion:** ready\n\n| Task | Status |\n|---|---|\n| probe | ok |'

const envelope = (body: string) =>
  `[ASYNC DELEGATION COMPLETE — probe]\nPrivate instructions\nOriginal goal: private goal\nRole: leaf\n--- RESULT ---\n${body}`

function hydrate(content: string) {
  return toRuntimeMessage(
    toChatMessages([
      {
        role: 'user',
        content,
        display_kind: 'async_delegation_complete',
        display_metadata: { delegation_id: 'probe', task_count: 1 }
      }
    ])[0]
  )
}

describe('async report hydration', () => {
  it('keeps only result bodies beside compact system metadata across current and legacy deliveries', () => {
    for (const input of [
      envelope(report),
      envelope(
        `Cron job 'probe' (id) finished its manual run.\nResult: ok\nDelivery target: local\n--- JOB OUTPUT ---\n${report}`
      ),
      report
    ]) {
      const message = hydrate(input)
      expect(message.role).toBe('system')
      expect(message.content).toEqual([{ type: 'text', text: '1 background agent finished' }])
      expect(message.metadata.custom.asyncResult).toBe(report)
    }
  })

  it('separates batch result blocks without leaking preambles or transcript plumbing', () => {
    const content = `[ASYNC DELEGATION BATCH COMPLETE — batch]\nPrivate instructions\nRole: leaf\n\n--- ✓ TASK 1/2: private goal  (status=completed) ---\n${report}\nFull live transcript (complete tool/assistant trace): /private/path\n\n--- ✓ TASK 2/2: private goal\nprivate continuation  (status=completed) ---\nPlain result`
    expect(hydrate(content).metadata.custom.asyncResult).toBe(`${report}\n\nPlain result`)
    expect(
      hydrate('[ASYNC DELEGATION COMPLETE — malformed]\nPrivate instructions').metadata.custom.asyncResult
    ).toBeUndefined()
    expect(hydrate('').metadata.custom.asyncResult).toBeUndefined()
  })
})
