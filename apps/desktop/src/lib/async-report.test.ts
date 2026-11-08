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

describe('cron result note hydration (#148449)', () => {
  const card = "Cron job 'News radar' finished"
  const body = '3 takes are ready.\n\nMEDIA:/tmp/News Radar/animation.gif'

  const note = (content: string, display_metadata: Record<string, unknown>) =>
    toRuntimeMessage(toChatMessages([{ role: 'user', content, display_kind: 'internal_notification', display_metadata }])[0])

  it('paints a cron note as its card with the job output as the report, media lines intact', () => {
    const wrapped = `Cronjob Response: News radar\n(job_id: 2f64841b4ac0)\n-------------\n\n${body}\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder News radar").`

    for (const [content, report] of [
      [wrapped, `${body}\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder News radar").`],
      [body, body] // cron.wrap_response: false
    ]) {
      const message = note(content, { display_text: card, completion_note: ['["cron","k","1"]'] })
      expect(message.role).toBe('system')
      expect(message.content).toEqual([{ type: 'text', text: card }])
      expect(message.metadata.custom.asyncResult).toBe(report)
    }
  })

  it('leaves an internal notification without a card as the row it was', () => {
    const message = note('[IMPORTANT: Background process proc_1 matched a watch pattern]', {})
    expect(message.role).toBe('user')
    expect(message.metadata.custom.asyncResult).toBeUndefined()
  })
})
