// The transcript's activity row has to account for every second the app spends
// claiming to work. Before this, it ran off a text-length signature and only
// deferred to the tail part, so it went silent — and stopped counting — in the
// gaps: between a tool result landing and the next call arriving, and while a
// hoisted tool that renders nothing was in flight.
import { describe, expect, it } from 'vitest'

import { activitySignature, toolNarratesWait } from './turn-activity'

const text = (value: string) => ({ text: value, type: 'text' })

const call = (toolName: string, settled: boolean) => ({
  toolName,
  type: 'tool-call',
  ...(settled ? { result: 'ok' } : {})
})

describe('activitySignature', () => {
  it('changes when a tool call lands its result', () => {
    // The result MUTATES the existing part: same part count, same text. A
    // signature blind to it reads the finished call as more of the same
    // silence, so the gap after it gets dated from when the call started.
    expect(activitySignature([text('working'), call('terminal', false)])).not.toBe(
      activitySignature([text('working'), call('terminal', true)])
    )
  })

  it('changes when prose streams and when a part is appended', () => {
    const base = activitySignature([text('wor')])

    expect(activitySignature([text('working')])).not.toBe(base)
    expect(activitySignature([text('wor'), call('read_file', false)])).not.toBe(base)
  })

  it('is stable while nothing visible happens', () => {
    const parts = [text('working'), call('terminal', true)]

    expect(activitySignature(parts)).toBe(activitySignature([...parts]))
  })
})

describe('toolNarratesWait', () => {
  it('defers to a tool call in flight — it has its own row and timer', () => {
    expect(toolNarratesWait([text('working'), call('terminal', false)])).toBe(true)
  })

  it('does not defer to a settled call: the gap after it belongs to nobody', () => {
    expect(toolNarratesWait([text('working'), call('terminal', true)])).toBe(false)
  })

  it('does not defer to silent tools, which render nothing to narrate with', () => {
    expect(toolNarratesWait([call('todo', false)])).toBe(false)
    expect(toolNarratesWait([call('react_to_message', false)])).toBe(false)
  })

  it('defers to a call in flight even when a later part follows it', () => {
    // Tail-part-only detection missed this: a turn that starts prose while a
    // call is still running showed the row under the tool's own timer.
    expect(toolNarratesWait([call('terminal', false), text('meanwhile')])).toBe(true)
  })
})
