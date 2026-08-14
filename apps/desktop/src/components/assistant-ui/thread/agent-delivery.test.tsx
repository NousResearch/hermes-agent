import { describe, expect, it } from 'vitest'

import { deliveryTargetFromCommand, replyTextFromResult } from './agent-delivery'

// Sender-side inter-agent deliveries render as "Messaged X" / "Message from
// X" notices instead of terminal transcript rows. This pins the detection
// (the canonical Bot Mode command shape) and the reply extraction.
describe('delivery command detection', () => {
  it('matches the canonical delivery command', () => {
    const cmd =
      'hermes -p turqoise chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Hermes (@hermes): hi there"'

    expect(deliveryTargetFromCommand(cmd)).toBe('turqoise')
  })

  it('matches with a cd prefix and timeout wrapper', () => {
    const cmd =
      'cd ~ && timeout 240 hermes -p mr-tester chat --in "~" -Q -q "Message from 🤖 Hermes: hello"'

    expect(deliveryTargetFromCommand(cmd)).toBe('mr-tester')
  })

  it('ignores ordinary terminal commands', () => {
    expect(deliveryTargetFromCommand('ls -la')).toBeNull()
    expect(deliveryTargetFromCommand('hermes -p turqoise chat -q "plain question"')).toBeNull()
    expect(deliveryTargetFromCommand('hermes sessions list')).toBeNull()
  })
})

describe('reply extraction', () => {
  it('strips session_id bookkeeping and keeps the reply', () => {
    const output = 'session_id: 20260813_220347_f69ac6\nHi Hermes! Good to hear from you.'

    expect(replyTextFromResult({ output })).toBe('Hi Hermes! Good to hear from you.')
  })

  it('unwraps JSON-shaped terminal results', () => {
    const result = JSON.stringify({ exit_code: 0, output: 'session_id: abc\nack' })

    expect(replyTextFromResult(result)).toBe('ack')
  })

  it('returns empty for empty results', () => {
    expect(replyTextFromResult(undefined)).toBe('')
    expect(replyTextFromResult({ output: '' })).toBe('')
  })
})
