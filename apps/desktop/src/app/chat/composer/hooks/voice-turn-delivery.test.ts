import { describe, expect, it, vi } from 'vitest'

import { deliverVoiceTurn } from './voice-turn-delivery'

// A voice turn used to be dropped on the floor when the agent was busy
// (`if (busy) return`) or when the submit path refused it (returns false while
// the session is busy). Words the user said must never vanish: queue them
// behind the running turn, or — with no queue — park them in the composer.
describe('deliverVoiceTurn', () => {
  const base = () => ({
    enqueue: vi.fn(() => ({ id: 'q1' })),
    insertText: vi.fn(),
    onSubmit: vi.fn(async () => true),
    queueKey: 'session-1'
  })

  it('submits when the agent is idle', async () => {
    const args = base()

    expect(await deliverVoiceTurn({ ...args, busy: false, text: 'hello' })).toBe('submitted')
    expect(args.onSubmit).toHaveBeenCalledWith('hello')
    expect(args.enqueue).not.toHaveBeenCalled()
  })

  it('queues instead of dropping when the agent is busy', async () => {
    const args = base()

    expect(await deliverVoiceTurn({ ...args, busy: true, text: 'also this' })).toBe('queued')
    expect(args.onSubmit).not.toHaveBeenCalled()
    expect(args.enqueue).toHaveBeenCalledWith('session-1', { attachments: [], text: 'also this' })
  })

  it('queues when the submit path refuses (busy edge the prop had not seen)', async () => {
    const args = { ...base(), onSubmit: vi.fn(async () => false) }

    expect(await deliverVoiceTurn({ ...args, busy: false, text: 'late words' })).toBe('queued')
    expect(args.enqueue).toHaveBeenCalledWith('session-1', { attachments: [], text: 'late words' })
  })

  it('parks the words in the composer when there is no queue yet', async () => {
    const args = { ...base(), queueKey: null }

    expect(await deliverVoiceTurn({ ...args, busy: true, text: 'first words' })).toBe('drafted')
    expect(args.insertText).toHaveBeenCalledWith('first words')
  })
})
