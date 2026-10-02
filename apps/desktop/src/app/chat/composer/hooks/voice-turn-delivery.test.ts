import { describe, expect, it, vi } from 'vitest'

import { deliverVoiceTurn, reclaimVoiceAsides } from './voice-turn-delivery'

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

// "Stop" means the user wants the floor. Asides queued by voice while the
// agent worked must not sit (parked) ahead of what they say next, nor auto-send
// and restart the agent — move them into the composer: kept, but not sent.
describe('reclaimVoiceAsides', () => {
  it('moves voice-queued entries out of the queue and into the composer', () => {
    const queue = [
      { id: 'typed', text: 'typed earlier' },
      { id: 'v1', text: 'also the logs' },
      { id: 'v2', text: 'and the date' }
    ]

    const remove = vi.fn((_key: string, id: string) => {
      const i = queue.findIndex(e => e.id === id)

      return i >= 0 && queue.splice(i, 1).length > 0
    })

    const insertText = vi.fn()

    reclaimVoiceAsides({ getQueued: () => queue, ids: ['v1', 'v2'], insertText, key: 's1', remove })

    expect(queue.map(e => e.id)).toEqual(['typed'])
    expect(insertText).toHaveBeenCalledWith('also the logs and the date')
  })

  it('skips asides that already sent', () => {
    const insertText = vi.fn()

    reclaimVoiceAsides({ getQueued: () => [], ids: ['gone'], insertText, key: 's1', remove: vi.fn(() => false) })

    expect(insertText).not.toHaveBeenCalled()
  })
})
