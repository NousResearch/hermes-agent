/**
 * What the causal trail actually renders.
 *
 * The walk is covered in provenance.test.ts; this is the presentation
 * contract — that a chain reads as "who, what they read, who started it", and
 * that a room without recorded causes says so instead of drawing a stub.
 */

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProvenanceTrail } from './provenance-trail'
import type { GroupChat, GroupMessage } from './types'

afterEach(cleanup)

function user(id: string, text: string): GroupMessage {
  return { id, at: 0, from: { kind: 'user', name: 'You' }, text, thread: 'main' }
}

function member(id: string, name: string, text: string, by?: string, saw?: [number, number]): GroupMessage {
  return {
    id,
    at: 0,
    from: { kind: 'member', name },
    text,
    thread: 'main',
    ...(by || saw ? { cause: { ...(by ? { by } : {}), ...(saw ? { saw } : {}) } } : {})
  }
}

const room: GroupChat = {
  log: [
    user('m0', 'summarize the incident'),
    member('m1', 'radar', 'found three candidates in the gateway logs', 'You', [0, 1]),
    member('m2', 'scribe', 'written up, ranked by blast radius', 'radar', [1, 2])
  ],
  watermarks: {}
}

const labelFor = (name: string) => (name === 'radar' ? 'Radar' : name === 'scribe' ? 'Scribe' : name)

describe('ProvenanceTrail', () => {
  it('renders one row per hop, newest first, ending at the user who started it', () => {
    const { container } = render(<ProvenanceTrail labelFor={labelFor} messageId="m2" room={room} />)
    const rows = container.querySelectorAll('li')

    expect(rows).toHaveLength(3)
    expect(rows[0].textContent).toContain('Scribe')
    expect(rows[1].textContent).toContain('Radar')
    expect(rows[2].textContent).toContain('You')
    expect(rows[2].textContent).toContain('started this')
  })

  it('reports how much each turn read, pluralized', () => {
    render(<ProvenanceTrail labelFor={labelFor} messageId="m2" room={room} />)

    expect(screen.getAllByText('read 1 message')).toHaveLength(2)
    expect(screen.queryByText('read 1 messages')).toBeNull()
  })

  it('resolves display names through the injected label callback', () => {
    // The trail must never import the roster — private agent names stay out of
    // this reusable primitive.
    render(<ProvenanceTrail labelFor={labelFor} messageId="m2" room={room} />)

    expect(screen.getByText('Radar')).toBeTruthy()
    expect(screen.queryByText('radar')).toBeNull()
  })

  it('says so plainly when a room predates cause stamping', () => {
    const legacy: GroupChat = { log: [user('m0', 'go'), member('m1', 'radar', 'reply')], watermarks: {} }

    render(<ProvenanceTrail labelFor={labelFor} messageId="m1" room={legacy} />)

    expect(screen.getByText(/No recorded cause/)).toBeTruthy()
  })

  it('shows the empty state for a user send, which has no cause by definition', () => {
    render(<ProvenanceTrail labelFor={labelFor} messageId="m0" room={room} />)

    expect(screen.getByText(/No recorded cause/)).toBeTruthy()
  })

  it('truncates a long line rather than letting it widen the row', () => {
    const long = 'x'.repeat(400)

    const wide: GroupChat = {
      log: [user('m0', 'go'), member('m1', 'radar', long, 'You', [0, 1])],
      watermarks: {}
    }

    render(<ProvenanceTrail labelFor={labelFor} messageId="m1" room={wide} />)

    expect(screen.getByText(/^x+…$/).textContent!.length).toBeLessThanOrEqual(72)
  })
})
