import { afterEach, describe, expect, it } from 'vitest'

import { selectAutoSpeakReply } from './auto-speak-reply'
import { clearSpokenRepliesForTests, markAssistantIdSpoken } from './spoken-reply'

const user = (id: string, text = 'go') => ({ id, role: 'user' as const, text })
const assistant = (
  id: string,
  text: string,
  extra: { hidden?: boolean; interim?: boolean; pending?: boolean } = {}
) => ({ id, role: 'assistant' as const, text, ...extra })

afterEach(() => {
  clearSpokenRepliesForTests()
})

describe('selectAutoSpeakReply', () => {
  it('speaks a newly completed assistant reply', () => {
    const messages = [user('u1'), assistant('assistant-stream-1', 'Hallo Holger.')]

    expect(selectAutoSpeakReply('s', messages)).toEqual({
      id: 'assistant-stream-1',
      pending: false,
      text: 'Hallo Holger.'
    })
  })

  it('does not re-speak after the live row is rewritten under a durable id', () => {
    const live = [user('u1'), assistant('assistant-stream-1', 'Hallo Holger.')]
    markAssistantIdSpoken('s', live, 'assistant-stream-1')

    const rewritten = [user('u1'), assistant('42', 'Hallo Holger.')]

    expect(selectAutoSpeakReply('s', rewritten)).toBeNull()
  })

  it('does not re-speak an appended copy whose wording drifted', () => {
    const live = [user('u1'), assistant('assistant-stream-1', 'Ja, Updates kannst du machen.')]
    markAssistantIdSpoken('s', live, 'assistant-stream-1')

    const drifted = [
      user('u1'),
      assistant('assistant-stream-1', 'Ja, Updates kannst du machen.'),
      assistant('42', 'Ja, Updates kannst du machen.\\n\\nBei dir gilt stash.')
    ]

    expect(selectAutoSpeakReply('s', drifted)).toBeNull()
  })

  it('does not re-speak when the durable copy is appended beside the live row', () => {
    const live = [user('u1'), assistant('assistant-stream-1', 'Hallo Holger.')]
    markAssistantIdSpoken('s', live, 'assistant-stream-1')

    // Hydrate/history sync can briefly keep the live tail AND append the
    // committed row — same answer, new id. Auto-speak used to treat that as
    // a new turn: first reading, then "Preparing audio", then the same text.
    const both = [
      user('u1'),
      assistant('assistant-stream-1', 'Hallo Holger.'),
      assistant('42', 'Hallo Holger.')
    ]

    expect(selectAutoSpeakReply('s', both)).toBeNull()
  })

  it('does not speak sealed interim narration', () => {
    const messages = [
      user('u1'),
      assistant('assistant-stream-1', 'Ich schaue kurz nach.', { interim: true })
    ]

    expect(selectAutoSpeakReply('s', messages)).toBeNull()
  })

  it('speaks the final answer after a different interim bubble', () => {
    const messages = [
      user('u1'),
      assistant('assistant-stream-1', 'Ich schaue kurz nach.', { interim: true }),
      assistant('assistant-stream-2', 'Fertig. Hier ist die Antwort.')
    ]

    expect(selectAutoSpeakReply('s', messages)).toEqual({
      id: 'assistant-stream-2',
      pending: false,
      text: 'Fertig. Hier ist die Antwort.'
    })
  })

  it('still speaks a later turn that happens to say the same thing', () => {
    const first = [user('u1'), assistant('a1', 'Done.')]
    markAssistantIdSpoken('s', first, 'a1')

    const second = [user('u1'), assistant('a1', 'Done.'), user('u2', 'nochmal'), assistant('a2', 'Done.')]

    expect(selectAutoSpeakReply('s', second)).toEqual({
      id: 'a2',
      pending: false,
      text: 'Done.'
    })
  })

  it('reports pending so the hook waits instead of speaking a partial', () => {
    const messages = [user('u1'), assistant('assistant-stream-1', 'Hal', { pending: true })]

    expect(selectAutoSpeakReply('s', messages)).toEqual({
      id: 'assistant-stream-1',
      pending: true,
      text: 'Hal'
    })
  })
})
