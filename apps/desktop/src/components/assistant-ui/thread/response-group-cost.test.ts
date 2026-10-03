import { expect, it } from 'vitest'

import { messageContentText, PROCESS_NOTIFICATION_RE } from './content'
import { responseMessageRole } from './response-group'

// Regression for #126486: the structural signature threaded through the
// transcript is `messages.map((m, i) => `${i}:${m.id}:${responseMessageRole(m)}`)`,
// re-evaluated on every assistant-ui notification (~30/s while a reply streams).
// Every user message in it costs a `responseMessageRole` call, which classified
// background deliveries by extracting the message's WHOLE text and regex-testing
// it. Settled messages cannot change classification, but nothing remembered
// that, so each flush re-copied and re-trimmed every user message's text in the
// thread: per-frame work proportional to the whole transcript's characters.
//
// These tests count the property reads that extraction costs, so the bound is
// measured rather than asserted by hand. They assert the roles are byte-identical
// to a naive transcription, because the caching is only valid if the value it
// returns is the value the scan produced.

const PARTS_PER_MESSAGE = 3
const PART_CHARS = 400
const FLUSHES = 20
/** Measured: `partText` costs 4 property reads per part (2x `type`, 1x `text`, 1x the miss path). */
const READS_PER_EXTRACTION = 4

const BACKGROUND = '[IMPORTANT: Background process proc_example completed normally (exit code 0).\nOutput:\nVerified.]'

/** Shared across every part of a message so extraction cost is attributable. */
interface Counter {
  count: number
}

const counting = (): Counter => ({ count: 0 })

/**
 * A content part that charges every `type`/`text` read to `counter`. Counting the
 * reads measures the extraction without instrumenting production code.
 */
function countedPart(text: string, counter: Counter): object {
  const target = { type: 'text', text }

  return new Proxy(target, {
    get(source, key) {
      if (key === 'type' || key === 'text') {
        counter.count += 1
      }

      return source[key as keyof typeof source]
    }
  })
}

interface BuiltMessage {
  id: string
  role: string
  content: object[]
  metadata?: { custom?: Record<string, unknown> }
}

function buildMessage(id: string, role: string, text: string, counter: Counter): BuiltMessage {
  const content: object[] = []

  for (let part = 0; part < PARTS_PER_MESSAGE; part += 1) {
    content.push(countedPart(`${text} part ${part} ${'x'.repeat(PART_CHARS)}`, counter))
  }

  return { id, role, content }
}

/** Transcription of the pre-#126486 scan, used as the behaviour oracle. */
function naiveRole(message: { role: string; content: unknown; metadata?: { custom?: Record<string, unknown> } }): string {
  const custom = message.metadata?.custom

  const background =
    message.role === 'system'
      ? Boolean(custom?.asyncResult || custom?.asyncResultKind)
      : message.role === 'user' && PROCESS_NOTIFICATION_RE.test(messageContentText(message.content))

  return background ? 'background' : message.role
}

/** Mirrors the structural-signature selector's access pattern over one flush. */
function structuralSignature(messages: readonly BuiltMessage[]): string {
  return messages.map((message, index) => `${index}:${message.id}:${responseMessageRole(message)}`).join('\n')
}

function transcript(turns: number, counter: Counter): BuiltMessage[] {
  const messages: BuiltMessage[] = []

  for (let turn = 0; turn < turns; turn += 1) {
    // Every third prompt is a background delivery, so classification is exercised
    // rather than trivially "all user".
    const prompt = turn % 3 === 0 ? BACKGROUND : `Question ${turn}`

    messages.push(buildMessage(`u${turn}`, 'user', prompt, counter))
    messages.push(buildMessage(`a${turn}`, 'assistant', `Answer ${turn}`, counter))
  }

  return messages
}

it('extracts each settled user message at most once across a whole streaming run', () => {
  const settled = counting()
  const messages = transcript(200, settled)
  const settledPrompts = messages.filter(message => message.role === 'user').length
  const liveTail = messages[messages.length - 1]!

  for (let flush = 0; flush < FLUSHES; flush += 1) {
    // A streaming tail publishes a fresh content array per delta, so its reads
    // are unavoidable and are charged to a throwaway counter, not to history.
    const previous = liveTail.content
    liveTail.content = [countedPart(`delta ${flush}`, counting())]

    structuralSignature([...messages.slice(0, -1), liveTail])
    expect(previous).not.toBe(liveTail.content)
  }

  // Exactly one extraction of the settled history for the WHOLE run, not one per
  // flush: the bound is the cost of reading the history once, so the assertion
  // holds for any FLUSHES. Before the fix this was FLUSHES x this number.
  expect(settled.count).toBeLessThanOrEqual(settledPrompts * PARTS_PER_MESSAGE * READS_PER_EXTRACTION)
})

it('keeps the structural signature byte-identical to the naive scan while streaming', () => {
  const messages = transcript(200, counting())
  const liveTail = messages[messages.length - 1]!

  const expected = transcript(200, counting())
    .map((message, index) => `${index}:${message.id}:${naiveRole(message)}`)
    .join('\n')

  for (let flush = 0; flush < FLUSHES; flush += 1) {
    liveTail.content = [countedPart(`delta ${flush}`, counting())]

    expect(structuralSignature(messages)).toBe(expected)
  }

  // A live assistant tail is never a background delivery, so appending text to
  // it must not reclassify it.
  expect(responseMessageRole(liveTail)).toBe('assistant')
})

it('reclassifies a user message whose content is replaced', () => {
  const settled = counting()
  const message = buildMessage('u0', 'user', 'Ordinary prompt', settled)

  expect(responseMessageRole(message)).toBe('user')
  expect(settled.count).toBe(PARTS_PER_MESSAGE * READS_PER_EXTRACTION)

  const beforeReplacement = settled.count

  // Same length, same role, same id — only the content array is new. A
  // same-length replacement must invalidate, or a rewritten prompt would keep
  // the classification of the text it no longer holds.
  message.content = [countedPart(BACKGROUND, settled)]

  expect(responseMessageRole(message)).toBe('background')
  expect(structuralSignature([message])).toBe(`0:${message.id}:background`)
  expect(settled.count).toBeGreaterThan(beforeReplacement)

  // Re-reading the same array is the case the cache exists for.
  const afterReplacement = settled.count

  expect(responseMessageRole(message)).toBe('background')
  expect(settled.count).toBe(afterReplacement)
})

it('classifies system deliveries by metadata without reading their text', () => {
  const settled = counting()
  const plain = buildMessage('s0', 'system', BACKGROUND, settled)

  const delivered = {
    ...buildMessage('s1', 'system', BACKGROUND, settled),
    metadata: { custom: { asyncResult: true } }
  }

  expect(responseMessageRole(plain)).toBe('system')
  expect(responseMessageRole(delivered)).toBe('background')
  expect(structuralSignature([plain, delivered])).toBe('0:s0:system\n1:s1:background')
  expect(settled.count).toBe(0)
})
