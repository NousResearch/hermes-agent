import { chatMessageText } from './parts'
import type { ChatMessage, VoiceLiveRun } from './types'

/** A speaker who resumes within this long is continuing the same sentence: the
 *  other speaker's rows in between were said over them (crosstalk). An ordinary
 *  exchange leaves seconds between turns; an interrupted sentence does not. */
const CROSSTALK_RESUME_MS = 600

const timed = (message: ChatMessage) =>
  Boolean(
    message.voiceLive?.callKey &&
      message.voiceLive.startMs !== null &&
      message.voiceLive.endMs !== null &&
      !message.hidden
  )

const sameCall = (a: ChatMessage, b: ChatMessage) => a.voiceLive?.callKey === b.voiceLive?.callKey

function runsOf(message: ChatMessage): VoiceLiveRun[] {
  if (message.voiceLiveRuns) {
    return message.voiceLiveRuns
  }

  const voice = message.voiceLive!

  return [{ ...voice, text: chatMessageText(message) }]
}

// A number keeps a sign written right before it (−5, -5), so 5 never matches −5.
const words = (text: string) =>
  text.toLowerCase().match(/(?:(?<![\p{L}\p{N}])[-−](?=\p{N}))?[\p{L}\p{N}]+/gu)?.map(word => word.replace('−', '-')) ?? []

// Words that can flip a statement (`t` is the tail of n't after splitting), and
// spoken signs: their counts must match the reply's.
const NEGATIONS = new Set(['cannot', 'minus', 'negative', 'never', 'no', 'none', 'nor', 'not', 'nothing', 't'])

const count = (list: string[], keep: (word: string) => boolean) =>
  list.filter(keep).reduce((acc, word) => acc.set(word, (acc.get(word) ?? 0) + 1), new Map<string, number>())

const sameCounts = (a: Map<string, number>, b: Map<string, number>) =>
  a.size === b.size && [...a].every(([word, n]) => b.get(word) === n)

/** A reading that could change the reply's meaning: its negations differ from
 *  the reply's (added or dropped), or it says a number the reply does not. */
function differs(said: string[], written: string[]): boolean {
  const negation = (word: string) => NEGATIONS.has(word)
  const reply = new Set(written)

  return !sameCounts(count(said, negation), count(written, negation)) || said.some(word => /\d/.test(word) && !reply.has(word))
}

/** Whether a spoken row is the voice reading the reply above it: most of what
 *  was said is in the reply (the voice paraphrases and adds a lead-in like
 *  "Sure, checking."). A changed negation or number makes it `differs`; only a
 *  `repeat` may be hidden. */
export function relayOf(spoken: string, written: string): ChatMessage['spokenRelay'] {
  const said = words(spoken)
  const reply = new Set(words(written))
  const shared = said.filter(word => reply.has(word)).length

  if (shared < 3 || shared / Math.max(said.length, 1) < 0.5) {
    return undefined
  }

  return differs(said, words(written)) ? 'differs' : 'repeat'
}

function joinText(left: string, right: string): string {
  return !left || !right || /\s$/.test(left) || /^\s/.test(right) ? left + right : `${left} ${right}`
}

/** Fold `next` (a later run of the same speaker) into `into`. */
function join(into: ChatMessage, next: ChatMessage): ChatMessage {
  const runs = [...runsOf(into), ...runsOf(next)]
  const text = joinText(chatMessageText(into), chatMessageText(next))
  const ends = runs.map(run => run.endMs)

  return {
    ...into,
    parts: [{ text, type: 'text' }],
    voiceLive: {
      ...into.voiceLive!,
      endMs: ends.includes(null) ? null : Math.max(...(ends as number[]))
    },
    voiceLiveRuns: runs
  }
}

/**
 * Display-only shaping of saved GPT-Live rows; storage and the model's history
 * are untouched.
 *
 * Crosstalk: the owner saves each speaker's run in arrival order, so talking
 * over each other splits a sentence across rows ("That was one of" / "Okay," /
 * "them" / "good."). Within consecutive timed voice rows, a speaker's row joins
 * that speaker's previous row when only the other speaker's rows lie between
 * them and the speaker resumed within CROSSTALK_RESUME_MS. A Hermes reply,
 * an untimed row or a real pause ends the run of joins.
 *
 * Spoken relay: the voice reads each Hermes reply aloud, and that reading is
 * saved as the next voice row. It is marked `spokenRelay` (so the thread can
 * collapse or hide it, voice.gpt_live.spoken_replies) only when it repeats the
 * reply: original speech after a reply stays visible.
 *
 * Render-time only (runtime-repository): session state keeps one durable row
 * per message for paging, rewind and reactions.
 */
export function shapeVoiceLiveMessages(messages: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = []
  // Each speaker's latest row in the current timed stretch: its input position
  // (to check what lies between in arrival order) and its output position.
  let last: Partial<Record<ChatMessage['role'], { input: number; output: number }>> = {}

  messages.forEach((message, input) => {
    if (!timed(message)) {
      last = {}
      out.push(message)

      return
    }

    const seen = last[message.role]
    const between = seen ? messages.slice(seen.input + 1, input) : []

    if (
      seen &&
      between.length > 0 &&
      sameCall(out[seen.output], message) &&
      between.every(row => timed(row) && row.role !== message.role && sameCall(row, message)) &&
      message.voiceLive!.startMs! - out[seen.output].voiceLive!.endMs! <= CROSSTALK_RESUME_MS
    ) {
      out[seen.output] = join(out[seen.output], message)
      last[message.role] = { input, output: seen.output }

      return
    }

    last[message.role] = { input, output: out.length }
    out.push(message)
  })

  return out.map((message, i) => {
    const previous = out[i - 1]

    const relay =
      message.role === 'assistant' && message.voiceLive && previous?.role === 'assistant' && !previous.voiceLive && !previous.hidden
        ? relayOf(chatMessageText(message), chatMessageText(previous))
        : undefined

    return relay ? { ...message, spokenRelay: relay } : message
  })
}
