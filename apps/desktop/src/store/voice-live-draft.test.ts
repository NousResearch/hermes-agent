import { afterEach, expect, it } from 'vitest'

import {
  $voiceLiveDrafts,
  appendVoiceLiveDraft,
  markVoiceLiveDraftClaimed,
  newVoiceLiveDraftRunId,
  retireVoiceLiveDrafts,
  savedVoiceLiveDraftIds
} from './voice-live-draft'

afterEach(() => $voiceLiveDrafts.set({}))

const timed = (startMs: number, endMs: number, eventId?: string) => ({ endMs, eventId, startMs, timingMissing: false })

it('grows the speaker run in arrival order with its call timing, and starts a new run on a speaker change', () => {
  const user = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('chat', user, 'user', 'Ask ', timed(900, 1200, 'ev-1'))
  appendVoiceLiveDraft('chat', user, 'user', 'Hermes', timed(800, 1050, 'ev-2'))
  const reply = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('chat', reply, 'assistant', 'Sure.', { endMs: 0, startMs: 0, timingMissing: true })
  expect($voiceLiveDrafts.get().chat).toEqual([
    { endMs: 1200, firstEventId: 'ev-1', id: user, role: 'user', startMs: 800, text: 'Ask Hermes' },
    { endMs: null, firstEventId: null, id: reply, role: 'assistant', startMs: null, text: 'Sure.' }
  ])
  expect($voiceLiveDrafts.get().other).toBeUndefined()
})

it('retires only the named runs and drops an emptied chat', () => {
  const [a, b] = [newVoiceLiveDraftRunId(), newVoiceLiveDraftRunId()]
  appendVoiceLiveDraft('chat', a, 'user', 'one', timed(0, 100))
  appendVoiceLiveDraft('chat', b, 'assistant', 'two', timed(100, 200))
  retireVoiceLiveDrafts('chat', new Set([a]))
  expect($voiceLiveDrafts.get().chat?.map(run => run.id)).toEqual([b])
  retireVoiceLiveDrafts('chat', new Set([b]))
  expect('chat' in $voiceLiveDrafts.get()).toBe(false)
})

it('matches a run only to a saved voice row with its speaker, call timing and words', () => {
  const runs = [
    { endMs: 1700, firstEventId: null, id: 'u', role: 'user' as const, startMs: 800, text: ' Ask\tHermes ' },
    { endMs: 2200, firstEventId: null, id: 'a', role: 'assistant' as const, startMs: 2000, text: 'Okay.' },
    { endMs: null, firstEventId: null, id: 'x', role: 'assistant' as const, startMs: null, text: 'Untimed.' }
  ]

  // Hermes's own reply and an optimistic bubble carry no voice timing: never this run.
  const unsaved = [
    { role: 'assistant', text: 'Okay.' },
    { role: 'user', text: 'Ask Hermes' },
    // Another call's row: same words, different timing.
    { role: 'assistant', text: 'Okay.', voiceLive: { endMs: 600, startMs: 400 } }
  ]

  expect(savedVoiceLiveDraftIds(runs, unsaved)).toEqual(new Set())

  const saved = [
    ...unsaved,
    { role: 'user', text: 'Ask Hermes', voiceLive: { endMs: 1700, startMs: 800 } },
    { role: 'assistant', text: 'Okay.', voiceLive: { endMs: 2200, startMs: 2000 } },
    { role: 'assistant', text: 'Untimed.', voiceLive: { endMs: null, startMs: null } }
  ]

  // Untimed words without ids stay up until the ended call's idle cleanup: they repeat too easily.
  expect(savedVoiceLiveDraftIds(runs, saved)).toEqual(new Set(['u', 'a']))
})

it('identifies a run by its first provider event before timing or words', () => {
  const runs = [
    { endMs: null, firstEventId: 'ev-9', id: 'new', role: 'assistant' as const, startMs: null, text: 'Okay.' },
    { endMs: 1300, firstEventId: 'ev-12', id: 'grouped', role: 'user' as const, startMs: 1000, text: 'Ask Hermes now' }
  ]

  // An earlier call's untimed "Okay." (null timing like this run) is not this run.
  expect(savedVoiceLiveDraftIds(runs, [
    { role: 'assistant', text: 'Okay.', voiceLive: { endMs: null, firstEventId: 'ev-2', startMs: null } }
  ])).toEqual(new Set())

  // Its own row retires it even when the owner's words or end time differ slightly.
  expect(savedVoiceLiveDraftIds(runs, [
    { role: 'assistant', text: 'Okay. ', voiceLive: { endMs: null, firstEventId: 'ev-9', startMs: null } },
    { role: 'user', text: 'Ask Hermes now.', voiceLive: { endMs: 1500, firstEventId: 'ev-12', startMs: 1000 } }
  ])).toEqual(new Set(['new', 'grouped']))

  // Without ids on either side, untimed words never match (they repeat too easily).
  expect(savedVoiceLiveDraftIds([{ ...runs[0], firstEventId: null }], [
    { role: 'assistant', text: 'Okay.', voiceLive: { endMs: null, startMs: null } }
  ])).toEqual(new Set())
})

it('never falls back to timing for a run that has an id', () => {
  const run = { endMs: 1200, firstEventId: 'ev-new', id: 'r', role: 'assistant' as const, startMs: 800, text: 'Okay' }
  // A row saved before rows carried ids, at the same call-relative time in an older call.
  expect(savedVoiceLiveDraftIds([run], [{ role: 'assistant', text: 'Okay', voiceLive: { endMs: 1200, startMs: 800 } }])).toEqual(new Set())
})

it('marks only the run a delegation claimed', () => {
  const [first, second] = [newVoiceLiveDraftRunId(), newVoiceLiveDraftRunId()]
  appendVoiceLiveDraft('chat', first, 'user', 'Yes', timed(0, 200, 'ev-a'))
  appendVoiceLiveDraft('chat', second, 'user', 'Yes', timed(900, 1100, 'ev-b'))
  markVoiceLiveDraftClaimed('chat', 'ev-b', 'voice-live:["call","d2"]')
  expect($voiceLiveDrafts.get().chat?.map(run => [run.id, run.claimedBy ?? null])).toEqual([
    [first, null],
    [second, 'voice-live:["call","d2"]']
  ])
})
