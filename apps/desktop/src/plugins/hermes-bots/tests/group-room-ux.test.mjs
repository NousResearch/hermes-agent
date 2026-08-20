import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Room UX cluster (Aug 2026 desktop audit): three confirmed defects in the
// group-chat surface, fixed together.
//
//   #89884 — the composer was a single-line Input; Enter always submitted and
//            multi-line prompts were impossible.
//   #89835 — rooms opened at scroll position 0 with no anchoring; new replies
//            streamed in below the fold.
//   #89545 — a member whose reply outlived the turn loop stayed stranded
//            until the user's NEXT send (harvest only ran inside the loop).

test('room composer is a textarea with Enter=submit, Shift+Enter=newline (#89884)', () => {
  const component = source.slice(
    source.indexOf('function GroupMentionInput'),
    source.indexOf('function GroupChatWorkspace')
  )

  // The control is the SDK Textarea, not the single-line Input.
  assert.match(component, /jsx\(Textarea, \{/)
  assert.doesNotMatch(component, /jsx\(Input, \{/)
  // Enter (no Shift) submits through the injected handler...
  assert.match(component, /event\.key === 'Enter' && !event\.shiftKey/)
  assert.match(component, /onSubmitDraft\?\.\(\)/)
  // ...and the popover branch still owns Enter while it is open.
  assert.match(component, /insert\(options\[active\]\.handle\)/)
})

test('both room composers wire onSubmitDraft (#89884)', () => {
  assert.match(source, /onSubmitDraft: submit,/)
  assert.match(source, /onSubmitDraft: \(\) => submitReply\(id\),/)
})

test('room log anchors to the bottom with a user-scroll guard (#89835)', () => {
  const workspace = source.slice(source.indexOf('function GroupChatWorkspace'))
  assert.match(workspace, /bottomSentinelRef/)
  assert.match(workspace, /stickToBottomRef/)
  // Effect keys on log growth; the guard keeps history reading stable.
  assert.match(workspace, /\[room\.log\.length, room\.running\]/)
  assert.match(workspace, /scrollIntoView/)
})

test('stranded replies get a bounded background harvest after the loop settles (#89545)', () => {
  assert.match(source, /function harvestStrandedUntilSettled\(/)
  // The loop's finally block kicks it off only when members remain stranded.
  assert.match(source, /strandedLeft\.length/)
  // Bounded: interval + max tries, and it yields to a live loop.
  const harvester = source.slice(source.indexOf('async function harvestStrandedUntilSettled'))
  assert.match(harvester.slice(0, 1600), /HARVEST_MAX_TRIES/)
  assert.match(harvester.slice(0, 1600), /room\.running/)
})
