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
  // Slice ONLY GroupMentionInput — functions merged between it and
  // GroupChatWorkspace (e.g. GroupClarifyCard) legitimately use Input.
  const start = source.indexOf('function GroupMentionInput')
  const nextFn = source.indexOf('\nfunction ', start + 1)
  const component = source.slice(start, nextFn)

  // The control is the SDK Textarea, not the single-line Input.
  assert.match(component, /jsx\(Textarea, \{/)
  assert.doesNotMatch(component, /jsx\(Input, \{/)
  // Enter (no Shift) submits through the injected handler...
  assert.match(component, /event\.key === 'Enter' && !event\.shiftKey/)
  assert.match(component, /onSubmitDraft\?\.\(\)/)
  // ...and the popover branch still owns Enter while it is open.
  assert.match(component, /insert\(options\[active\]\.handle\)/)
})

test('room composer swallows IME composition Enters before submit/mention (#93528)', () => {
  // macOS Chinese IME: Enter that confirms a candidate word fires a keydown
  // the composer mistook for a send — the message went out mid-composition.
  // The guard must run BEFORE the popover branch AND the submit branch, and
  // cover both Chromium's isComposing flag and the keyCode 229 legacy
  // VK_PROCESSKEY that macOS IMEs emit after compositionend.
  const start = source.indexOf('function GroupMentionInput')
  const nextFn = source.indexOf('\nfunction ', start + 1)
  const component = source.slice(start, nextFn)

  const guard = component.indexOf('event.nativeEvent?.isComposing || event.keyCode === 229')
  assert.ok(guard >= 0, 'IME guard present in GroupMentionInput')
  const afterGuard = component.slice(guard)
  // Guard returns immediately (no mention insert, no submit)...
  assert.match(afterGuard.slice(0, 80), /return/)
  // ...and it sits before both the popover branch and the submit branch.
  assert.ok(afterGuard.indexOf('if (open) {') > 0, 'guard precedes popover branch')
  assert.ok(component.indexOf("event.key === 'Enter' && !event.shiftKey") > guard, 'guard precedes submit branch')
})

test('clarify free-text input swallows IME composition Enters (#93528)', () => {
  const clarify = source.slice(source.indexOf('function GroupClarifyCard'), source.indexOf('function openGroupChat'))
  const guard = clarify.indexOf('event.nativeEvent?.isComposing || event.keyCode === 229')
  assert.ok(guard >= 0, 'IME guard present in GroupClarifyCard')
  assert.ok(clarify.indexOf("event.key === 'Enter' && questions.length === 1") > guard, 'guard precedes submit branch')
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

test('retained-pane reopen re-anchors to the bottom (#89835 follow-up)', () => {
  // A hot-mounted room pane never remounts when the user tabs back to it, so
  // the workspace re-anchors on the hidden → visible edge, fed by the
  // feature-detected host.paneVisibility authority (always-visible fallback
  // keeps older SDKs on the previous behavior).
  const workspace = source.slice(source.indexOf('function GroupChatWorkspace'))
  assert.match(workspace, /visible = true/)
  assert.match(workspace, /wasVisibleRef/)
  assert.match(workspace, /\[visible\]/)
  const mainView = source.slice(source.indexOf('function GroupChatMainView'), source.indexOf('function openGroupChat'))
  assert.match(mainView, /typeof host\.paneVisibility === 'function'/)
  assert.match(mainView, /plugin-workspace:\$\{ID\}:group:\$\{slugify\(group\)\}/)
  assert.match(mainView, /atom\(true\)/)
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
