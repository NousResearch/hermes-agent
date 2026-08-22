import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Room message-body overflow (2026-08-21): a bot reply containing a long
// fenced code block did not wrap and the room offered no horizontal scroll,
// so the code was clipped on BOTH visible edges — unreadable. The approval
// card's command block already wraps (whitespace-pre-wrap break-all); the
// message body was the outlier with a bare `[&_pre]:overflow-x-auto`, and the
// room's scroll viewport clips x with no scroll affordance.
//
// Tracked upstream: #70451 (markdown/code forces horizontal scroll), #91706
// (approval Respond unreachable — the overflowing pre sat over the footer).

test('bot message body code blocks wrap instead of clipping (#70451)', () => {
  // Slice ONLY renderEntry — the function owning the message-body container.
  const start = source.indexOf('const renderEntry = (entry, index) => {')
  assert.ok(start !== -1, 'renderEntry exists')
  const end = source.indexOf('\n  // Threads:', start + 1)
  const component = source.slice(start, end === -1 ? source.length : end)

  // The body must wrap long code (pre-wrap) and break unbreakable tokens
  // (break-words) so nothing is ever clipped off the room's fixed x extent.
  assert.match(component, /\[&_pre\]:whitespace-pre-wrap/)
  assert.match(component, /\[&_pre\]:break-words/)
  // A bare overflow-x-auto (horizontal scroll with no room affordance) must
  // not be the only handling — it is what produced the clipped edges.
  assert.doesNotMatch(component, /\[&_pre\]:overflow-x-auto$/)
})
