import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import type { GroupHoldStamp } from './group-chat'
import { applyGroupHoldDirective, parseGroupChatMentions } from './group-rounds'
import type { GroupMember } from './types'

// The same vectors the hosted-room policy runs in tests/gateway/test_hosted_room_holds.py. Both
// implementations answer them with their real helpers, so "stop" and "resume" cannot come to mean
// two different things depending on which frontend the user typed into.
function repoFile(relative: string): string {
  // The renderer suite runs under jsdom, where import.meta.url is an http URL: walk up from the
  // working directory instead so the path holds however the runner is invoked.
  for (let directory = resolve(process.cwd()); ; directory = dirname(directory)) {
    const candidate = join(directory, relative)

    if (existsSync(candidate)) {return candidate}

    if (dirname(directory) === directory) {throw new Error(`${relative} was not found above ${process.cwd()}`)}
  }
}

const vectors = JSON.parse(
  readFileSync(repoFile('tests/fixtures/hosted_room_hold_directives.json'), 'utf8')
) as {
  members: Array<{ member_id: string; display_name: string; handle: string }>
  cases: Array<{ name: string; text: string; hold: string[]; release: string[] }>
}

// Member keys are names here, so both mention resolvers address the same members. The renamed
// friendly name rides as display_name — the same field the hosted roster carries and the same one
// botMentionTag builds its inserted tag from. Titles stay outside this shared contract.
const members: GroupMember[] = vectors.members.map(member => ({
  name: member.handle, title: '', display_name: member.display_name
}))

const keys = members.map(member => member.name)
const idFor = new Map(vectors.members.map(member => [member.handle, member.member_id]))
const stamp: GroupHoldStamp = { at: 1, byMessageId: null, thread: 'mock-thread' }
const allHeld = () => Object.fromEntries(keys.map(key => [key, { ...stamp }]))

/** Effective hold/release of one message, derived from the real native helpers: what it adds to
 *  an empty hold map, and what it removes from a fully held one. */
function nativeEffect(text: string) {
  const mentions = parseGroupChatMentions(text, members)
  const added = applyGroupHoldDirective({}, mentions, text, stamp, keys)
  const remaining = applyGroupHoldDirective(allHeld(), mentions, text, stamp, keys)

  return {
    hold: Object.keys(added).map(key => idFor.get(key)).sort(),
    release: keys.filter(key => !(key in remaining)).map(key => idFor.get(key)).sort()
  }
}

describe('shared hold-directive vectors', () => {
  it('covers both control words and both @all forms', () => {
    expect(vectors.cases.length).toBeGreaterThan(20)
    expect(vectors.cases.some(one => one.text.includes('@all'))).toBe(true)
    expect(vectors.cases.some(one => one.text.includes('@everyone'))).toBe(true)
  })

  it.each(vectors.cases.map(one => [one.name, one] as const))('%s', (_name, one) => {
    expect(nativeEffect(one.text)).toEqual({
      hold: [...one.hold].sort(),
      release: [...one.release].sort()
    })
  })
})
