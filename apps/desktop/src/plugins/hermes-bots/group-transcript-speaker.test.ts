/**
 * A cross-machine room's transcript names each line's speaker as that
 * speaker's OWN machine reports it, also once the speaker has left the room
 * (or before its source has painted after a relaunch). An unseated remote
 * `default` used to fall back to a bare `{ name: 'default' }` row and read as
 * this device's "Hermes".
 *
 * Drives the real `routing` + `labels` modules; only the SDK `host` is mocked.
 */

import { describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', () => ({ host: { state: {} } }))

import { displayName } from './labels'
import { groupTranscriptSpeaker, groupTranscriptSpeakerMeta } from './routing'
import type { RosterRow } from './types'

const localDefault = { name: 'default' } as RosterRow
const scribe = { name: 'scribe' } as RosterRow
const homelabDefault = { name: 'default', connectionId: 'homelab', connectionLabel: 'Homelab', remoteSource: true, sourceScoped: true } as RosterRow
const remoteLine = { from: { kind: 'member' as const, name: 'default', source: 'Homelab' } }
const localLine = { from: { kind: 'member' as const, name: 'default' } }

const label = (entry: typeof remoteLine | typeof localLine, members: RosterRow[], roster: RosterRow[] = [], allMeta = {}) =>
  displayName(groupTranscriptSpeaker(entry, members, roster)!, groupTranscriptSpeakerMeta(entry, members, allMeta, roster))

describe('group transcript speaker', () => {
  it('names a seated remote default by its machine, the local one as Hermes', () => {
    expect(label(remoteLine, [localDefault, homelabDefault])).toBe('Homelab')
    expect(label(localLine, [localDefault, homelabDefault])).toBe('Hermes')
  })

  it('keeps a removed remote member owner-qualified instead of reading as this device', () => {
    expect(label(remoteLine, [localDefault, scribe])).toBe('Homelab')
    expect(label(remoteLine, [localDefault, scribe], [localDefault, scribe, homelabDefault])).toBe('Homelab')
  })

  it('uses the title the removed member\u2019s own backend reports, never the local twin\u2019s', () => {
    const titled = { ...homelabDefault, title: 'Atlas' } as RosterRow
    const allMeta = { default: { title: 'Local Title' } }

    expect(label(remoteLine, [localDefault, scribe], [localDefault, titled], allMeta)).toBe('Atlas')
    expect(label(remoteLine, [localDefault, scribe], [], allMeta)).toBe('Homelab')
  })
})
