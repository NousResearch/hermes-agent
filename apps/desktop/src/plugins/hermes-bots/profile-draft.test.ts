import { describe, expect, it } from 'vitest'

import {
  applyDraftCapabilities,
  composeDraftSoul,
  decideField,
  mergeDraftPart,
  mergeSoulSections,
  type ProfileDraftResult
} from './profile-draft'

const saved = [
  '# Scout',
  '',
  '## Role',
  '',
  'Old role.',
  '',
  '## Messaging other agents',
  '',
  'Keep me exactly.',
  ''
].join('\n')

describe('mergeSoulSections', () => {
  it('replaces an accepted section in place and keeps everything else byte-for-byte', () => {
    const out = mergeSoulSections(saved, [{ key: 'role', title: 'Role', text: 'New role.' }])

    expect(out).toBe(saved.replace('Old role.', 'New role.'))
  })

  it('appends a section the saved SOUL does not have', () => {
    const out = mergeSoulSections(saved, [{ key: 'voice', title: 'Voice', text: 'Terse.' }])

    expect(out.startsWith(saved.trimEnd())).toBe(true)
    expect(out.endsWith('## Voice\n\nTerse.\n')).toBe(true)
  })

  it('replaces a section under a closed heading instead of appending a duplicate', () => {
    const closed = saved.replace('## Role', '## Role ##')
    const out = mergeSoulSections(closed, [{ key: 'role', title: 'Role', text: 'New role.' }])

    expect(out).toBe(closed.replace('Old role.', 'New role.'))
  })
})

describe('mergeDraftPart', () => {
  it('applies out-of-order Regenerate replies without restoring the other section', () => {
    const draft = (role: string, voice: string) =>
      ({
        sections: [
          { key: 'role', title: 'Role', text: role },
          { key: 'voice', title: 'Voice', text: voice }
        ]
      }) as ProfileDraftResult

    // Voice then Role were regenerated from the same snapshot; Role's reply lands first.
    const latest = mergeDraftPart(draft('Old role.', 'Old voice.'), draft('New role.', 'Old voice.'), 'role')

    expect(mergeDraftPart(latest, draft('Old role.', 'New voice.'), 'voice').sections.map(s => s.text)).toEqual([
      'New role.',
      'New voice.'
    ])
  })
})

describe('decideField', () => {
  it('keeps typed text through other decisions and restores it when its proposal is rejected', () => {
    const typed = { value: 'Typed title' }

    // Accepting only Voice: the title's decision did not change.
    expect(decideField(typed, undefined, undefined)).toEqual(typed)

    const accepted = decideField(typed, undefined, 'Proposed title')

    expect(accepted.value).toBe('Proposed title')
    expect(decideField(accepted, 'Proposed title', undefined).value).toBe('Typed title')
  })
})

describe('composeDraftSoul', () => {
  it('skips empty sections', () => {
    expect(
      composeDraftSoul('Mail', [
        { key: 'role', title: 'Role', text: 'R' },
        { key: 'voice', title: 'Voice', text: ' ' }
      ])
    ).toBe('# Mail\n\n## Role\n\nR\n')
  })
})

describe('applyDraftCapabilities', () => {
  it('turns picks on without turning anything else off', async () => {
    const calls: [string, Record<string, unknown> | undefined][] = []

    const request = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
      calls.push([method, params])

      return {
        skills: [
          { name: 'himalaya', enabled: false },
          { name: 'arxiv', enabled: false }
        ],
        toolsets: [
          { name: 'terminal', enabled: true },
          { name: 'web', enabled: false }
        ],
        mcp_servers: [{ name: 'gmail', enabled: false }]
      } as T
    }

    await applyDraftCapabilities(request, 'mailbot', {
      skills: ['email/himalaya'],
      toolsets: ['web'],
      mcp_servers: ['gmail']
    })

    expect(calls[1]).toEqual([
      'profiles.configure',
      { name: 'mailbot', disabled_skills: ['arxiv'], enabled_toolsets: ['terminal', 'web'], enabled_mcp_servers: ['gmail'] }
    ])
  })
})
