import { describe, expect, it } from 'vitest'

import { changedRows } from './draft-change-panel'
import type { ProfileDraftResult } from './profile-draft'

const section = (key: string, title: string, text: string) => ({ key, title, text })

describe('changedRows', () => {
  it('offers a removal (emptied section or field) for review', () => {
    const res = {
      title: 'Scout',
      description: '',
      sections: [section('role', 'Role', 'Same role.'), section('voice', 'Voice', '')],
      previous: {
        title: 'Scout',
        description: 'Old blurb.',
        soul: '',
        sections: [section('role', 'Role', 'Same role.'), section('voice', 'Voice', 'Old voice.')]
      }
    } as unknown as ProfileDraftResult

    expect(changedRows(res, { title: 'Title', description: 'Description' }).map(r => [r.key, r.after])).toEqual([
      ['description', ''],
      ['voice', '']
    ])
  })
})
