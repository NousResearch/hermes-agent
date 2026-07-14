import { describe, expect, it } from 'vitest'

import { ALL_PROFILES } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

import { filterSessionsByProfileScope } from './profile-scope'

const session = (id: string, profile?: string): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  profile,
  source: 'feishu',
  started_at: 1,
  title: null,
  tool_call_count: 0
})

describe('filterSessionsByProfileScope', () => {
  it('keeps only messaging rows from the selected profile', () => {
    const rows = [session('default-row', 'default'), session('research-row', 'research')]

    expect(filterSessionsByProfileScope(rows, 'research').map(s => s.id)).toEqual(['research-row'])
  })

  it('treats missing profiles as default', () => {
    const rows = [session('legacy-row'), session('research-row', 'research')]

    expect(filterSessionsByProfileScope(rows, 'default').map(s => s.id)).toEqual(['legacy-row'])
  })

  it('keeps every messaging row when persisted All profiles mode outlives the profile switcher', () => {
    const rows = [session('default-row', 'default'), session('research-row', 'research')]

    expect(filterSessionsByProfileScope(rows, ALL_PROFILES).map(s => s.id)).toEqual([
      'default-row',
      'research-row'
    ])
  })
})
