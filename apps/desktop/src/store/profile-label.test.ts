import { atom } from 'nanostores'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({ $gateway: atom(null) }))
vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))

const { profileLabel } = await import('./profile')

describe('profileLabel presentation-only names (#146434)', () => {
  it('prefers the Bot title over the core display name without touching identity', () => {
    const profile = Object.freeze({ name: 'research-agent', bot_title: '  中文研究助手  ', display_name: 'Research' })

    expect(profileLabel(profile)).toBe('中文研究助手')
    expect(profile.name).toBe('research-agent')
  })

  it('reads the Bot title RPC rows carry inside ui_meta', () => {
    expect(profileLabel({
      name: 'ud55c-uae00',
      display_name: 'Old name',
      ui_meta: { 'hermes-bots': { title: ' 한글 ' } }
    })).toBe('한글')
  })

  it('falls back to the core display name before the canonical id', () => {
    expect(profileLabel({ name: 'ops', bot_title: '  ', display_name: ' 運用担当 ' })).toBe('運用担当')
  })

  it('ignores a non-string Bot title from an older or damaged row', () => {
    expect(profileLabel({ name: 'ops', bot_title: 42 as unknown as string, display_name: 'Operations' }))
      .toBe('Operations')
  })

  it('preserves the ordinary default fallback unless a name is configured', () => {
    expect(profileLabel({ name: 'default' })).toBe('default')
    expect(profileLabel({ name: 'default', display_name: 'My assistant' })).toBe('My assistant')
  })

  it.each([undefined, false, 7, 'legacy', [], { 'hermes-bots': null }, { 'hermes-bots': { title: 42 } }])(
    'keeps the display name when optional metadata is malformed: %j',
    ui_meta => {
      expect(profileLabel({ name: 'ops', display_name: 'Operations', ui_meta })).toBe('Operations')
    }
  )

  it('keeps the canonical id as the last fallback without mutating the record', () => {
    const profile = Object.freeze({ name: 'u5c0f-u52a9-u624b', bot_title: ' ', display_name: ' ' })

    expect(profileLabel(profile)).toBe('u5c0f-u52a9-u624b')
    expect(profile).toEqual({ name: 'u5c0f-u52a9-u624b', bot_title: ' ', display_name: ' ' })
  })
})
