import { describe, expect, it } from 'vitest'

import { profileInitial } from './profile-color'

describe('profileInitial (#118765)', () => {
  it('extracts and capitalizes ASCII initials', () => {
    expect(profileInitial('work')).toBe('W')
    expect(profileInitial('developer')).toBe('D')
    expect(profileInitial('Personal')).toBe('P')
  })

  it('skips leading punctuation and symbols to find the first letter or digit', () => {
    expect(profileInitial('@team')).toBe('T')
    expect(profileInitial('#general')).toBe('G')
    expect(profileInitial('  --dev--  ')).toBe('D')
  })

  it('supports numeric profile names', () => {
    expect(profileInitial('42')).toBe('4')
    expect(profileInitial('3d-render')).toBe('3')
  })

  it('supports CJK profile names without collapsing to "?"', () => {
    expect(profileInitial('研究助手')).toBe('研')
    expect(profileInitial('テスト')).toBe('テ')
    expect(profileInitial('인공지능')).toBe('인')
  })

  it('supports Cyrillic and Greek profile names with uppercase transformation', () => {
    expect(profileInitial('работа')).toBe('Р')
    expect(profileInitial('Ελληνικά')).toBe('Ε')
  })

  it('supports accented and non-ASCII Latin characters with uppercase transformation', () => {
    expect(profileInitial('árbol')).toBe('Á')
    expect(profileInitial('über')).toBe('Ü')
    expect(profileInitial('ñoño')).toBe('Ñ')
  })

  it('supports emoji-only profile names and preserves compound emojis', () => {
    expect(profileInitial('🚀')).toBe('🚀')
    expect(profileInitial('🤖')).toBe('🤖')
    expect(profileInitial('👨‍💻')).toBe('👨‍💻')
    expect(profileInitial('  ✨  ')).toBe('✨')
  })

  it('prioritizes letters when mixed with emojis', () => {
    expect(profileInitial('🤖 Robot')).toBe('R')
  })

  it('falls back to "?" for empty or whitespace-only inputs', () => {
    expect(profileInitial('')).toBe('?')
    expect(profileInitial('   ')).toBe('?')
    expect(profileInitial(null)).toBe('?')
    expect(profileInitial(undefined)).toBe('?')
  })

  it('returns the first grapheme when input consists solely of symbols without letters or digits', () => {
    expect(profileInitial('---')).toBe('-')
    expect(profileInitial('***')).toBe('*')
  })
})
