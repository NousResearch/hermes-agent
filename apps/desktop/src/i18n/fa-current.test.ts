import { describe, expect, it } from 'vitest'

import { fa } from './fa'

describe('current Persian desktop catalog', () => {
  it('translates post-locale desktop surfaces without English fallback', () => {
    expect(fa.settings.connections.title).toBe('دروازه‌های ثبت‌شده')
    expect(fa.settings.appearance.sessionDensityCompact).toBe('فشرده')
    expect(fa.contextMenu.link.openInApp).toBe('باز کردن در مرورگر داخلی')
    expect(fa.tips.items['new-session'].title).toBe('شروع تازه')
  })

  it('keeps interpolated values intact in Persian copy', () => {
    expect(fa.profiles.remoteOverride.savedMessage('پژوهش', 'server.example')).toContain('server.example')
    expect(fa.assistant.clarify.questionProgress(2, 3)).toBe('2 از 3 پاسخ داده شد')
    expect(fa.settings.mcp.importConfirmMany(4)).toBe('افزودن 4 سرور به mcp.json')
  })
})
