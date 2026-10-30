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
    expect(fa.settings.sessions.defaultDirUpdated('Ctrl+N')).toContain('(Ctrl+N)')
    expect(fa.settings.sessions.defaultDirUpdated(null)).not.toMatch(/null|Ctrl|⌘|\(\)/)
  })

  it('preserves recovery parameters and handles optional installation details', () => {
    expect(fa.assistant.thread.errorRateLimitResetBody('Provider-X', '12:45')).toContain('Provider-X')
    expect(fa.assistant.thread.errorRateLimitResetBody('Provider-X', '12:45')).toContain('12:45')
    const body = fa.assistant.thread.errorCodes.agent_init_failed.body
    expect(typeof body).toBe('function')

    if (typeof body !== 'function') {throw new Error('Expected an interpolated error message')}
    expect(body('Provider-X', 'model-X')).toContain('Provider-X')
    expect(fa.settings.plugins.installModal.sourceMoved('abc123')).toContain('abc123')
    expect(fa.settings.plugins.installModal.partialInstall('abc123')).toContain('abc123')
    expect(fa.settings.plugins.installModal.partialInstall('')).not.toMatch(/undefined|null|commit/)
    expect(fa.cron.control.lastCancelled('reason-X')).toContain('reason-X')
    expect(fa.cron.control.lastCancelled('')).not.toContain(':')
  })
})
