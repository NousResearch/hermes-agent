import { describe, expect, it, vi } from 'vitest'

import { nativeMessages, registerNativeLocaleIpc } from './native-i18n'
import { quitPromptFor } from './quit-guard'

describe('native dialog localization', () => {
  it.each(['sv', 'sv-SE', 'sv_FI', 'svenska', 'Swedish'])('resolves %s', locale => {
    expect(nativeMessages(locale).saveFile).toBe('Spara fil')
  })

  it('keeps sender languages isolated and removes destroyed windows', () => {
    const on = vi.fn()
    const copyFor = registerNativeLocaleIpc({ on })
    expect(on).toHaveBeenCalledWith('hermes:ui-locale', expect.any(Function))
    const receive = on.mock.calls[0][1]
    const first = { id: 1, once: vi.fn() }
    const second = { id: 2, once: vi.fn() }
    receive({ sender: first }, 'sv')
    receive({ sender: second }, 'en')
    expect(copyFor(1, 'en').saveFile).toBe('Spara fil')
    expect(copyFor(2, 'sv').saveFile).toBe('Save File')
    expect(copyFor(undefined, 'sv').saveFile).toBe('Spara fil')
    receive({ sender: first }, 'unknown')
    expect(copyFor(1, 'sv').saveFile).toBe('Save File')
    expect(first.once).toHaveBeenCalledTimes(1)
    first.once.mock.calls[0][1]()
    expect(copyFor(1, 'sv').saveFile).toBe('Spara fil')
  })

  it('translates quit warnings without changing active-work or handoff behavior', () => {
    const copy = nativeMessages('sv')
    const work = { count: 2, titles: ['Original title'] }
    const prompt = quitPromptFor(work, false, copy)
    expect(prompt?.message).toContain('2 chattar')
    expect(prompt?.detail).toContain('Original title')
    expect(prompt?.detail).toContain('• 1 till')
    expect(prompt?.detail).toContain('går förlorat')
    expect(quitPromptFor(work, true, copy)).toBeNull()
    expect(quitPromptFor({ count: 0, titles: [] }, false, copy)).toBeNull()
  })

  it('preserves diagnostic paths and plugin identifiers in Swedish notices', () => {
    const copy = nativeMessages('sv')
    expect(copy.updateDetails('External error', '/example/update.log')).toContain('/example/update.log')
    expect(copy.pluginRow('my-plugin', 2, 'old.module', 'new.module')).toContain('old.module → new.module')
    expect(copy.pluginMessage(1, '2026-09-14', true)).toContain('2026-09-14')
    expect(copy.pluginDetail('my-plugin', '2026-09-14', true)).toContain('plugins.allow_deprecated_imports: true')
  })
})
