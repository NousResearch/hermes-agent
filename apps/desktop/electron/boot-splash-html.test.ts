import { describe, expect, it } from 'vitest'

import { BOOT_SPLASH_SHOW_AFTER_MS, buildBootSplashHtml, bootSplashStatusScript } from './boot-splash-html'

describe('buildBootSplashHtml', () => {
  const meta = { version: '1.2.3', stampLabel: 'abc123def456 (main)' }

  it('renders the still-booting title and the version', () => {
    const html = buildBootSplashHtml(meta)
    expect(html).toContain('Still booting — please wait')
    expect(html).toContain('Hermes Desktop v1.2.3')
  })

  it('shows the install-stamp label when present and hides it when absent', () => {
    const stamped = buildBootSplashHtml(meta)
    expect(stamped).toContain('abc123def456 (main)')

    const bare = buildBootSplashHtml({ version: '1.2.3', stampLabel: null })
    expect(bare).toContain('Hermes Desktop v1.2.3')
    expect(bare).not.toContain('abc123def456')
  })

  it('escapes HTML in version and stamp label', () => {
    const html = buildBootSplashHtml({ version: '1.2.3', stampLabel: '<img src=x onerror=1> (main)' })
    expect(html).toContain('&lt;img src=x onerror=1&gt;')
    expect(html).not.toContain('<img src=x onerror=1>')
  })

  it('exposes the live status updater hook', () => {
    const html = buildBootSplashHtml(meta)
    expect(html).toContain('window.__hermesBootStatus')
    expect(html).toContain('id="boot-status"')
  })
})

describe('bootSplashStatusScript', () => {
  it('calls the updater with a JSON-encoded message', () => {
    expect(bootSplashStatusScript('hi')).toBe('window.__hermesBootStatus && window.__hermesBootStatus("hi")')
  })

  it('keeps quotes and backslashes safe inside the generated source', () => {
    const script = bootSplashStatusScript('backend "spoke" \\ fast')
    expect(script).toContain('window.__hermesBootStatus("backend \\"spoke\\" \\\\ fast")')
  })
})

describe('splash timing constants', () => {
  it('waits long enough to never flash on a normal launch, and polls cheaply', () => {
    expect(BOOT_SPLASH_SHOW_AFTER_MS).toBeGreaterThanOrEqual(4000)
  })
})
