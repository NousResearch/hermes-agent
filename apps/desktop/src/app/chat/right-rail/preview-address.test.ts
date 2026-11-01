import { describe, expect, it } from 'vitest'

import { normalizePreviewAddress, START_SHORTCUTS, startShortcutAddress } from './preview-address'

describe('normalizePreviewAddress', () => {
  it('passes through http and https URLs', () => {
    expect(normalizePreviewAddress('https://example.com/path')).toBe('https://example.com/path')
    expect(normalizePreviewAddress('http://localhost:3000/')).toBe('http://localhost:3000/')
  })

  it('gives a bare host https and a loopback host http', () => {
    expect(normalizePreviewAddress('example.com')).toBe('https://example.com')
    expect(normalizePreviewAddress('localhost:5173')).toBe('http://localhost:5173')
    expect(normalizePreviewAddress('127.0.0.1:8080/app')).toBe('http://127.0.0.1:8080/app')
  })

  it('rejects empty input', () => {
    expect(normalizePreviewAddress('')).toBeNull()
    expect(normalizePreviewAddress('   ')).toBeNull()
  })

  // This is a browser: the schemes a browser navigates to all work, and the
  // user already owns the machine, so `file://` is theirs to open.
  it('passes through the other schemes a browser can navigate', () => {
    expect(normalizePreviewAddress('about:blank')).toBe('about:blank')
    expect(normalizePreviewAddress('file:///etc/hosts')).toBe('file:///etc/hosts')
    expect(normalizePreviewAddress('view-source:https://example.com')).toBe('view-source:https://example.com')
  })

  // These EXECUTE in the guest partition instead of navigating it — an address
  // bar is not an eval box, and no shipping browser honors them here either.
  it('rejects script-bearing schemes', () => {
    expect(normalizePreviewAddress('javascript:alert(1)')).toBeNull()
    expect(normalizePreviewAddress('data:text/html,evil')).toBeNull()
    expect(normalizePreviewAddress('JavaScript:alert(1)')).toBeNull()
  })

  it('rejects strings that are not addresses at all', () => {
    expect(normalizePreviewAddress('://broken')).toBeNull()
    expect(normalizePreviewAddress('http://')).toBeNull()
  })

  it('trims whitespace before validating', () => {
    expect(normalizePreviewAddress('  https://example.com  ')).toBe('https://example.com')
  })
})

describe('startShortcutAddress', () => {
  it('offers the ports a local app lands on', () => {
    expect(START_SHORTCUTS).toEqual(['localhost:3000', 'localhost:5173', 'localhost:8000'])
  })

  // A chip is a dev server on the machine the pane runs on, so every one of
  // them has to resolve to loopback http — https there has no certificate and
  // nothing listening on 443.
  it('resolves every chip to loopback http', () => {
    for (const shortcut of START_SHORTCUTS) {
      const address = startShortcutAddress(shortcut)

      expect(address).toBe(`http://${shortcut}`)
      expect(address && new URL(address).hostname).toBe('localhost')
    }
  })

  it('returns null for anything that is not a plain address', () => {
    expect(startShortcutAddress('')).toBeNull()
    expect(startShortcutAddress('javascript:alert(1)')).toBeNull()
    expect(startShortcutAddress('data:text/html,evil')).toBeNull()
  })
})
