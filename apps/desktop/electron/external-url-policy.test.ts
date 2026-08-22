import { describe, expect, it } from 'vitest'

import { isAllowedShellExternalProtocol } from './external-url-policy'

describe('external URL protocol policy', () => {
  it('allows web, mail, and Obsidian handoffs', () => {
    expect(isAllowedShellExternalProtocol('http:')).toBe(true)
    expect(isAllowedShellExternalProtocol('https:')).toBe(true)
    expect(isAllowedShellExternalProtocol('mailto:')).toBe(true)
    expect(isAllowedShellExternalProtocol('obsidian:')).toBe(true)
  })

  it('rejects arbitrary custom and executable protocols', () => {
    expect(isAllowedShellExternalProtocol('javascript:')).toBe(false)
    expect(isAllowedShellExternalProtocol('vscode:')).toBe(false)
    expect(isAllowedShellExternalProtocol('unknown-app:')).toBe(false)
  })
})
