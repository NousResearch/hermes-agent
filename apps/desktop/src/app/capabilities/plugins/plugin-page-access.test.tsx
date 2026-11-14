// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $previewScriptGrants, ensurePreviewScriptConsent, grantPreviewScript } from '@/store/plugin-preview-grants'

import { PluginPageAccess } from './plugin-page-access'

beforeEach(() => {
  window.localStorage.clear()
  $previewScriptGrants.set({})
})
afterEach(cleanup)

describe('PluginPageAccess', () => {
  it('stays hidden for a plugin that did not declare preview scripts', () => {
    grantPreviewScript('quiet', 'https://mail.example.com')
    const { container } = render(<PluginPageAccess plugin={{ capabilities: [], id: 'quiet' }} pluginName="Quiet" />)

    expect(container.textContent).toBe('')
  })

  it('lists granted origins and revoking one makes the next call ask again', async () => {
    grantPreviewScript('page-reader', 'https://mail.example.com')
    grantPreviewScript('page-reader', 'https://docs.example.com')
    render(
      <PluginPageAccess
        plugin={{ capabilities: ['preview.executeJavaScript'], id: 'page-reader' }}
        pluginName="Page Reader"
      />
    )

    expect(screen.getByText('https://mail.example.com')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Revoke access to https://mail.example.com' }))

    expect($previewScriptGrants.get()['page-reader']).toEqual(['https://docs.example.com'])
    expect(screen.queryByText('https://mail.example.com')).toBeNull()

    let settled = false
    void ensurePreviewScriptConsent('page-reader', 'Page Reader', 'https://mail.example.com').then(
      () => (settled = true)
    )
    await Promise.resolve()
    expect(settled).toBe(false)
  })

  it('says it will ask when the plugin declared the door but holds no grant', () => {
    render(
      <PluginPageAccess plugin={{ capabilities: ['preview.executeJavaScript'], id: 'fresh' }} pluginName="Fresh" />
    )

    expect(screen.getByText('Asks before running scripts on a page')).toBeTruthy()
  })
})
