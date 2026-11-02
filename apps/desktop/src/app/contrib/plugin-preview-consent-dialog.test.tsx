// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $previewScriptConsentQueue,
  $previewScriptGrants,
  ensurePreviewScriptConsent
} from '@/store/plugin-preview-grants'

import { PluginPreviewConsentDialog } from './plugin-preview-consent-dialog'

beforeEach(() => {
  window.localStorage.clear()
  $previewScriptGrants.set({})
  $previewScriptConsentQueue.set([])
})

afterEach(cleanup)

describe('PluginPreviewConsentDialog', () => {
  it('renders nothing while no plugin is asking', () => {
    render(<PluginPreviewConsentDialog />)

    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('names the plugin and the page origin, and Allow persists the grant', async () => {
    render(<PluginPreviewConsentDialog />)

    let answer!: Promise<boolean>
    act(() => {
      answer = ensurePreviewScriptConsent('page-reader', 'Page Reader', 'https://mail.example.com')
    })

    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain('Page Reader')
    expect(dialog.textContent).toContain('https://mail.example.com')

    fireEvent.click(screen.getByRole('button', { name: 'Allow on this site' }))

    await expect(answer).resolves.toBe(true)
    expect($previewScriptGrants.get()['page-reader']).toEqual(['https://mail.example.com'])
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('treats Escape as a refusal, never as consent', async () => {
    render(<PluginPreviewConsentDialog />)

    let answer!: Promise<boolean>
    act(() => {
      answer = ensurePreviewScriptConsent('page-reader-esc', 'Page Reader', 'https://mail.example.com')
    })

    fireEvent.keyDown(await screen.findByRole('dialog'), { key: 'Escape' })

    await expect(answer).resolves.toBe(false)
    expect($previewScriptGrants.get()).toEqual({})
  })
})
