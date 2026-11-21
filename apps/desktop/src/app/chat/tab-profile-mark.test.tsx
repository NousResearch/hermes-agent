import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { TabProfileMark } from './tab-profile-mark'

const html = (profile: null | string | undefined) =>
  renderToStaticMarkup(
    <I18nProvider configClient={null} initialLocale="en">
      <TabProfileMark profile={profile} />
    </I18nProvider>
  )

describe('a session tab profile mark', () => {
  it('says nothing for the default profile — the workspace is the default', () => {
    expect(html('default')).toBe('')
    expect(html(undefined)).toBe('')
    expect(html('')).toBe('')
  })

  it("marks a named profile with its initial and names it, like the sidebar row", () => {
    const markup = html('ynov')

    expect(markup).toContain('>y<')
    expect(markup).toContain('aria-label')
  })
})
