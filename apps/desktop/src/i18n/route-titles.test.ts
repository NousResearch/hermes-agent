import { describe, expect, it } from 'vitest'

import { APP_ROUTES } from '@/app/routes'
import { TRANSLATIONS } from './catalog'

// The sr-only <h1> (RouteHeading) announces t.shell.routeTitles[view] for the
// routed AppView. That index assumes every locale covers every view, so this
// makes the invariant executable: any locale that omits a route title (or
// ships an empty string) fails here instead of announcing a blank heading.
describe('i18n route titles', () => {
  const views = APP_ROUTES.map(route => route.view)

  it('every locale defines a non-empty route title for every routed view', () => {
    for (const [locale, translations] of Object.entries(TRANSLATIONS) as Array<
      [string, (typeof TRANSLATIONS)[keyof typeof TRANSLATIONS]]
    >) {
      for (const view of views) {
        const title = translations.shell.routeTitles[view]
        expect(title, `${locale} missing route title for ${view}`).toBeTruthy()
        expect(title.trim(), `${locale} route title for ${view} is empty`).not.toHaveLength(0)
      }
    }
  })
})
