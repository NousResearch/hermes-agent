import { render, screen } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import { I18nProvider, useI18n } from './context'
import { getManagedRolloutMessages, managedRolloutsZh } from './managed-rollouts'

function MessageProbe() {
  const { t } = useI18n()

  return createElement('p', null, getManagedRolloutMessages(t).sections.preparation)
}

describe('managed rollout catalog namespace', () => {
  it('loads the selected feature copy through the catalog and I18nProvider', () => {
    render(createElement(I18nProvider, {
      configClient: null,
      initialLocale: 'zh',
      children: createElement(MessageProbe)
    }))

    expect(screen.getByText(managedRolloutsZh.sections.preparation)).toBeTruthy()
    expect(getManagedRolloutMessages(TRANSLATIONS.zh)).toBe(managedRolloutsZh)
  })
})
