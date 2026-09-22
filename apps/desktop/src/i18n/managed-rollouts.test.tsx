import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from './context'
import { TRANSLATIONS } from './catalog'
import {
  managedRolloutsAr,
  managedRolloutsEn,
  managedRolloutsJa,
  managedRolloutsRu,
  managedRolloutsZh,
  managedRolloutsZhHant,
  getManagedRolloutMessages
} from './managed-rollouts'
import type { ManagedRolloutMessages } from './managed-rollouts-types'

const locales = {
  en: managedRolloutsEn,
  zh: managedRolloutsZh,
  'zh-hant': managedRolloutsZhHant,
  ja: managedRolloutsJa,
  ar: managedRolloutsAr,
  ru: managedRolloutsRu
} satisfies Record<string, ManagedRolloutMessages>

function MessageProbe() {
  const { t, locale } = useI18n()
  const messages = getManagedRolloutMessages(t, locale)

  return (
    <div>
      <p>{messages.sections.preparation}</p>
      <p>{messages.actions.recheckOutcome}</p>
      <p>{messages.status.unknown}</p>
      <p>{messages.status.fenced}</p>
    </div>
  )
}

function leafPaths(value: unknown, prefix = ''): string[] {
  if (typeof value === 'function' || typeof value === 'string') {
    return [prefix]
  }

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return []
  }

  return Object.entries(value).flatMap(([key, child]) => leafPaths(child, prefix ? `${prefix}.${key}` : key))
}

describe('managed rollout locale namespace', () => {
  it('resolves a non-English namespace through the catalog and useI18n path', () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <MessageProbe />
      </I18nProvider>
    )

    expect(screen.getByText(managedRolloutsZh.sections.preparation)).toBeTruthy()
    expect(screen.getByText(managedRolloutsZh.actions.recheckOutcome)).toBeTruthy()
    expect(getManagedRolloutMessages(TRANSLATIONS.zh, 'zh')).toEqual(managedRolloutsZh)
  })

  it('keeps warning, action, unknown, and fence copy as distinct semantic leaves', () => {
    for (const messages of Object.values(locales)) {
      expect(messages.warnings.commitmentBoundary).not.toBe(messages.actions.stop)
      expect(messages.warnings.unknownOutcome).not.toBe(messages.actions.recheckOutcome)
      expect(messages.status.unknown).not.toBe(messages.status.fenced)
      expect(messages.warnings.unknownAndFenced).not.toBe(messages.status.unknown)
      expect(messages.warnings.unknownAndFenced).not.toBe(messages.status.fenced)
    }
  })

  it('keeps every locale key-complete with the English namespace', () => {
    const expected = leafPaths(managedRolloutsEn).sort()

    for (const [locale, messages] of Object.entries(locales)) {
      expect(leafPaths(messages).sort(), locale).toEqual(expected)
    }
  })

  it('retains a readable full-sentence long-text warning in every locale', () => {
    for (const messages of Object.values(locales)) {
      expect(messages.descriptions.commitmentBoundary.length).toBeGreaterThan(60)
      expect(messages.descriptions.commitmentBoundary).not.toMatch(/undefined|null/)
    }
  })

  it('keeps RTL copy semantic rather than directional', () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <MessageProbe />
      </I18nProvider>
    )

    expect(document.documentElement.dir).toBe('rtl')
    expect(screen.getByText(managedRolloutsAr.status.unknown)).toBeTruthy()
    expect(screen.getByText(managedRolloutsAr.status.fenced)).toBeTruthy()
  })
})
