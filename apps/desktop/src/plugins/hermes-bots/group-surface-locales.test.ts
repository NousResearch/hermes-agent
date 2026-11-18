import { expect, it } from 'vitest'

import { BOTS_LOCALES } from './i18n'
import { translateBotsIn } from './i18n-test-helper'

// The shared group views, creation/member/settings dialogs, and roster consume
// these namespaces. Requiring each English key catches additions as well as gaps.
const namespaces = ['group', 'canonical', 'crossGateway', 'roster', 'sections'] as const

it.each([
  ['fr', 'Section « Notes » supprimée', 'bot sans section', 'bots sans section'],
  ['es', 'Sección «Notes» eliminada', 'bot sin sección', 'bots sin sección']
])('names the deleted section with feminine agreement in %s', (locale, sentence, singular, plural) => {
  const t = translateBotsIn(locale)
  expect([0, 1, 2].map(count => t('sections.deleted', 'Notes', count))).toEqual([
    sentence, `${sentence} — 1 ${singular}`, `${sentence} — 2 ${plural}`
  ])
})

const sharedKeys = {
  bot: ['newChatWith', 'newTitle', 'thisDevice', 'removeFromAllGroups', 'openBotChat', 'openBotChatInNewTab', 'openingChat',
    'pinToTop', 'unpin', 'pinnedToast', 'unpinnedToast', 'hide', 'unhide', 'hiddenToast', 'unhiddenToast',
    'groupsMenu', 'manageGroups', 'metadataLoadFailed', 'loadFailed', 'groupsLoadFailed', 'attentionFallback',
    'attentionProviderAuth', 'attentionQuota', 'attentionMissingConfig', 'attentionBlocked', 'editMenu', 'duplicate', 'duplicateFailed'],
  avatar: ['upload', 'generate', 'generating'],
  screen: ['menu', 'autoOpenMenu', 'autoOpenOnToast', 'autoOpenOffToast']
}

function requireOwnTranslations(english: unknown, localized: unknown, path: string) {
  expect(localized, path).toBeDefined()
  expect(typeof localized, path).toBe(typeof english)

  if (english && typeof english === 'object') {
    for (const [key, value] of Object.entries(english)) {
      requireOwnTranslations(value, (localized as Record<string, unknown> | undefined)?.[key], `${path}.${key}`)
    }
  }
}

it.each(['ar', 'ru', 'fr', 'de', 'es', 'ja', 'zh', 'zh-hant'])('owns every group-screen translation in %s without English fallback', locale => {
  const bundle = BOTS_LOCALES[locale] as Record<string, Record<string, unknown>>
  const english = BOTS_LOCALES.en as Record<string, Record<string, unknown>>

  for (const namespace of namespaces) {
    requireOwnTranslations(english[namespace], bundle[namespace], `${locale}.${namespace}`)
  }

  for (const [namespace, keys] of Object.entries(sharedKeys)) {
    for (const key of keys) {
      requireOwnTranslations(english[namespace][key], bundle[namespace]?.[key], `${locale}.${namespace}.${key}`)
    }
  }
})
