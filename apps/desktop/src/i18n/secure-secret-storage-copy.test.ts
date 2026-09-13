import assert from 'node:assert/strict'

import { test } from 'vitest'

import { TRANSLATIONS } from './catalog'
import { en } from './en'
import {
  SECURE_SECRET_STORAGE_DESCRIPTIONS,
  withSecureSecretStorageCopy
} from './secure-secret-storage-copy'
import type { Locale } from './types'

const LOCALES: Locale[] = ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru']

test('secure-default secret-storage copy covers every supported locale', () => {
  assert.deepEqual(Object.keys(SECURE_SECRET_STORAGE_DESCRIPTIONS).sort(), [...LOCALES].sort())

  for (const locale of LOCALES) {
    assert.equal(
      TRANSLATIONS[locale].settings.gateway.keychainEncryptionDesc,
      SECURE_SECRET_STORAGE_DESCRIPTIONS[locale]
    )
    assert.match(SECURE_SECRET_STORAGE_DESCRIPTIONS[locale], /\S/)
  }
})

test('catalog overlay is immutable and removes the shipped default-off claim', () => {
  const originalDescription = en.settings.gateway.keychainEncryptionDesc
  const overlaid = withSecureSecretStorageCopy('en', en)

  assert.notEqual(overlaid, en)
  assert.equal(en.settings.gateway.keychainEncryptionDesc, originalDescription)
  assert.equal(overlaid.settings.gateway.keychainEncryptionDesc, SECURE_SECRET_STORAGE_DESCRIPTIONS.en)
  assert.doesNotMatch(overlaid.settings.gateway.keychainEncryptionDesc, /off by default/i)
})
