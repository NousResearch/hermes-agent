import type { Translations } from '@/i18n/types'
import { prettyName } from '@/lib/text'
import type { ConfigFieldSchema } from '@/types/hermes'

import { FIELD_DESCRIPTIONS, FIELD_LABELS } from './constants'
import { fieldCopyForSchemaKey } from './field-copy'

export function resolveConfigFieldCopy(t: Translations, schemaKey: string, schema: ConfigFieldSchema) {
  const label =
    fieldCopyForSchemaKey(t.settings.fieldLabels, schemaKey) ??
    fieldCopyForSchemaKey(FIELD_LABELS, schemaKey) ??
    prettyName(schemaKey.split('.').pop() ?? schemaKey)

  const normalize = (v: string) =>
    v
      .toLowerCase()
      .normalize('NFC')
      .replace(/[^\p{L}\p{M}\p{N}]+/gu, '')

  const rawDescription = (
    fieldCopyForSchemaKey(t.settings.fieldDescriptions, schemaKey) ??
    fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, schemaKey) ??
    schema.description ??
    ''
  ).trim()

  const normalizedDesc = normalize(rawDescription)

  const description =
    rawDescription && normalizedDesc !== normalize(label) && normalizedDesc !== normalize(schemaKey)
      ? rawDescription
      : undefined

  return { label, description }
}
