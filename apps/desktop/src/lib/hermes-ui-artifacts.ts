export interface HermesUiChecklistItem {
  id: string
  label: string
  description?: string
}

export interface HermesUiChecklistArtifact {
  type: 'checklist'
  direction?: 'auto' | 'ltr' | 'rtl'
  id?: string
  title?: string
  description?: string
  items: HermesUiChecklistItem[]
}

export type HermesUiArtifact = HermesUiChecklistArtifact

export interface HermesUiArtifactParseSuccess {
  artifact: HermesUiArtifact
  ok: true
}

export interface HermesUiArtifactParseFailure {
  error: string
  ok: false
}

export type HermesUiArtifactParseResult = HermesUiArtifactParseFailure | HermesUiArtifactParseSuccess

const MAX_TITLE_LENGTH = 160
const MAX_DESCRIPTION_LENGTH = 500
const MAX_ITEMS = 100
const MAX_ITEM_ID_LENGTH = 120
const MAX_LABEL_LENGTH = 800
const MAX_ITEM_DESCRIPTION_LENGTH = 1000

function normalizeDirection(value: unknown): HermesUiChecklistArtifact['direction'] | HermesUiArtifactParseFailure {
  if (value === undefined) {
    return undefined
  }

  if (value === 'auto' || value === 'ltr' || value === 'rtl') {
    return value
  }

  return { error: 'direction must be auto, ltr, or rtl', ok: false }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeText(value: unknown, maxLength: number, field: string): string | HermesUiArtifactParseFailure {
  if (typeof value !== 'string') {
    return { error: `${field} must be a string`, ok: false }
  }

  const text = value.replace(/\0/g, '').trim()

  if (text.length > maxLength) {
    return { error: `${field} is too long`, ok: false }
  }

  return text
}

function optionalText(
  value: unknown,
  maxLength: number,
  field: string
): HermesUiArtifactParseFailure | string | undefined {
  if (value === undefined) {
    return undefined
  }

  const text = normalizeText(value, maxLength, field)

  return typeof text === 'string' && text.length === 0 ? undefined : text
}

function normalizeIdentity(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, MAX_ITEM_ID_LENGTH)
}

export function parseHermesUiArtifact(source: string): HermesUiArtifactParseResult {
  let parsed: unknown

  try {
    parsed = JSON.parse(source)
  } catch {
    return { error: 'Invalid JSON', ok: false }
  }

  if (!isRecord(parsed)) {
    return { error: 'Artifact must be an object', ok: false }
  }

  if (parsed.type !== 'checklist') {
    return { error: 'Unsupported artifact type', ok: false }
  }

  if (!Array.isArray(parsed.items) || parsed.items.length === 0) {
    return { error: 'Checklist items are required', ok: false }
  }

  if (parsed.items.length > MAX_ITEMS) {
    return { error: 'Checklist has too many items', ok: false }
  }

  const id = optionalText(parsed.id, MAX_ITEM_ID_LENGTH, 'id')

  if (id && typeof id !== 'string') {
    return id
  }

  const title = optionalText(parsed.title, MAX_TITLE_LENGTH, 'title')

  if (title && typeof title !== 'string') {
    return title
  }

  const description = optionalText(parsed.description, MAX_DESCRIPTION_LENGTH, 'description')

  if (description && typeof description !== 'string') {
    return description
  }

  const direction = normalizeDirection(parsed.direction)

  if (direction && typeof direction === 'object') {
    return direction
  }

  const seenIds = new Set<string>()
  const items: HermesUiChecklistItem[] = []

  for (const [index, rawItem] of parsed.items.entries()) {
    if (!isRecord(rawItem)) {
      return { error: `items[${index}] must be an object`, ok: false }
    }

    const itemId = normalizeText(rawItem.id, MAX_ITEM_ID_LENGTH, `items[${index}].id`)

    if (typeof itemId !== 'string') {
      return itemId
    }

    if (!itemId) {
      return { error: `items[${index}].id is required`, ok: false }
    }

    if (seenIds.has(itemId)) {
      return { error: `Duplicate item id: ${itemId}`, ok: false }
    }

    seenIds.add(itemId)

    const label = normalizeText(rawItem.label, MAX_LABEL_LENGTH, `items[${index}].label`)

    if (typeof label !== 'string') {
      return label
    }

    if (!label) {
      return { error: `items[${index}].label is required`, ok: false }
    }

    const itemDescription = optionalText(
      rawItem.description,
      MAX_ITEM_DESCRIPTION_LENGTH,
      `items[${index}].description`
    )

    if (itemDescription && typeof itemDescription !== 'string') {
      return itemDescription
    }

    items.push({ description: itemDescription, id: itemId, label })
  }

  return {
    artifact: { description, direction, id, items, title, type: 'checklist' },
    ok: true
  }
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`
  }

  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .filter(key => value[key] !== undefined)
      .map(key => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
      .join(',')}}`
  }

  return JSON.stringify(value)
}

function stableHash(value: string): string {
  let hash = 0x811c9dc5

  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }

  return (hash >>> 0).toString(36)
}

export function stableArtifactStorageKey(artifact: HermesUiChecklistArtifact): string {
  const identity = artifact.id ? normalizeIdentity(artifact.id) : ''
  const suffix = identity || stableHash(stableStringify(artifact))

  return `hermes-ui:checklist:${suffix}`
}
