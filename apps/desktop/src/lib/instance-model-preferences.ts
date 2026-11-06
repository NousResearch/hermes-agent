/** Instance preferences never change a running model or a profile default. */
export interface InstanceModelRow {
  provider: string
  model: string
  favorite: boolean
  hidden: boolean | null
  favoriteRevision: string
  hiddenRevision: string
}
export interface InstanceModelSnapshot {
  version: 1
  instanceId: string
  revision: string
  rows: InstanceModelRow[]
}
export type PreferenceField = 'favorite' | 'hidden'
export const tuple = (provider: string, model: string) => JSON.stringify([provider, model])

const revision = (v: unknown): v is string =>
  typeof v === 'string' && /^(0|[1-9][0-9]{0,18})$/.test(v) && BigInt(v) < 2n ** 63n

function identifier(value: unknown, max: number, empty = false): value is string {
  return (
    typeof value === 'string' &&
    (empty || value.length > 0) &&
    value.length <= max &&
    [...value].every(c => c.charCodeAt(0) >= 32)
  )
}

function validRow(row: InstanceModelRow, upper: string): boolean {
  return (
    Boolean(row) &&
    identifier(row.provider, 128) &&
    identifier(row.model, 512, true) &&
    typeof row.favorite === 'boolean' &&
    (row.hidden === null || typeof row.hidden === 'boolean') &&
    revision(row.favoriteRevision) &&
    revision(row.hiddenRevision) &&
    BigInt(row.favoriteRevision) <= BigInt(upper) &&
    BigInt(row.hiddenRevision) <= BigInt(upper)
  )
}

export function parseInstanceModels(raw: unknown): InstanceModelSnapshot {
  const s = raw as InstanceModelSnapshot

  if (
    !s ||
    s.version !== 1 ||
    typeof s.instanceId !== 'string' ||
    !/^[a-f0-9-]{36}$/i.test(s.instanceId) ||
    !revision(s.revision) ||
    !Array.isArray(s.rows) ||
    s.rows.length > 5000
  )
    {throw new Error('Invalid preference snapshot')}

  const keys = new Set<string>()

  for (const row of s.rows) {
    if (!validRow(row, s.revision) || keys.has(tuple(row.provider, row.model)))
      {throw new Error('Invalid preference row')}

    keys.add(tuple(row.provider, row.model))
  }

  return s
}

export function hiddenModel(snapshot: InstanceModelSnapshot | null, provider: string, model: string): boolean {
  const row = snapshot?.rows.find(r => r.provider === provider && r.model === model)

  if (row?.hidden !== null && row?.hidden !== undefined) {
    return row.hidden
  }

  return snapshot?.rows.some(r => r.provider === provider && r.model === '' && r.hidden === true) ?? false
}

export interface PreferenceMutation {
  instanceId: string
  provider: string
  model: string
  field: PreferenceField
  baseRevision: string
  value: boolean | null
}

export function preferenceMutation(
  snapshot: InstanceModelSnapshot,
  provider: string,
  model: string,
  field: PreferenceField,
  value: boolean | null,
  importing = false
): PreferenceMutation {
  const row = snapshot.rows.find(r => r.provider === provider && r.model === model)

  return {
    instanceId: snapshot.instanceId,
    provider,
    model,
    field,
    baseRevision: importing ? '0' : (row?.[field === 'favorite' ? 'favoriteRevision' : 'hiddenRevision'] ?? '0'),
    value
  }
}
