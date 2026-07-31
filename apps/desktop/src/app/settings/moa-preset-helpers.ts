import type { MoaConfigResponse, MoaModelSlot, MoaPresetConfig } from '@/types/hermes'

export type MoaPreset = MoaPresetConfig
export type MoaPresetNameIssue = 'blank' | 'duplicate'

export const hasOwnMoaPreset = (config: MoaConfigResponse, name: string): boolean =>
  Object.prototype.hasOwnProperty.call(config.presets, name)

export const getOwnMoaPreset = (config: MoaConfigResponse, name: string): MoaPresetConfig | undefined =>
  hasOwnMoaPreset(config, name) ? config.presets[name] : undefined

export function validateMoaPresetName(
  config: MoaConfigResponse,
  candidate: string,
  currentName?: string
): MoaPresetNameIssue | null {
  const name = candidate.trim()

  if (!name) {
    return 'blank'
  }

  if (name !== currentName && hasOwnMoaPreset(config, name)) {
    return 'duplicate'
  }

  return null
}

function checkedPresetName(config: MoaConfigResponse, candidate: string, currentName?: string): string {
  const name = candidate.trim()
  const issue = validateMoaPresetName(config, name, currentName)

  if (issue) {
    throw new Error(issue)
  }

  return name
}

export function renameMoaPreset(config: MoaConfigResponse, from: string, to: string): MoaConfigResponse {
  const preset = getOwnMoaPreset(config, from)

  if (!preset) {
    throw new Error('missing-preset')
  }

  const name = checkedPresetName(config, to, from)

  if (name === from) {
    return config
  }

  const presets = Object.fromEntries([
    ...Object.entries(config.presets).filter(([presetName]) => presetName !== from),
    [name, preset]
  ])

  return {
    ...config,
    active_preset: config.active_preset === from ? name : config.active_preset,
    default_preset: config.default_preset === from ? name : config.default_preset,
    presets
  }
}

function withoutContinuityId(slot: MoaModelSlot): MoaModelSlot {
  const { continuity_id: _continuityId, ...aggregator } = slot

  return aggregator as MoaModelSlot
}

export function createMoaPreset(config: MoaConfigResponse, to: string): MoaConfigResponse {
  const name = checkedPresetName(config, to)

  const blank: MoaPresetConfig = {
    aggregator: { model: '', provider: '' },
    aggregator_temperature: null,
    enabled: true,
    fanout: 'user_turn',
    max_tokens: 4096,
    reference_max_tokens: null,
    reference_models: [{ model: '', provider: '' }],
    reference_temperature: null
  }

  return {
    ...config,
    presets: Object.fromEntries([...Object.entries(config.presets), [name, blank]])
  }
}

export function duplicateMoaPreset(config: MoaConfigResponse, source: string, to: string): MoaConfigResponse {
  const preset = getOwnMoaPreset(config, source)

  if (!preset) {
    throw new Error('missing-preset')
  }

  const name = checkedPresetName(config, to)

  const copy: MoaPresetConfig = {
    ...preset,
    aggregator: withoutContinuityId({ ...preset.aggregator }),
    // continuity_id is intentionally copied: matching ids across named presets
    // are deliberate shared advisory lanes, while each slot object stays
    // independent so later edits cannot alias across presets.
    reference_models: preset.reference_models.map(slot => ({ ...slot }))
  }

  return {
    ...config,
    presets: Object.fromEntries([...Object.entries(config.presets), [name, copy]])
  }
}

export function deleteMoaPreset(config: MoaConfigResponse, name: string): MoaConfigResponse {
  if (!hasOwnMoaPreset(config, name)) {
    throw new Error('missing-preset')
  }

  if (Object.keys(config.presets).length <= 1) {
    throw new Error('last-preset')
  }

  const presets = Object.fromEntries(Object.entries(config.presets).filter(([presetName]) => presetName !== name))
  const fallback = Object.keys(presets)[0]

  return {
    ...config,
    active_preset: config.active_preset === name ? fallback : config.active_preset,
    default_preset: config.default_preset === name ? fallback : config.default_preset,
    presets
  }
}

export function moveMoaReference(preset: MoaPreset, from: number, to: number): MoaPreset {
  if (
    from === to ||
    from < 0 ||
    to < 0 ||
    from >= preset.reference_models.length ||
    to >= preset.reference_models.length
  ) {
    return preset
  }

  const referenceModels = [...preset.reference_models]
  const [slot] = referenceModels.splice(from, 1)
  referenceModels.splice(to, 0, slot)

  return { ...preset, reference_models: referenceModels }
}

export function addMoaReference(preset: MoaPreset): MoaPreset {
  const source = preset.reference_models.at(-1) ?? preset.aggregator
  const continuityIds = new Set(preset.reference_models.map(slot => slot.continuity_id).filter(Boolean))
  let ordinal = preset.reference_models.length + 1
  let continuityId = `reference-${ordinal}`

  while (continuityIds.has(continuityId)) {
    ordinal += 1
    continuityId = `reference-${ordinal}`
  }

  return {
    ...preset,
    reference_models: [...preset.reference_models, { ...source, continuity_id: continuityId }]
  }
}

export function updateMoaSlot(slot: MoaModelSlot, patch: Partial<MoaModelSlot>): MoaModelSlot {
  const next = { ...slot, ...patch }

  if (typeof patch.provider === 'string' && patch.provider !== slot.provider) {
    next.model = ''
  }

  return next
}

export function prepareMoaConfigForSave(config: MoaConfigResponse): MoaConfigResponse {
  return {
    ...config,
    aggregator: withoutContinuityId(config.aggregator),
    presets: Object.fromEntries(
      Object.entries(config.presets).map(([name, preset]) => [
        name,
        { ...preset, aggregator: withoutContinuityId(preset.aggregator) }
      ])
    )
  }
}

export const moaSlotComplete = (slot: MoaModelSlot): boolean => {
  const provider = typeof slot.provider === 'string' ? slot.provider.trim().toLowerCase() : ''
  const model = typeof slot.model === 'string' ? slot.model.trim() : ''

  return !!provider && provider !== 'moa' && !!model
}

const finiteOptional = (value: unknown): boolean => value == null || (typeof value === 'number' && Number.isFinite(value))
const positiveInteger = (value: unknown): boolean => typeof value === 'number' && Number.isInteger(value) && value > 0

/**
 * Fanout acceptance aligned with hermes_cli/moa_config._coerce_fanout:
 * `user_turn` | `per_iteration` | `every_n:<N>` where N >= 2.
 * Unknown/invalid values are incomplete in Studio (fail closed for Save).
 */
export function isValidMoaFanout(value: unknown): boolean {
  // Absent fanout is valid (backend default user_turn). Explicit values must canonicalize.
  if (value == null || value === '') return true
  return canonicalizeMoaFanout(value) != null
}

/** Normalize a fanout value to the canonical string, or null if invalid/absent. */
export function canonicalizeMoaFanout(value: unknown): string | null {
  if (value == null || value === '') {
    return null
  }

  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    const record = value as Record<string, unknown>
    const mode = String(record.mode ?? '').trim().toLowerCase()
    if (mode === 'every_n') {
      const n = Number(record.n)
      if (Number.isInteger(n) && n >= 2) return `every_n:${n}`
      if (n === 1) return 'per_iteration'
      return null
    }
    if (mode === 'user_turn' || mode === 'per_iteration') return mode
    return null
  }

  if (typeof value !== 'string') return null
  const mode = value.trim().toLowerCase()
  if (mode === 'user_turn' || mode === 'per_iteration') return mode
  if (mode.startsWith('every_n')) {
    const sep = mode.indexOf(':')
    const rest = sep >= 0 ? mode.slice(sep + 1).trim() : ''
    const n = Number(rest)
    if (Number.isInteger(n) && n >= 2) return `every_n:${n}`
    if (n === 1) return 'per_iteration'
    return null
  }
  return null
}

export function moaFanoutMode(value: unknown): 'user_turn' | 'per_iteration' | 'every_n' {
  const formatted = canonicalizeMoaFanout(value)
  if (formatted == null) {
    // nullish original means default; invalid also falls back for UI display only
    if (value == null || value === '') return 'user_turn'
    return 'user_turn'
  }
  if (formatted.startsWith('every_n:')) return 'every_n'
  return formatted as 'user_turn' | 'per_iteration'
}

export function moaFanoutEveryN(value: unknown, fallback = 3): number {
  const formatted = canonicalizeMoaFanout(value)
  if (formatted && formatted.startsWith('every_n:')) {
    const n = Number(formatted.slice('every_n:'.length))
    if (Number.isInteger(n) && n >= 2) return n
  }
  return fallback >= 2 ? fallback : 3
}

export function buildMoaFanout(mode: 'user_turn' | 'per_iteration' | 'every_n', everyN = 3): string {
  if (mode === 'every_n') {
    const n = Number.isInteger(everyN) && everyN >= 2 ? everyN : 3
    return `every_n:${n}`
  }
  return mode
}


export const moaConfigComplete = (config: MoaConfigResponse): boolean => {
  const presetNames = Object.keys(config.presets)

  if (
    presetNames.length === 0 ||
    !hasOwnMoaPreset(config, config.default_preset) ||
    (config.active_preset !== '' && !hasOwnMoaPreset(config, config.active_preset))
  ) {
    return false
  }

  return Object.values(config.presets).every(preset => {
    const referenceMaxTokens = preset.reference_max_tokens
    const fanout = preset.fanout

    return (
      preset.reference_models.length > 0 &&
      preset.reference_models.every(moaSlotComplete) &&
      moaSlotComplete(preset.aggregator) &&
      positiveInteger(preset.max_tokens) &&
      (referenceMaxTokens == null || positiveInteger(referenceMaxTokens)) &&
      finiteOptional(preset.reference_temperature) &&
      finiteOptional(preset.aggregator_temperature) &&
      (fanout == null || isValidMoaFanout(fanout)) &&
      typeof preset.enabled === 'boolean'
    )
  })
}
