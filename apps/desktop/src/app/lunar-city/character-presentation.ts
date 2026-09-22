import type { CharacterAssetManifest, LunarEntity, WorkerCharacterKit, WorkerCharacterPresentation } from './model'

function exactPrimaryKit(entity: LunarEntity, assets: CharacterAssetManifest): WorkerCharacterKit | undefined {
  const presentation = entity.presentation
  const primaryGroupId = presentation?.placement.primaryGroupId

  if (!presentation || presentation.metadata.state === 'unavailable' || !primaryGroupId) {
    return undefined
  }

  const primary = presentation.groups.find(group => group.id === primaryGroupId)

  if (!primary) {
    return undefined
  }

  return assets.groupKits.find(kit => kit.group === primary.name)
}

function identityAccent(key: string): string {
  const bytes = new TextEncoder().encode(key)

  return Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
}

/**
 * Builds presentation from immutable exact identity and the exact-source
 * configured primary group only. The reversible UTF-8 accent is collision
 * free for distinct canonical keys and allocates no GPU resource.
 */
export function characterPresentationForEntity(
  entity: LunarEntity,
  assets: CharacterAssetManifest
): WorkerCharacterPresentation | undefined {
  if (entity.identity.kind !== 'profile') {
    return undefined
  }

  const placement = entity.presentation!.placement
  const lod = placement.slot === undefined || placement.lodHint >= 2 ? 'far' : placement.lodHint === 1 ? 'mid' : 'near'
  const renderMode = placement.slot === undefined ? 'aggregate' : lod === 'near' ? 'animated' : 'instanced'

  if (placement.slot === undefined) {
    return Object.freeze({ lod, renderMode })
  }

  const kit = exactPrimaryKit(entity, assets)
  const accent = identityAccent(entity.key)
  const accentCode = placement.slot

  const signature = {
    ...(kit?.signature ?? {}),
    body: assets.workerVocabulary.bodies[accentCode % assets.workerVocabulary.bodies.length]!,
    head:
      assets.workerVocabulary.heads[Math.floor(accentCode / 2) % assets.workerVocabulary.heads.length] ??
      assets.workerVocabulary.heads[0]!,
    palette:
      assets.workerVocabulary.palettes[Math.floor(accentCode / 4) % assets.workerVocabulary.palettes.length] ??
      assets.workerVocabulary.palettes[0]!
  }

  return Object.freeze({
    accentCode,
    identityAccent: accent,
    ...(kit ? { kitId: kit.kitId } : {}),
    lod,
    renderMode,
    signature: Object.freeze(signature),
    visibleSignature: [
      signature.body,
      signature.head,
      signature.silhouetteAccessory ?? 'neutral',
      signature.palette,
      signature.emblem ?? 'neutral',
      accent
    ].join(':')
  })
}
