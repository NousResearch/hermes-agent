import {
  canonicalDesktopSlashCommand,
  type DesktopActionId,
  type DesktopPickerId,
  resolveDesktopCommand,
  type SlashCommandBuildCtx
} from './desktop-slash-commands'

/**
 * Stable catalog identity consumed by the desktop client projection.
 *
 * `command_id` is present on commands.catalog.v2. Older backends only expose
 * `name`; the canonical slash name is therefore the mixed-version fallback.
 */
export interface DesktopCatalogCommandIdentity {
  command_id?: string | null
  name: string
}

/** Execution-only binding contributed by the desktop client. */
export type DesktopClientCommandSurface =
  | Readonly<{ action: DesktopActionId; kind: 'action' }>
  | Readonly<{ kind: 'picker'; picker: DesktopPickerId }>
  | Readonly<{
      buildParams: (ctx: SlashCommandBuildCtx) => Record<string, unknown>
      kind: 'rpc'
      rpc: string
      timeoutMs?: number
    }>

/**
 * A client-owned execution binding. Catalog semantics and presentation stay
 * with the server catalog; this object says only how Desktop fulfils the id.
 */
export interface DesktopClientCommandBinding {
  readonly canonicalName: string
  readonly commandId: string
  readonly surface: DesktopClientCommandSurface
}

function projectClientSurface(command: string): DesktopClientCommandSurface | null {
  const surface = resolveDesktopCommand(command)?.surface

  switch (surface?.kind) {
    case 'action':
      return Object.freeze({ action: surface.action, kind: 'action' })
    case 'picker':
      return Object.freeze({ kind: 'picker', picker: surface.picker })
    case 'rpc': {
      const binding = {
        buildParams: surface.buildParams,
        kind: 'rpc' as const,
        rpc: surface.rpc,
        ...(surface.timeoutMs === undefined ? {} : { timeoutMs: surface.timeoutMs })
      }

      return Object.freeze(binding)
    }
    case 'exec':
    case 'unavailable':
    case undefined:
      return null
  }
}

function stableCommandId(identity: DesktopCatalogCommandIdentity, canonicalName: string): string {
  if (identity.command_id == null) {
    return canonicalName
  }

  const commandId = identity.command_id.trim()

  if (!commandId) {
    throw new Error(`Desktop command binding for "${canonicalName}" has an empty command_id`)
  }

  return commandId
}

/**
 * Resolve one catalog row into a Desktop-owned client binding.
 *
 * Server-executed, unavailable, and unknown commands deliberately return null:
 * they must settle through the shared command invocation path instead of being
 * re-owned by this client projection.
 */
export function resolveDesktopClientCommandBinding(
  identity: DesktopCatalogCommandIdentity
): DesktopClientCommandBinding | null {
  const canonicalName = canonicalDesktopSlashCommand(identity.name)
  const surface = projectClientSurface(canonicalName)

  if (!surface) {
    return null
  }

  return Object.freeze({
    canonicalName,
    commandId: stableCommandId(identity, canonicalName),
    surface
  })
}

/**
 * Project a catalog into immutable Desktop client bindings.
 *
 * Duplicate stable ids or duplicate canonical commands are authority
 * collisions and fail closed rather than allowing array order to choose an
 * execution owner.
 */
export function projectDesktopClientCommandBindings(
  identities: readonly DesktopCatalogCommandIdentity[]
): readonly DesktopClientCommandBinding[] {
  const bindings: DesktopClientCommandBinding[] = []
  const byCommandId = new Map<string, string>()
  const byCanonicalName = new Map<string, string>()

  for (const identity of identities) {
    const binding = resolveDesktopClientCommandBinding(identity)

    if (!binding) {
      continue
    }

    const idOwner = byCommandId.get(binding.commandId)

    if (idOwner) {
      throw new Error(
        `Desktop command_id collision: "${binding.commandId}" is bound by both "${idOwner}" and "${binding.canonicalName}"`
      )
    }

    const canonicalOwner = byCanonicalName.get(binding.canonicalName)

    if (canonicalOwner) {
      throw new Error(
        `Desktop canonical command collision: "${binding.canonicalName}" is projected by both "${canonicalOwner}" and "${identity.name}"`
      )
    }

    byCommandId.set(binding.commandId, binding.canonicalName)
    byCanonicalName.set(binding.canonicalName, identity.name)
    bindings.push(binding)
  }

  return Object.freeze(bindings)
}