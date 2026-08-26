/**
 * Workspace scoping for contributions.
 *
 * Pure presentation-ownership helpers: which workspace surface (sessions vs
 * bots) a contribution belongs to, and — within the bots surface — which exact
 * bot it belongs to. Owner keys are opaque exact strings supplied by callers;
 * this module never parses profile names or infers connections.
 *
 * No persistence here by design: the remembered active-pane map is window-local
 * memory so a switch away and back can restore where the user was, without any
 * of it surviving the window.
 */

import { atom, batch } from 'nanostores'

import type { WorkspaceMode } from '../../contrib/types'

/** Re-exported so workspace consumers can import it from here. */
export type { WorkspaceMode } from '../../contrib/types'

/** Default workspace mode when the host has not switched surfaces. */
export const $workspaceMode = atom<WorkspaceMode>('sessions')

/** Default workspace owner key: none (unscoped / global ownership). */
export const $workspaceOwnerKey = atom<string | null>(null)

/** Exact route for a fresh session in the current workspace. Kept structural
 *  here so the generic pane shell does not depend on profile/gateway stores. */
export interface WorkspaceSessionRoute {
  connectionId: string
  mode?: 'local' | 'remote'
  profile: string
  targetProfile?: string
}

/** What the shared `+` / session.newTab command means in this workspace. */
export type WorkspaceNewSessionTarget =
  { kind: 'blocked'; message: string } | { kind: 'route'; route: WorkspaceSessionRoute }

/** Sessions uses its established ambient behavior (`null`). Bots publishes an
 *  exact route or a concise reason that a generic session is unavailable. */
export const $workspaceNewSessionTarget = atom<WorkspaceNewSessionTarget | null>(null)

/** One key for window-local active-pane memory. Owner keys stay opaque. */
export function workspaceScopeKey(mode: WorkspaceMode, ownerKey: string | null): string {
  return mode === 'sessions' ? 'sessions' : `bots:${ownerKey ?? ''}`
}

function sameNewSessionTarget(a: WorkspaceNewSessionTarget | null, b: WorkspaceNewSessionTarget | null): boolean {
  if (a === b) {
    return true
  }

  if (!a || !b || a.kind !== b.kind) {
    return false
  }

  if (a.kind === 'blocked' && b.kind === 'blocked') {
    return a.message === b.message
  }

  if (a.kind === 'route' && b.kind === 'route') {
    return (
      a.route.connectionId === b.route.connectionId &&
      a.route.mode === b.route.mode &&
      a.route.profile === b.route.profile &&
      a.route.targetProfile === b.route.targetProfile
    )
  }

  return false
}

/** Publish one coherent presentation and creation scope without an
 *  intermediate mixed frame. Sessions always retains its existing ambient
 *  new-session behavior; alternate workspaces must state their intent. */
export function setWorkspaceScope(
  mode: WorkspaceMode,
  ownerKey: string | null = null,
  newSessionTarget: WorkspaceNewSessionTarget | null = null
): boolean {
  const nextOwnerKey = mode === 'bots' ? ownerKey : null
  const nextNewSessionTarget = mode === 'bots' ? newSessionTarget : null

  if (
    $workspaceMode.get() === mode &&
    $workspaceOwnerKey.get() === nextOwnerKey &&
    sameNewSessionTarget($workspaceNewSessionTarget.get(), nextNewSessionTarget)
  ) {
    return false
  }

  batch(() => {
    $workspaceMode.set(mode)
    $workspaceOwnerKey.set(nextOwnerKey)
    $workspaceNewSessionTarget.set(nextNewSessionTarget)
  })

  return true
}

/**
 * The slice of {@link Contribution} metadata that scopes it to a workspace.
 * A contribution with neither field set is global: it participates in every
 * workspace, preserving pre-existing behavior.
 */
export interface WorkspaceScope {
  /** Surface this contribution belongs to. Omit for global visibility. */
  workspaceMode?: WorkspaceMode
  /** Exact opaque owner key within the `'bots'` surface. Ignored otherwise. */
  workspaceOwnerKey?: string
}

/**
 * Whether a contribution participates in the given workspace.
 *
 * - Unscoped/global (no `workspaceMode`) => always participates.
 * - Scoped with a mode mismatch => does not participate.
 * - Sessions match => participates.
 * - Bots match => participates only when the owner key is non-empty and equals
 *   the current owner key (exact string equality).
 *
 * Defaults reflect the un-switched window state when omitted.
 */
export function contributesToWorkspace(
  scope: WorkspaceScope | undefined,
  mode: WorkspaceMode = $workspaceMode.get(),
  ownerKey: string | null = $workspaceOwnerKey.get()
): boolean {
  const { workspaceMode, workspaceOwnerKey } = scope ?? {}

  if (workspaceMode == null) {
    return true
  }

  if (workspaceMode !== mode) {
    return false
  }

  if (workspaceMode === 'sessions') {
    return true
  }

  return Boolean(workspaceOwnerKey) && workspaceOwnerKey === ownerKey
}

/**
 * Filter contributions down to those participating in the given workspace,
 * preserving input order.
 *
 * Preserves reference identity on a no-op (every contribution participates),
 * so callers can hand the result straight to React without a wasted re-render.
 */
export function filterContributionsForWorkspace<T extends WorkspaceScope>(
  contributions: readonly T[],
  mode: WorkspaceMode,
  ownerKey: string | null
): readonly T[] {
  let filtered: T[] | null = null

  for (let i = 0; i < contributions.length; i += 1) {
    if (contributesToWorkspace(contributions[i], mode, ownerKey)) {
      filtered?.push(contributions[i])

      continue
    }

    filtered ??= contributions.slice(0, i)
  }

  return filtered ?? contributions
}

/**
 * Window-local memory of the active pane per exact workspace owner key.
 * Keys are opaque exact strings; similar-looking keys never collide because
 * nothing here parses them.
 */
const rememberedActivePanes = new Map<string, string>()

/** Remember which pane was active for an exact owner key. */
export function rememberActivePane(ownerKey: string, paneId: string): void {
  rememberedActivePanes.set(ownerKey, paneId)
}

/**
 * Resolve the pane to activate for an owner key against the currently eligible
 * panes. A remembered pane that has since been removed must not restore: the
 * fallback is the first eligible pane, or null when none are eligible.
 */
export function resolveRememberedActivePane(ownerKey: string, eligiblePaneIds: readonly string[]): string | null {
  const remembered = rememberedActivePanes.get(ownerKey)

  if (remembered != null && eligiblePaneIds.includes(remembered)) {
    return remembered
  }

  return eligiblePaneIds[0] ?? null
}

/** Forget the remembered pane for one owner key. */
export function forgetActivePane(ownerKey: string): void {
  rememberedActivePanes.delete(ownerKey)
}

/** Forget a pane removed from the layout, regardless of which owners used it. */
export function forgetRememberedPane(paneId: string): void {
  for (const [ownerKey, rememberedPaneId] of rememberedActivePanes) {
    if (rememberedPaneId === paneId) {
      rememberedActivePanes.delete(ownerKey)
    }
  }
}

/** Test-only: clear all remembered panes. */
export function resetRememberedActivePanes(): void {
  rememberedActivePanes.clear()
}
