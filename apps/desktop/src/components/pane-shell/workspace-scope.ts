/**
 * Which workspace surface the window is looking at — sessions, or one exact
 * bot inside Bot Mode. Owner keys are opaque exact strings supplied by callers;
 * this module never parses profile names or infers connections.
 *
 * This is a SIGNAL, not a filter: consumers adapt to it — the `+` routes a new
 * session at the selected bot's profile — but nothing here decides whether a
 * pane renders. Scoping panes by workspace is how the main zone came to vanish
 * when Bot Mode's last chat closed; bot chats are ordinary tabs in the main
 * strip, beside session tabs.
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

/** Where the shared `+` / session.newTab command should aim in this workspace.
 *  `blocked` states why this owner has no route of its own (a group chat, an
 *  orphaned roster row) — it never disables the `+`, which falls back to an
 *  ordinary session: bot chats and session tabs share the one main strip. */
export type WorkspaceNewSessionTarget =
  { kind: 'blocked'; message: string } | { kind: 'route'; route: WorkspaceSessionRoute }

/** Sessions uses its established ambient behavior (`null`). */
export const $workspaceNewSessionTarget = atom<WorkspaceNewSessionTarget | null>(null)

/** Display name per exact owner key, published by the workspace that owns the
 *  key (Bot Mode: the roster's display name). Presentation only — never a
 *  session title, which for a canonical Bot Chat is an identity the backend
 *  resolves by name and must stay exactly as stored. */
export const $workspaceOwnerLabels = atom<Readonly<Record<string, string>>>({})

export function setWorkspaceOwnerLabel(ownerKey: string, label: string): void {
  if ($workspaceOwnerLabels.get()[ownerKey] !== label) {
    $workspaceOwnerLabels.set({ ...$workspaceOwnerLabels.get(), [ownerKey]: label })
  }
}

/** The caption a workspace-owned tab shows: its owner's label while the stored
 *  row still carries only the placeholder its opener registered — every bot's
 *  canonical chat is stored under the same name, so the tab reads the bot's
 *  (#99152). Any other title (a `+` side thread, a Sessions tab) is untouched. */
export function workspaceOwnerTitle(
  title: string,
  scope: { workspaceMode?: WorkspaceMode; workspaceOwnerKey?: string; workspaceTabTitle?: string } | undefined
): string {
  if (scope?.workspaceMode !== 'bots' || !scope.workspaceOwnerKey || title !== scope.workspaceTabTitle) {
    return title
  }

  return $workspaceOwnerLabels.get()[scope.workspaceOwnerKey] ?? title
}

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

/** Publish one coherent surface and creation scope without an intermediate
 *  mixed frame. Sessions always retains its existing ambient new-session
 *  behavior; alternate workspaces must state their intent. */
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
