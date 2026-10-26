import type { HermesConnection } from '@/global'

interface LaunchWorkspace {
  cwd: string
  profile: string
}

let launchWorkspace: LaunchWorkspace | null = null

export function applyLaunchWorkspace(workspace: LaunchWorkspace | null | undefined): void {
  launchWorkspace = workspace ?? null
}

/** Machine-local intent never follows a remote gateway or a different profile. */
export function launchWorkspaceCwd(connection: HermesConnection | null): string {
  if (
    connection?.mode !== 'local' ||
    (connection.connectionId && connection.connectionId !== 'local') ||
    (connection.profile?.trim() || 'default') !== launchWorkspace?.profile
  ) {
    return ''
  }

  return launchWorkspace.cwd
}

/** A draft can target a different owner without activating that owner's socket first. */
export function cwdForSessionOwner(
  cwd: string,
  explicit: boolean,
  configured: string,
  profile: string,
  connectionId: string
): string {
  if (!launchWorkspace || cwd !== launchWorkspace.cwd || explicit || configured) {
    return cwd
  }

  return (profile || 'default') === launchWorkspace.profile && connectionId === 'local' ? cwd : ''
}
