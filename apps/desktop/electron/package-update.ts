const FLATPAK_APP_ID = 'com.nousresearch.Hermes'
const SNAP_NAME = 'hermes-desktop'

export interface PackageUpdateInstruction {
  kind: 'flatpak' | 'snap'
  command: string
  message: string
}

export function resolvePackageUpdate(env: Record<string, string | undefined>): PackageUpdateInstruction | null {
  if (env.HERMES_DESKTOP_FLATPAK === '1' || env.FLATPAK_ID === FLATPAK_APP_ID) {
    return {
      kind: 'flatpak',
      command: `flatpak update --user ${FLATPAK_APP_ID}`,
      message: 'This Flatpak is updated through Flatpak.'
    }
  }

  if (env.HERMES_DESKTOP_SNAP === '1' || env.SNAP) {
    return {
      kind: 'snap',
      command: `sudo snap refresh ${SNAP_NAME}`,
      message: 'This Snap is updated through Snap.'
    }
  }

  return null
}
