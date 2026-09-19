export type LocalBackendKind = 'local' | 'remote' | 'url'

export function sessionProfileSpawnSpec({
  selectedProfile,
  hermesHome,
  backendKind = 'local'
}: {
  selectedProfile: string
  hermesHome: string | null
  backendKind?: LocalBackendKind
}) {
  if (backendKind !== 'local' || !hermesHome) {
    return null
  }

  const profile = selectedProfile.trim() || 'default'

  return {
    argvProfileFlag: ['--profile', profile],
    envOverlay: { HERMES_HOME: hermesHome }
  }
}
