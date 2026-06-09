const NO_WORKSPACE_KEY = 'workspace:default'

export function normalizeWorkspacePath(cwd: string): string {
  const trimmed = cwd.trim().replaceAll('\\', '/')

  if (!trimmed) {
    return ''
  }

  const normalized = trimmed.replace(/\/+$/, '') || '/'

  return normalized.replace(/^([A-Z]):/, (_, drive: string) => `${drive.toLowerCase()}:`)
}

export function workspaceKey(cwd: string): string {
  const normalized = normalizeWorkspacePath(cwd)

  return normalized ? `workspace:${normalized}` : NO_WORKSPACE_KEY
}
