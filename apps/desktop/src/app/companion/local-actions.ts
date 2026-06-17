export interface CompanionAction {
  id: string
  kind: 'open-path'
  label: string
  target: string
}

function pathLeaf(path: string): string {
  const normalized = path.replace(/[\\/]+$/g, '')
  const segments = normalized.split(/[\\/]/)

  return segments.at(-1) || path
}

export function buildFileSearchActions(paths: string[]): CompanionAction[] {
  return paths.slice(0, 4).map((path, index) => ({
    id: `open-${index}`,
    kind: 'open-path',
    label: `打开 ${pathLeaf(path)}`,
    target: path
  }))
}

export function filePathToExternalUrl(path: string): string {
  const normalized = path.replace(/\\/g, '/')

  if (/^[a-zA-Z]:\//.test(normalized)) {
    return encodeURI(`file:///${normalized}`)
  }

  return encodeURI(`file://${normalized.startsWith('/') ? '' : '/'}${normalized}`)
}
