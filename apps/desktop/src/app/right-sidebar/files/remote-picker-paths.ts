export interface RemotePathCrumb {
  label: string
  path: string
}

function clean(path: string) {
  if (/^[A-Za-z]:[\\/]$/.test(path)) {
    return path
  }

  return path.replace(/[\\/]+$/, '') || '/'
}

export function buildRemotePathCrumbs(currentPath: string): RemotePathCrumb[] {
  const value = clean(currentPath)
  const separator = value.includes('\\') && !value.includes('/') ? '\\' : '/'
  const parts = value.split(/[\\/]+/).filter(Boolean)
  const drive = /^[A-Za-z]:[\\/]/.test(value) ? parts.shift() : null
  const out = [{ label: drive ? `${drive}${separator}` : '/', path: drive ? `${drive}${separator}` : '/' }]
  let acc = drive ? `${drive}${separator}` : value.startsWith('\\\\') ? '\\\\' : value.startsWith('/') ? '/' : ''

  for (const part of parts) {
    acc = acc && !/[\\/]$/.test(acc) ? `${acc}${separator}${part}` : `${acc}${part}`
    out.push({ label: part, path: acc })
  }

  return out
}
