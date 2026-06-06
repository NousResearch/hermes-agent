export async function fetchResolvedProjectDir(): Promise<string> {
  try {
    const result = await window.hermesDesktop?.settings?.getResolvedProjectDir?.()

    return typeof result?.dir === 'string' ? result.dir.trim() : ''
  } catch {
    return ''
  }
}

export function compactProjectPath(path: string, max = 56): string {
  const trimmed = path.trim()

  if (trimmed.length <= max) {
    return trimmed
  }

  const head = Math.ceil((max - 1) / 2)
  const tail = max - head - 1

  return `${trimmed.slice(0, head)}…${trimmed.slice(-tail)}`
}