// A native path may contain a literal `#` or `?` (`C:\Music\track #07.mp3`).
// Cutting the path there drops the extension, so the file is mistyped. A real
// URL still loses its query and fragment first. Returns the lowercased
// extension without its dot.
export function pathExtension(path: string, known: (ext: string) => boolean): string | undefined {
  const extOf = (value: string): string | undefined => {
    const base = value.split(/[\\/]/).pop() ?? ''
    const dot = base.lastIndexOf('.')

    if (dot <= 0 || dot === base.length - 1) {
      return undefined
    }

    return base.slice(dot + 1).toLowerCase()
  }

  const isUrl = /^[a-z][a-z0-9+.-]*:/i.test(path) && !/^[a-z]:[\\/]/i.test(path)

  if (isUrl) {
    return extOf(path.split(/[?#]/, 1)[0] ?? '')
  }

  const literal = extOf(path)

  if (literal && known(literal)) {
    return literal
  }

  return extOf(path.split(/[?#]/, 1)[0] ?? '')
}
