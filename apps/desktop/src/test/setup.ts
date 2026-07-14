class MemoryStorage implements Storage {
  private readonly entries = new Map<string, string>()

  get length(): number {
    return this.entries.size
  }

  clear(): void {
    this.entries.clear()
  }

  getItem(key: string): null | string {
    return this.entries.get(String(key)) ?? null
  }

  key(index: number): null | string {
    return [...this.entries.keys()][index] ?? null
  }

  removeItem(key: string): void {
    this.entries.delete(String(key))
  }

  setItem(key: string, value: string): void {
    this.entries.set(String(key), String(value))
  }
}

// Node 26 exposes an unusable global localStorage unless the process receives
// --localstorage-file. A fresh in-memory instance per Vitest file keeps renderer
// tests browser-shaped without sharing state between workers. Individual tests
// may still reload modules and observe persistence within their own file.
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  value: new MemoryStorage()
})

function cssEscape(value: string): string {
  const input = String(value)
  let result = ''
  let index = -1
  const firstCodeUnit = input.charCodeAt(0)

  while (++index < input.length) {
    const codeUnit = input.charCodeAt(index)

    if (codeUnit === 0) {
      result += '\uFFFD'
    } else if (
      (codeUnit >= 1 && codeUnit <= 31) ||
      codeUnit === 127 ||
      (index === 0 && codeUnit >= 48 && codeUnit <= 57) ||
      (index === 1 && codeUnit >= 48 && codeUnit <= 57 && firstCodeUnit === 45)
    ) {
      result += `\\${codeUnit.toString(16)} `
    } else if (index === 0 && codeUnit === 45 && input.length === 1) {
      result += '\\-'
    } else if (
      codeUnit >= 128 ||
      codeUnit === 45 ||
      codeUnit === 95 ||
      (codeUnit >= 48 && codeUnit <= 57) ||
      (codeUnit >= 65 && codeUnit <= 90) ||
      (codeUnit >= 97 && codeUnit <= 122)
    ) {
      result += input.charAt(index)
    } else {
      result += `\\${input.charAt(index)}`
    }
  }

  return result
}

const css = globalThis.CSS ?? ({} as typeof CSS)

if (!globalThis.CSS) {
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    value: css
  })
}

if (typeof css.escape !== 'function') {
  Object.defineProperty(css, 'escape', {
    configurable: true,
    value: cssEscape
  })
}
