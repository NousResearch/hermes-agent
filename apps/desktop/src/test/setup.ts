// jsdom (v29) does not implement the CSSOM `CSS` namespace, but Electron and
// every target browser do. Timeline/cron code calls CSS.escape() to build
// attribute selectors, so give the test environment the spec-compliant
// polyfill (https://drafts.csswg.org/cssom/#the-css.escape()-method).
if (typeof globalThis.CSS === 'undefined' || typeof globalThis.CSS.escape !== 'function') {
  const cssEscape = (value: string): string => {
    const str = String(value)
    const { length } = str
    let result = ''

    for (let index = 0; index < length; index++) {
      const codeUnit = str.charCodeAt(index)

      // NULL → REPLACEMENT CHARACTER
      if (codeUnit === 0x0000) {
        result += '\uFFFD'
        continue
      }

      const isControl = (codeUnit >= 0x0001 && codeUnit <= 0x001f) || codeUnit === 0x007f
      const isLeadingDigit =
        codeUnit >= 0x0030 && codeUnit <= 0x0039 && (index === 0 || (index === 1 && str.charCodeAt(0) === 0x002d))

      if (isControl || isLeadingDigit) {
        result += `\\${codeUnit.toString(16)} `
        continue
      }

      // A lone leading "-"
      if (index === 0 && length === 1 && codeUnit === 0x002d) {
        result += `\\${str.charAt(index)}`
        continue
      }

      const isSafe =
        codeUnit >= 0x0080 ||
        codeUnit === 0x002d || // -
        codeUnit === 0x005f || // _
        (codeUnit >= 0x0030 && codeUnit <= 0x0039) || // 0-9
        (codeUnit >= 0x0041 && codeUnit <= 0x005a) || // A-Z
        (codeUnit >= 0x0061 && codeUnit <= 0x007a) // a-z

      result += isSafe ? str.charAt(index) : `\\${str.charAt(index)}`
    }

    return result
  }

  const existing = (globalThis as { CSS?: object }).CSS ?? {}
  ;(globalThis as { CSS: object }).CSS = Object.assign(existing, { escape: cssEscape })

  if (typeof window !== 'undefined') {
    ;(window as unknown as { CSS: object }).CSS = (globalThis as { CSS: object }).CSS
  }
}

export {}
