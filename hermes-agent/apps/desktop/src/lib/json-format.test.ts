import { describe, expect, it } from 'vitest'

import { tryFormatJson } from './json-format'

describe('tryFormatJson', () => {
  it('pretty-prints compact JSON', () => {
    expect(tryFormatJson('{"a":1,"b":[2,3]}')).toEqual({
      ok: true,
      text: '{\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}'
    })
  })

  it('leaves empty input unchanged', () => {
    expect(tryFormatJson('   ')).toEqual({ ok: true, text: '   ' })
  })

  it('reports parse errors', () => {
    const result = tryFormatJson('{bad')

    expect(result.ok).toBe(false)

    if (!result.ok) {
      expect(result.error.length).toBeGreaterThan(0)
    }
  })
})
