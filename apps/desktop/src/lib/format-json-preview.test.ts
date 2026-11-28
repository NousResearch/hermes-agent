import { describe, expect, it } from 'vitest'

import { formatMinifiedJson } from './format-json-preview'

describe('formatMinifiedJson', () => {
  it('indents minified JSON losslessly (big ints, escapes, empty containers)', () => {
    const text = '{"id":12345678901234567890,"s":"a\\"b,c:{}","list":[1,{"x":[]}],"o":{}}'

    expect(formatMinifiedJson(text)).toBe(
      [
        '{',
        '  "id": 12345678901234567890,',
        '  "s": "a\\"b,c:{}",',
        '  "list": [',
        '    1,',
        '    {',
        '      "x": []',
        '    }',
        '  ],',
        '  "o": {}',
        '}'
      ].join('\n')
    )
  })

  it('leaves invalid, multi-line, scalar and oversized documents alone', () => {
    expect(formatMinifiedJson('{"a":')).toBeNull()
    expect(formatMinifiedJson('{\n"a":1}')).toBeNull()
    expect(formatMinifiedJson('{"a":1}\r{"b":2}')).toBeNull()
    expect(formatMinifiedJson('"just a string"')).toBeNull()
    expect(formatMinifiedJson('[]')).toBeNull()
    expect(formatMinifiedJson('['.repeat(80) + ']'.repeat(80))).toBeNull()
    expect(formatMinifiedJson(`[${Array.from({ length: 20_000 }, (_, i) => i).join(',')}]`)).toBeNull()
  })
})
