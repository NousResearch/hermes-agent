import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock } from './markdown-code'

describe('isLikelyProseCodeBlock', () => {
  it('detects prose that Streamdown mislabels as an unknown language', () => {
    expect(
      isLikelyProseCodeBlock(
        'heads',
        [
          '- Pure white (`#ffffff`), roughness 0.55, no emissive',
          '- Black wireframe edges at 35% opacity',
          '',
          'Want the bunny gone, or want me to keep riffing on it?'
        ].join('\n')
      )
    ).toBe(true)
  })

  it('keeps real code blocks', () => {
    expect(isLikelyProseCodeBlock('ts', 'const value = { bunny: true };\nreturn value')).toBe(false)
  })

  it('keeps diagram-like text blocks preformatted', () => {
    const diagram = ['User', ' │', ' ├─ Desktop', ' └─ Terminal', ' │', ' ▼', 'Runtime', ' │', ' ▼', 'Model'].join(
      '\n'
    )

    expect(isLikelyProseCodeBlock('text', diagram)).toBe(false)
  })

  it('keeps labeled plain-ASCII branches preformatted without standalone connector rows', () => {
    const diagram = ['User', '+- Desktop', '`- Terminal', 'Runtime', '+--> Tools', 'Model'].join('\n')

    expect(isLikelyProseCodeBlock('text', diagram)).toBe(false)
  })

  it('keeps arbitrary-length plain-ASCII branches preformatted', () => {
    const diagram = ['User', '+----- Desktop', '`----- Terminal', 'Runtime', 'Model'].join('\n')

    expect(isLikelyProseCodeBlock('text', diagram)).toBe(false)
  })

  it('does not mistake inline Unicode arrows in prose for diagram rows', () => {
    const prose = ['The request moves from client → server.', 'The response moves from server → client.', 'Both remain prose.'].join(
      '\n'
    )

    expect(isLikelyProseCodeBlock('heads', prose)).toBe(true)
  })

  it('recognizes CRLF-terminated ASCII connector rows', () => {
    const diagram = ['User', '+- Desktop', '`- Terminal', 'Runtime'].join('\r\n')

    expect(isLikelyProseCodeBlock('text', diagram)).toBe(false)
  })
})
