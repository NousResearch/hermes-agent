import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock, isLikelyProseFence, isLikelyStructuredText } from './markdown-code'

// Explicit-tag regression matrix (XpycT's four-block repro on Windows 11,
// Hermes v0.19.1): text/markdown/gdscript/yaml fences whose content looks
// prose-like (>=3 lines, no code signals) must stay code blocks. Only bare
// fences ('') still fall through to the prose heuristic.
const LOREM_3 = ['lorem ipsum dolor sit amet', 'consectetur adipiscing elit', 'sed do eiusmod tempor'].join('\n')
const SENTENCE_3 = [
  'The quarterly report shows steady growth across all sectors.',
  'Regional performance exceeded expectations by twelve percent.',
  'Customer retention rates improved for the third consecutive month.'
].join('\n')
const MARKDOWN_BLOCK = ['# Deployment notes', '', 'This block should stay fenced'].join('\n')
const YAML_LIST = ['providers:', '  - name: deepseek', '  - name: openai'].join('\n')

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

  it('keeps an SSH config block fenced (regression: rendered as flat prose)', () => {
    const ssh = ['Host 192.168.0.159', '    HostName 192.168.0.159', '    User teknium', '    Port 22'].join('\n')

    expect(isLikelyProseCodeBlock('', ssh)).toBe(false)
    expect(isLikelyProseCodeBlock('text', ssh)).toBe(false)
  })

  it('keeps a flat key-value config fenced', () => {
    expect(isLikelyProseCodeBlock('', ['Host myserver', 'User teknium', 'Port 22'].join('\n'))).toBe(false)
  })

  it('keeps an .env-style dump fenced', () => {
    expect(isLikelyProseCodeBlock('', ['API_KEY=abc123', 'PORT=8080', 'DEBUG=true'].join('\n'))).toBe(false)
  })

  it('keeps text fences as code even with zero code signals', () => {
    expect(isLikelyProseCodeBlock('text', LOREM_3)).toBe(false)
    expect(isLikelyProseCodeBlock('text', SENTENCE_3)).toBe(false)
  })

  it('keeps plain/plaintext fences as code', () => {
    expect(isLikelyProseCodeBlock('plain', LOREM_3)).toBe(false)
    expect(isLikelyProseCodeBlock('plaintext', SENTENCE_3)).toBe(false)
  })

  it('keeps markdown/md fences as code instead of rich-rendering them', () => {
    expect(isLikelyProseCodeBlock('markdown', MARKDOWN_BLOCK)).toBe(false)
    expect(isLikelyProseCodeBlock('md', SENTENCE_3)).toBe(false)
  })

  it('keeps gdscript as code (non-COMMON explicit tag)', () => {
    expect(isLikelyProseCodeBlock('gdscript', SENTENCE_3)).toBe(false)
  })

  it('keeps yaml bullet lists as code', () => {
    expect(isLikelyProseCodeBlock('yaml', YAML_LIST)).toBe(false)
  })
})

describe('isLikelyStructuredText', () => {
  it('flags indented config stanzas', () => {
    expect(isLikelyStructuredText(['Host x', '    HostName 10.0.0.1', '    Port 22'].join('\n'))).toBe(true)
  })

  it('flags flat key-value / settings listings', () => {
    expect(isLikelyStructuredText(['Host myserver', 'User teknium', 'Port 22'].join('\n'))).toBe(true)
    expect(isLikelyStructuredText(['API_KEY=abc123', 'PORT=8080', 'DEBUG=true'].join('\n'))).toBe(true)
  })

  it('does NOT flag real wrapped prose', () => {
    expect(
      isLikelyStructuredText(
        [
          'This is the first sentence of a paragraph.',
          'Here is a second line that continues the thought.',
          'And a third concluding line follows here.'
        ].join('\n')
      )
    ).toBe(false)
  })

  it('does NOT flag prose without a config shape', () => {
    expect(
      isLikelyStructuredText(
        ['the quick brown fox jumps', 'over the lazy sleeping dog', 'while the sun sets slowly'].join('\n')
      )
    ).toBe(false)
  })

  it('ignores single-line blocks', () => {
    expect(isLikelyStructuredText('Port 22')).toBe(false)
  })
})

describe('isLikelyProseFence', () => {
  it('keeps an SSH config block fenced', () => {
    const ssh = ['Host 192.168.0.159', '    HostName 192.168.0.159', '    User teknium', '    Port 22'].join('\n')

    expect(isLikelyProseFence('', ssh)).toBe(false)
    expect(isLikelyProseFence('text', ssh)).toBe(false)
  })

  it('still unwraps a plain-language paragraph fence', () => {
    expect(
      isLikelyProseFence(
        '',
        [
          'This is the first sentence of a paragraph.',
          'Here is a second line that continues the thought.',
          'And a third concluding line follows here.'
        ].join('\n')
      )
    ).toBe(true)
  })

  it('keeps text fences as code', () => {
    expect(isLikelyProseFence('text', LOREM_3)).toBe(false)
    expect(isLikelyProseFence('text', SENTENCE_3)).toBe(false)
  })

  it('keeps plain/plaintext fences as code', () => {
    expect(isLikelyProseFence('plain', LOREM_3)).toBe(false)
    expect(isLikelyProseFence('plaintext', SENTENCE_3)).toBe(false)
  })

  it('keeps markdown/md fences as code instead of rich-rendering them', () => {
    expect(isLikelyProseFence('markdown', MARKDOWN_BLOCK)).toBe(false)
    expect(isLikelyProseFence('md', SENTENCE_3)).toBe(false)
  })

  it('keeps gdscript as code (non-COMMON explicit tag)', () => {
    expect(isLikelyProseFence('gdscript', SENTENCE_3)).toBe(false)
  })

  it('keeps yaml bullet lists as code', () => {
    expect(isLikelyProseFence('yaml', YAML_LIST)).toBe(false)
  })
})
