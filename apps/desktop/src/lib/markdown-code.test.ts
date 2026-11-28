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
const BULLET_2 = ['- item one', '- item two'].join('\n')

// Stone441 macOS repro (2026-08-14): long `text`/`prompt` fences with
// numbered Chinese prose were downgraded to rendered prose. Numbered lines
// ("1. 中文…") match proseLineCount's ASCII-leading regex, so they trip the
// prose heuristic; the explicit-tag guard must keep them fenced.
const CHINESE_NUMBERED = [
  '1. 这是一个较长的中文提示词，包含编号和具体步骤说明。',
  '2. 第二行继续描述用户希望保留的完整提示词内容。',
  '3. 第三行用于验证换行、编号和空格在代码块内保持原样。'
].join('\n')

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

  // Whole-class regression (spfcraze triage, 2026-08-14): non-COMMON explicit
  // tags with bullet-list bodies were prose-classified by the bullet heuristic
  // before the explicit-tag guard was reached. gdscript/zsh are non-COMMON,
  // text/plain/plaintext are NON_CODE (not COMMON) — all must stay code.
  it('keeps gdscript bullet lists as code (spfcraze repro)', () => {
    expect(isLikelyProseCodeBlock('gdscript', BULLET_2)).toBe(false)
  })

  it('keeps zsh bullet lists as code (spfcraze repro)', () => {
    expect(isLikelyProseCodeBlock('zsh', BULLET_2)).toBe(false)
  })

  it('keeps NON_CODE-family bullet lists as code (text/plain/plaintext)', () => {
    expect(isLikelyProseCodeBlock('text', BULLET_2)).toBe(false)
    expect(isLikelyProseCodeBlock('plain', BULLET_2)).toBe(false)
    expect(isLikelyProseCodeBlock('plaintext', BULLET_2)).toBe(false)
  })

  // Stone441 macOS repro: numbered Chinese prose inside text/prompt fences
  // must stay code blocks (the numbered lines trip the prose heuristic).
  it('keeps text fences with numbered Chinese prose as code (Stone441)', () => {
    expect(isLikelyProseCodeBlock('text', CHINESE_NUMBERED)).toBe(false)
  })

  it('keeps prompt fences with numbered Chinese prose as code (Stone441)', () => {
    expect(isLikelyProseCodeBlock('prompt', CHINESE_NUMBERED)).toBe(false)
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

  // Whole-class regression (spfcraze triage, 2026-08-14): same bullet-list
  // coverage on the fence layer — non-COMMON explicit tags must stay code.
  it('keeps gdscript/zsh bullet-list fences as code', () => {
    expect(isLikelyProseFence('gdscript', BULLET_2)).toBe(false)
    expect(isLikelyProseFence('zsh', BULLET_2)).toBe(false)
  })

  it('keeps NON_CODE-family bullet-list fences as code', () => {
    expect(isLikelyProseFence('text', BULLET_2)).toBe(false)
    expect(isLikelyProseFence('plain', BULLET_2)).toBe(false)
    expect(isLikelyProseFence('plaintext', BULLET_2)).toBe(false)
  })

  // Stone441 macOS repro: text/prompt fences with numbered Chinese prose.
  it('keeps text/prompt fences with numbered Chinese prose as code (Stone441)', () => {
    expect(isLikelyProseFence('text', CHINESE_NUMBERED)).toBe(false)
    expect(isLikelyProseFence('prompt', CHINESE_NUMBERED)).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// A real language tag must stay code even when the body mixes a bullet-looking
// line with a backtick or bold span. That shape is ordinary in real blocks
// (PowerShell parameter lines, diff context lines, install notes), but the
// bullet+markdown exception — which exists only for Streamdown's pseudo-tags
// like "heads" — used to swallow every tag outside COMMON_CODE_LANGUAGES.
// Reported by pxxD1998 and swarthyplacebo on 2026-11-05 against v0.22.7
// (issue #141521), and confirmed as a whole class on a local build.
const BULLET_WITH_INLINE_CODE = ['- Run the installer first', 'Then use `npm install` to set it up'].join('\n')
const BULLET_WITH_BOLD = ['- Note: **important** step here', '- Follow the downgrade guide'].join('\n')

const REAL_LANGUAGE_TAGS = [
  'powershell',
  'ps1',
  'pwsh',
  'ps',
  'shell',
  'console',
  'bat',
  'cmd',
  'batch',
  'dockerfile',
  'ini',
  'toml',
  'nginx',
  'log',
  'output',
  'kotlin',
  'lua',
  'dart',
  'zsh',
  'fish',
  'text',
  'plain',
  'plaintext',
  'prompt',
  'patch',
  'gitdiff'
]

describe('real language tags with bullet + inline markdown stay code', () => {
  it.each(REAL_LANGUAGE_TAGS)('keeps %s fences with an inline-code bullet as code', tag => {
    expect(isLikelyProseCodeBlock(tag, BULLET_WITH_INLINE_CODE)).toBe(false)
    expect(isLikelyProseFence(tag, BULLET_WITH_INLINE_CODE)).toBe(false)
  })

  it.each(REAL_LANGUAGE_TAGS)('keeps %s fences with a bold bullet as code', tag => {
    expect(isLikelyProseCodeBlock(tag, BULLET_WITH_BOLD)).toBe(false)
    expect(isLikelyProseFence(tag, BULLET_WITH_BOLD)).toBe(false)
  })

  // The exception still has to hold for pseudo-tags: the upstream test locks
  // "heads" as prose, and it must keep winning on that one tag.
  it('still classifies a pseudo-tag with bullet + emphasis as prose (upstream contract)', () => {
    expect(isLikelyProseCodeBlock('heads', BULLET_WITH_INLINE_CODE)).toBe(true)
    expect(isLikelyProseCodeBlock('heads', BULLET_WITH_BOLD)).toBe(true)
  })

  // pxxD1998 (2026-11-05): diff context lines start with " - ", and the added
  // line carries bold text, which together tripped the bullet-prose branch.
  it('keeps a diff fence whose context lines look like markdown bullets (pxxD1998)', () => {
    const DIFF = [
      '--- a/SKILL.md',
      '+++ b/SKILL.md',
      '@@ -1,3 +1,6 @@',
      ' - existing bullet',
      '-version: 1.13.3',
      '+version: 1.13.4',
      '+**New rule**: keep this fenced.'
    ].join('\n')

    expect(isLikelyProseCodeBlock('diff', DIFF)).toBe(false)
    expect(isLikelyProseFence('diff', DIFF)).toBe(false)
  })

  // kvnloo (2026-11-05): a PowerShell cmdlet block must keep its copy box
  // rather than demoting to prose and letting $vars reach the math renderer.
  it('keeps a PowerShell cmdlet fence as code (issue #141521)', () => {
    const PS = ['Get-ChildItem -Path C:\\Temp', 'Stop-Process -Name foo', 'Write-Host done'].join('\n')

    expect(isLikelyProseCodeBlock('powershell', PS)).toBe(false)
    expect(isLikelyProseFence('powershell', PS)).toBe(false)
  })

  // #141521's own matrix: two-line blocks under every tag the reporter tried.
  it('keeps short two-line PowerShell blocks as code for every reported tag', () => {
    const SHORT = ['Write-Host "T1"', 'Get-Date'].join('\n')

    for (const tag of ['powershell', 'ps1', 'pwsh', 'ps', 'shell', 'console', 'PowerShell', 'bat']) {
      expect(isLikelyProseCodeBlock(tag, SHORT)).toBe(false)
      expect(isLikelyProseFence(tag, SHORT)).toBe(false)
    }
  })

  // Cross-check against the alias list in #58023 (danspicytaco). That PR fixes
  // the same shell aliases by adding them to COMMON_CODE_LANGUAGES; this PR
  // must reach the same outcome without extending that set.
  it('covers every shell alias listed by #58023 without extending COMMON_CODE_LANGUAGES', () => {
    const SHELL_ALIASES = ['powershell', 'ps1', 'shell', 'console', 'cmd', 'fish', 'zsh']
    const BODY = ['- check the gateway status first', 'then `systemctl restart hermes`'].join('\n')

    for (const tag of SHELL_ALIASES) {
      expect(isLikelyProseCodeBlock(tag, BODY)).toBe(false)
      expect(isLikelyProseFence(tag, BODY)).toBe(false)
    }
  })
})
