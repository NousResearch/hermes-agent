import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock, isLikelyProseFence, isLikelyStructuredText } from './markdown-code'

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

  it('keeps zsh command blocks as code even when they look like prose lines', () => {
    expect(
      isLikelyProseCodeBlock(
        'zsh',
        [
          'cd ~/Documents/dan-personal',
          'bash -n install.sh',
          'brew bundle check --file=packages/Brewfile',
          'env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" SHELL=/bin/zsh TERM=xterm-256color \\',
          "  /bin/zsh -lic 'command -v brew && command -v starship && command -v gh && command -v stow'"
        ].join('\n')
      )
    ).toBe(false)
  })

  it('keeps explicit plain-text fence tags as code by author intent', () => {
    const shell = [
      'cd ~/Documents/',
      'bash -n install.sh',
      'brew bundle check --file=packages/Brewfile',
      'env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" SHELL=/bin/zsh TERM=xterm-256color \\',
      "/bin/zsh -lic 'command -v brew && command -v starship && command -v gh && command -v stow'"
    ].join('\n')

    for (const tag of ['text', 'plain', 'plaintext']) {
      expect(isLikelyProseCodeBlock(tag, shell)).toBe(false)
    }
  })

  it('keeps explicit plain-text file-list fences as code', () => {
    const paths = [
      'apps/desktop/src/components/assistant-ui/thread/index.tsx',
      'apps/desktop/src/components/assistant-ui/markdown-text.tsx',
      'apps/desktop/src/lib/markdown-code.ts',
      'apps/desktop/src/lib/markdown-preprocess.ts',
      'apps/desktop/src/styles.css'
    ].join('\n')

    for (const tag of ['text', 'plain', 'plaintext']) {
      expect(isLikelyProseFence(tag, paths)).toBe(false)
      expect(isLikelyProseCodeBlock(tag, paths)).toBe(false)
    }
  })

  it('still demotes untagged prose that starts with shell-like English words', () => {
    const prose = [
      'Make sure to invite everyone before Friday.',
      'Command the team to use the shared checklist.',
      'Git ready after the lunch update.'
    ].join('\n')

    expect(isLikelyProseFence('', prose)).toBe(true)
    expect(isLikelyProseCodeBlock('', prose)).toBe(true)
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
})
