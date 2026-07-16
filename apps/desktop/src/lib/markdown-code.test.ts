import { describe, expect, it } from 'vitest'

import { isLikelyProseCodeBlock, isLikelyProseFence } from './markdown-code'

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

  it('keeps text-labeled shell command blocks as code', () => {
    expect(
      isLikelyProseCodeBlock(
        'text',
        [
          'cd ~/Documents/',
          'bash -n install.sh',
          'brew bundle check --file=packages/Brewfile',
          'env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" SHELL=/bin/zsh TERM=xterm-256color \\',
          "/bin/zsh -lic 'command -v brew && command -v starship && command -v gh && command -v stow'"
        ].join('\n')
      )
    ).toBe(false)
  })

  it('keeps explicit plain-text file-list fences as code', () => {
    const paths = [
      'apps/desktop/src/components/assistant-ui/thread/index.tsx',
      'apps/desktop/src/components/assistant-ui/markdown-text.tsx',
      'apps/desktop/src/lib/markdown-code.ts',
      'apps/desktop/src/lib/markdown-preprocess.ts',
      'apps/desktop/src/styles.css'
    ].join('\n')

    expect(isLikelyProseFence('text', paths)).toBe(false)
    expect(isLikelyProseCodeBlock('text', paths)).toBe(false)
  })
})
