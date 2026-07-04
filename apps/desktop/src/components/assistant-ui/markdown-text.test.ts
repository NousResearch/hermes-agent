import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'

describe('preprocessMarkdown', () => {
  it('strips inline accidental triple-backtick starts', () => {
    const input = [
      'Working as intended.',
      "Here's your scene: ``` http://localhost:8812/",
      '',
      '- **Multicolored cube**',
      '- **Rotates**'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    expect(output).toContain("Here's your scene:")
    expect(output).not.toContain('http://localhost:8812/')
    expect(output).toContain('- **Multicolored cube**')
  })

  it('demotes invalid fenced prose blocks with closers', () => {
    const fence = '```'

    const input = [
      `${fence} http://localhost:8812/`,
      '- **Scroll wheel** - zoom',
      '- **Right-drag/pan** - disabled',
      fence
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    expect(output).not.toContain('http://localhost:8812/')
    expect(output).toContain('- **Scroll wheel** - zoom')
  })

  it('drops fences around a preview-only URL block', () => {
    const fence = '```'
    const input = ['Server is back.', '', fence, 'http://localhost:8812/', fence].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('Server is back.')
    expect(output).not.toContain('```')
    expect(output).not.toContain('http://localhost:8812/')
  })

  it('demotes prose sentence masquerading as fence info', () => {
    const input = ['```Heads up - a bunny got added', '- Pure white (`#ffffff`)', '- Ambient dropped to 0.18'].join(
      '\n'
    )

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```heads')
    expect(output).toContain('Heads up - a bunny got added')
    expect(output).toContain('- Pure white (`#ffffff`)')
  })

  it('keeps valid code fences intact', () => {
    const fence = '```'
    const input = [`${fence}ts`, 'const value = 1;', fence].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).toContain('```ts')
    expect(output).toContain('const value = 1;')
  })

  it('keeps dangling real code fences during streaming', () => {
    const input = ['```ts', 'const value = 1;'].join('\n')
    const output = preprocessMarkdown(input)

    expect(output.startsWith('```ts')).toBe(true)
    expect(output).toContain('const value = 1;')
  })

  it('demotes dangling prose fences', () => {
    const input = ['```', '- Pure white (`#ffffff`)', '- Ambient dropped to 0.18'].join('\n')
    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    expect(output).toContain('- Pure white (`#ffffff`)')
  })

  it('autolinks raw urls in prose', () => {
    const output = preprocessMarkdown(
      'Book here:\nhttps://www.getyourguide.com/culebra-island-l145468/from-fajardo-tour-t19894/'
    )

    expect(output).toContain('<https://www.getyourguide.com/culebra-island-l145468/from-fajardo-tour-t19894/>')
  })

  it('strips orphan numeric citation markers outside code spans', () => {
    const output = preprocessMarkdown('This is the source[0], but keep `items[0]` untouched.')

    expect(output).toContain('source,')
    expect(output).not.toContain('source[0]')
    expect(output).toContain('`items[0]`')
  })

  it('demotes title/url blocks wrapped in malformed inline fences', () => {
    const input = [
      '**🚢 TOMORROW (Fajardo, crystal clear cays, pickup avail):**',
      '',
      'Icacos Full-Day Catamaran — 6hr, $140, small group, pickup```',
      'https://www.getyourguide.com/fajardo-l882/from-fajardo-icacos-island-full-day-catamaran-trip-t19891/',
      '```Sail Getaway Luxury Cat (Cordillera Cays, water slide, unlimited rum) — 6hr, $195```',
      'https://www.getyourguide.com/fajardo-l882/icacos-all-inclusive-sailing-catamaran-beach-and-snorkel-t466138/'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    expect(output).toContain('Sail Getaway Luxury Cat')
    expect(output).toContain(
      '<https://www.getyourguide.com/fajardo-l882/from-fajardo-icacos-island-full-day-catamaran-trip-t19891/>'
    )
    expect(output).toContain(
      '<https://www.getyourguide.com/fajardo-l882/icacos-all-inclusive-sailing-catamaran-beach-and-snorkel-t466138/>'
    )
  })

  it('autolinks urls glued to prices and removes orphan fence tails', () => {
    const input = [
      '**🐢 TODAY (from San Juan, no driving):**',
      '',
      'Sea Turtles & Manatees Snorkel + Free Rum — 1.5hr,',
      '~$56```https://www.getyourguide.com/san-juan-puerto-rico-l355/san-juan-snorkel-sea-turtles-manatees-free-video-rum-t879147/ Old San Juan Sunset Cruise w/ Drinks + Hotel Pickup — 1.5hr, ~$99 (drinks, no snorkel)```',
      'https://www.getyourguide.com/en-gb/san-juan-puerto-rico-l355/san-juan-old-san-juan-sunset-cruise-with-drinks-transfer-t405191/'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    // Currency dollar amounts get escaped to `\$` in the preprocessor
    // so they don't get parsed as math delimiters by remark-math (we
    // enable singleDollarTextMath, which would otherwise greedy-match
    // `$56...$99` as one big inline math span). The escape is invisible
    // to the user — `\$` renders as a literal `$` in the final output.
    expect(output).toContain(
      '~\\$56<https://www.getyourguide.com/san-juan-puerto-rico-l355/san-juan-snorkel-sea-turtles-manatees-free-video-rum-t879147/> Old San Juan Sunset Cruise'
    )
    expect(output).toContain(
      '<https://www.getyourguide.com/en-gb/san-juan-puerto-rico-l355/san-juan-old-san-juan-sunset-cruise-with-drinks-transfer-t405191/>'
    )
  })

  it('demotes url-only fenced blocks to clickable markdown links', () => {
    const input = [
      'Sea Turtles & Manatees Snorkel + Free Rum — 1.5hr, ~$56',
      '```',
      'https://www.getyourguide.com/san-juan-puerto-rico-l355/san-juan-snorkel-sea-turtles-manatees-free-video-rum-t879147/',
      '```',
      '',
      'Old San Juan Sunset Cruise w/ Drinks + Hotel Pickup — 1.5hr, ~$99',
      '```',
      'https://www.getyourguide.com/en-gb/san-juan-puerto-rico-l355/san-juan-old-san-juan-sunset-cruise-with-drinks-transfer-t405191/',
      '```'
    ].join('\n')

    const output = preprocessMarkdown(input)

    expect(output).not.toContain('```')
    expect(output).toContain(
      '<https://www.getyourguide.com/san-juan-puerto-rico-l355/san-juan-snorkel-sea-turtles-manatees-free-video-rum-t879147/>'
    )
    expect(output).toContain(
      '<https://www.getyourguide.com/en-gb/san-juan-puerto-rico-l355/san-juan-old-san-juan-sunset-cruise-with-drinks-transfer-t405191/>'
    )
  })

  it('does not swallow trailing emphasis asterisks into an autolinked url', () => {
    const input = '**PR opened: https://github.com/NousResearch/hermes-agent/pull/12345**'

    const output = preprocessMarkdown(input)

    // The URL is autolinked WITHOUT the trailing `**` glued into the href,
    // and the bold emphasis run stays intact so it renders as bold + a link.
    expect(output).toContain('<https://github.com/NousResearch/hermes-agent/pull/12345>')
    expect(output).not.toContain('pull/12345**>')
    expect(output).not.toContain('12345*')
  })

  it('stops an autolinked url at mid-string bold markers', () => {
    const input = 'See https://github.com/foo/bar**bold** for details.'

    const output = preprocessMarkdown(input)

    expect(output).toContain('<https://github.com/foo/bar>')
    expect(output).toContain('**bold**')
  })

  it('keeps underscores and tildes inside autolinked url paths', () => {
    const input = 'Docs at https://example.com/a_b/c~d/page'

    const output = preprocessMarkdown(input)

    expect(output).toContain('<https://example.com/a_b/c~d/page>')
  })

  it('handles a fenced block larger than V8 spread-argument limit', () => {
    // A single huge code block (e.g. a logged minified bundle) used to throw
    // `RangeError: Maximum call stack size exceeded` via `out.push(...lines)`.
    const body = Array.from({ length: 200_000 }, (_, i) => `line ${i}`).join('\n')
    const input = `\`\`\`js\n${body}\n\`\`\``

    expect(() => preprocessMarkdown(input)).not.toThrow()
  })

  describe('file-path linkification', () => {
    it('linkifies a Windows drive path in prose', () => {
      const output = preprocessMarkdown('I updated C:\\Users\\me\\src\\main.ts for you.')

      expect(output).toContain('[C:\\Users\\me\\src\\main.ts](file:///C:/Users/me/src/main.ts)')
    })

    it('linkifies a Unix absolute path in prose', () => {
      const output = preprocessMarkdown('See /home/me/src/main.ts for details.')

      expect(output).toContain('[/home/me/src/main.ts](file:///home/me/src/main.ts)')
    })

    it('linkifies a dot-relative path in prose', () => {
      expect(preprocessMarkdown('Config at ./config/settings.yaml')).toContain(
        '[./config/settings.yaml](file://./config/settings.yaml)'
      )
      expect(preprocessMarkdown('Parent at ../src/App.tsx')).toContain(
        '[../src/App.tsx](file://../src/App.tsx)'
      )
    })

    it('keeps a line:col suffix in the label but not the href', () => {
      const output = preprocessMarkdown('Line ref: /home/me/src/main.ts:42:10')

      // The href targets the file only (file:// can't carry line info); the
      // visible label keeps the :42:10 suffix the model wrote.
      expect(output).toContain('[/home/me/src/main.ts:42:10](file:///home/me/src/main.ts)')
    })

    it('peels trailing sentence punctuation off the path', () => {
      const output = preprocessMarkdown('Sentence end: /home/me/main.ts.')

      expect(output).toContain('[/home/me/main.ts](file:///home/me/main.ts).')
    })

    it('does not linkify paths inside inline code spans', () => {
      const output = preprocessMarkdown('See `src/components/App.tsx` for the component.')

      expect(output).not.toContain('file://')
      expect(output).toContain('`src/components/App.tsx`')
    })

    it('does not linkify paths inside fenced code blocks', () => {
      const output = preprocessMarkdown('```ts\nconst p = "/home/me/src/main.ts"\n```')

      expect(output).not.toContain('file://')
    })

    it('does not linkify bare relative paths without a dot prefix', () => {
      // Bare relative paths (src/components/App.tsx) are intentionally not
      // matched — their false-positive rate is too high. Phase 1 links prose
      // paths only when they start with a drive letter, /, ./, or ../.
      const output = preprocessMarkdown('The file src/components/App.tsx was changed.')

      expect(output).not.toContain('file://')
    })

    it('does not linkify http(s) urls as file paths', () => {
      const output = preprocessMarkdown('Docs at https://example.com/page')

      expect(output).not.toContain('file://')
      expect(output).toContain('<https://example.com/page>')
    })

    it('linkifies multiple paths in one message', () => {
      const output = preprocessMarkdown('See /a/b/c.py and /d/e/f.go for the changes.')

      expect(output).toContain('[/a/b/c.py](file:///a/b/c.py)')
      expect(output).toContain('[/d/e/f.go](file:///d/e/f.go)')
    })
  })
})
