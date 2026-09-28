// @vitest-environment node
import fs from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))
const stylesheet = fs.readFileSync(resolve(SRC, 'styles.css'), 'utf8')

type Rule = { selector: string; body: string }

/** Every `selector { body }` block in the sheet. Braces never nest here (the
 *  only nesting is functional — `:is()`, `color-mix()` — so it stays flat). */
const rules: Rule[] = [...stylesheet.matchAll(/([^{}]*)\{([^{}]*)\}/g)]
  .map(([, selector, body]) => ({ selector: selector.replace(/\s+/g, ' ').trim(), body }))
  .filter(rule => rule.selector.length > 0)

const frameRules = rules.filter(rule => rule.selector.includes('[data-chat-transcript-frame]'))

/** The `selector { body }` block whose selector mentions the transcript frame. */
const transcriptRule = frameRules[0] ?? null

/** The pane's own fill rule — the frame directly under a translucency mode. */
const panelRule = frameRules.find(rule => /background:/.test(rule.body))
const panelDeclaration = panelRule?.body.match(/background:\s*([^;]+);/)?.[1] ?? ''

/** The top-level arguments of the first `color-mix()` in `text` — commas
 *  inside a nested call don't split. */
const colorMixArguments = (text: string): string[] | null => {
  const marker = 'color-mix('
  const start = text.indexOf(marker)

  if (start === -1) {
    return null
  }

  const args: string[] = []
  let current = ''
  let depth = 0

  for (const char of text.slice(start + marker.length)) {
    if (char === '(') {
      depth += 1
    } else if (char === ')') {
      if (depth === 0) {
        break
      }

      depth -= 1
    } else if (char === ',' && depth === 0) {
      args.push(current)
      current = ''

      continue
    }

    current += char
  }

  args.push(current)

  return args
}

/** How much of the wallpaper the fill still covers: the percentage of the mix's
 *  first (painted) stop times, recursively, whatever that stop itself covers.
 *  86% of the chrome straight into `transparent` = 0.86; a 95% tint wrapped in
 *  a 96% mix = 0.912. */
const coverage = (declaration: string): number => {
  const args = colorMixArguments(declaration)

  if (!args) {
    return 0
  }

  const first = args[1] ?? ''
  const percentage = first.match(/([\d.]+)%\s*$/)
  const own = percentage ? Number(percentage[1]) / 100 : 1
  const stop = first.replace(/([\d.]+)%\s*$/, '').trim()

  const underneath = stop.includes('color-mix(')
    ? coverage(stop)
    : stop.includes('var(--ui-bg-chrome)') && !stop.includes('transparent')
      ? 1
      : 0

  return own * underneath
}

const ruleMatching = (selector: RegExp) => frameRules.find(rule => selector.test(rule.selector))

const remOf = (rule: Rule | undefined, property = 'font-size') => {
  const value = rule?.body.match(new RegExp(`${property}:\\s*([\\d.]+)rem`))?.[1]

  return value === undefined ? Number.NaN : Number(value)
}

describe('glass chat transcript surface', () => {
  it('owns a tinted, framed and blurred fill instead of exposing the window backdrop', () => {
    expect(transcriptRule).not.toBeNull()
    const { body, selector } = transcriptRule!

    expect(body).toMatch(/background:\s*color-mix\([^;]+var\(--ui-bg-chrome\)\s+(?:[7-9]\d|100)%/)
    expect(body).not.toMatch(/background(?:-color)?:\s*transparent/)
    expect(body).toMatch(/border:\s*1px\s+solid\s+var\(--ui-stroke-tertiary\)/)
    expect(body).toMatch(/border-radius:\s*var\(--radius-/)
    expect(body).toMatch(/backdrop-filter:\s*blur\(/)
    expect(body).toMatch(/box-shadow:[^;]*inset\s+0\s+1px\s+0/)
    expect(selector).toContain(':root[data-hermes-glass]')
  })

  it('frames the transcript in clear mode too, where native opacity also reveals the desktop', () => {
    expect(transcriptRule?.selector).toContain(':root[data-hermes-clear]')
  })

  it('covers the wallpaper under BOTH translucency modes (glass and clear)', () => {
    // A rule that only names one mode leaves the transcript bare on the user's
    // wallpaper in the other — regressing either selector must go red here.
    const paneSelectors = frameRules.map(rule => rule.selector).join(' ')

    expect(paneSelectors).toContain(':root[data-hermes-glass] [data-chat-transcript-frame]')
    expect(paneSelectors).toContain(':root[data-hermes-clear] [data-chat-transcript-frame]')
    expect(panelRule).toBeDefined()
    expect(panelRule?.selector).toContain(':root[data-hermes-glass]')
    expect(panelRule?.selector).toContain(':root[data-hermes-clear]')
  })

  it('keeps the pane at 90%+ coverage — glass you can read through is not readable', () => {
    // Self-check first: the parser must read a nested tint as a product, or the
    // guard below could pass on a formula that never looks at the real rule.
    expect(coverage('color-mix(in srgb, var(--ui-bg-chrome) 86%, transparent)')).toBeCloseTo(0.86, 4)
    expect(
      coverage(
        'color-mix(in srgb, color-mix(in srgb, var(--ui-bg-chrome) 95%, var(--ui-text-primary) 5%) 96%, transparent)'
      )
    ).toBeCloseTo(0.912, 4)

    expect(coverage(panelDeclaration)).toBeGreaterThanOrEqual(0.9)
    // ...and the chrome tint alone stays near-opaque too, so re-nesting the mix
    // (or dropping the shell) cannot trade the coverage away.
    expect(panelDeclaration).toMatch(/var\(--ui-bg-chrome\)\s+(?:9\d|100)%/)
  })

  it('paints the pane with !important, ahead of the utility that clears it', () => {
    // Two paints wipe this fill and both used to win:
    //  - the frame element's own Tailwind `bg-(--ui-chat-surface-background)`,
    //    which resolves to `transparent` under [data-hermes-glass] (utilities
    //    layer, later than the base rule);
    //  - the Desktop dashboard's wholesale clear of that class in
    //    app/jarvis/core.css, which outranks the base rule on specificity.
    // Without the flag the pane is transparent in exactly the reported case
    // (light + glass, transcript bare on the wallpaper behind the orb).
    expect(panelDeclaration).toMatch(/!important/)
  })

  it('paints the pane with ink too, never leaving raw text on the material', () => {
    expect(panelRule?.body).toMatch(/color:\s*var\(--ui-text-primary\)/)
  })

  it('stands the pane down while the empty-session home hero owns the surface', () => {
    // The surface marks the frame with `data-chat-transcript-empty` while the
    // Desktop home hero shows. A session with NOTHING to read must not paint
    // the pane at all — no fill, blur, border or shadow over the wallpaper.
    // Every mode selector has to carry the guard: leaving one out repaints the
    // rectangle the user reported (light+glass, hero behind a glass slab).
    //
    // Read the selectors straight from the sheet: the flat rule parser above
    // drags the preceding comment's braces into the selector string, so the
    // exact-string check goes to the source.
    const paneSelectorPair = stylesheet.match(
      /(:root\[data-hermes-glass\]\s+\[data-chat-transcript-frame\][^,{]*,\s*:root\[data-hermes-clear\]\s+\[data-chat-transcript-frame\][^,{]*)\s*\{/
    )?.[1]

    const selectors = (paneSelectorPair ?? '')
      .split(',')
      .map(part => part.trim())
      .filter(Boolean)

    expect(selectors).toEqual([
      ':root[data-hermes-glass] [data-chat-transcript-frame]:not([data-chat-transcript-empty])',
      ':root[data-hermes-clear] [data-chat-transcript-frame]:not([data-chat-transcript-empty])'
    ])
  })

  it('keeps the pane (fill + blur) for a session that has content', () => {
    // The other direction: a message, a delegation report or a tool result
    // must still land on the readable pane. The empty guard is a `:not`
    // attribute test, never an ancestor/sibling condition that would also
    // silence the content case.
    expect(panelRule).toBeDefined()
    expect(panelDeclaration).toMatch(/var\(--ui-bg-chrome\)\s+(?:9\d|100)%/)
    expect(panelRule?.body).toMatch(/backdrop-filter:\s*blur\([\d.]+rem\)/)
    expect(coverage(panelDeclaration)).toBeGreaterThanOrEqual(0.9)
  })

  it('does not change the global glass painter or terminal surface contract', () => {
    const rootGlassRule = stylesheet.match(/:root\[data-hermes-glass\]\s*\{([^}]*)\}/)?.[1]

    expect(rootGlassRule).toContain('--ui-chat-surface-background: transparent')
    expect(rootGlassRule).toContain('--ui-terminal-surface-background: var(--ui-bg-chrome)')
  })
})

describe('transcript reading scale', () => {
  /** The frame-scoped token block that drives the whole conversation. */
  const scaleRule = frameRules.find(rule => rule.body.includes('--conversation-text-font-size'))
  const scaffoldRule = frameRules.find(rule => rule.body.includes('--conversation-scaffold-text'))

  it('drives the conversation from the frame, smaller than the app default', () => {
    expect(scaleRule).toBeDefined()
    // The app default is 13px/18px (`:root` --conversation-text-font-size).
    expect(remOf(scaleRule, '--conversation-text-font-size')).toBeLessThanOrEqual(0.75)
    expect(remOf(scaleRule, '--conversation-line-height')).toBeLessThanOrEqual(1)
    // The prose leaves carry `text-[length:var(--conversation-text-font-size)]`,
    // so the token has to actually be applied to the surface as well.
    expect(scaleRule?.body).toMatch(/font-size:\s*var\(--conversation-text-font-size\)/)
    expect(scaleRule?.body).toMatch(/--paragraph-gap:\s*[\d.]+rem/)
    expect(remOf(scaleRule, '--paragraph-gap')).toBeLessThanOrEqual(0.5)
    expect(remOf(scaleRule, '--turn-block-gap')).toBeLessThanOrEqual(0.5)
  })

  it('tightens prose leading below the 1.55 document default', () => {
    const proseRule = ruleMatching(/\[data-chat-transcript-frame\] \.aui-md,/)

    expect(proseRule).toBeDefined()
    const leading = Number(proseRule?.body.match(/line-height:\s*([\d.]+)/)?.[1] ?? '99')

    expect(leading).toBeLessThanOrEqual(1.45)
  })

  it('keeps headings muted and at pane scale instead of display type', () => {
    const headingRule = ruleMatching(/:is\(h1, h2\)/)
    const subHeadingRule = ruleMatching(/:is\(h3, h4, h5, h6\)/)

    expect(headingRule).toBeDefined()
    expect(subHeadingRule).toBeDefined()
    expect(remOf(headingRule)).toBeLessThanOrEqual(0.875)
    expect(remOf(subHeadingRule)).toBeLessThanOrEqual(0.8125)
    expect(headingRule?.body).toMatch(/font-weight:\s*600/)
    expect(headingRule?.body).toMatch(/color:\s*var\(--ui-text-secondary\)/)
    expect(subHeadingRule?.body).toMatch(/color:\s*var\(--ui-text-secondary\)/)
  })

  it('keeps lists compact — tight rows and a tighter gutter', () => {
    const listRule = ruleMatching(/:is\(ul, ol\)/)
    const rowRule = ruleMatching(/\.aui-md li/)

    expect(listRule).toBeDefined()
    expect(rowRule).toBeDefined()
    expect(remOf(listRule, 'padding-inline-start')).toBeLessThanOrEqual(1.125)
    expect(rowRule?.body).toMatch(/margin-block:\s*[\d.]+rem/)
  })

  it('scales code and tables to the pane, never above it', () => {
    const fenceRule = ruleMatching(/:is\(pre, \.aui-shiki/)
    const cellRule = ruleMatching(/:is\(th, td\)/)

    expect(fenceRule).toBeDefined()
    expect(cellRule).toBeDefined()
    expect(remOf(fenceRule)).toBeLessThanOrEqual(0.75)
    expect(remOf(cellRule)).toBeLessThanOrEqual(0.75)
    expect(fenceRule?.body).toMatch(/max-width:\s*100%/)
    expect(fenceRule?.body).toMatch(/overflow-x:\s*auto/)
  })

  it('lets nothing leave the pane sideways — fences and tables scroll, prose wraps', () => {
    const tableRules = frameRules.filter(rule => /aui-md-table|:is\(th, td\)/.test(rule.selector))
    const linkRule = ruleMatching(/\.aui-md a\b/)
    const breakRule = ruleMatching(/figcaption/)

    expect(tableRules.some(rule => /overflow-x:\s*auto/.test(rule.body))).toBe(true)
    expect(tableRules.some(rule => /max-width:\s*100%/.test(rule.body))).toBe(true)
    expect(linkRule?.body).toMatch(/overflow-wrap:\s*anywhere/)
    expect(breakRule?.body).toMatch(/overflow-wrap:\s*break-word/)
  })

  it('lifts the muted scaffolding ink so reasoning clears AA on the pane', () => {
    expect(scaffoldRule).toBeDefined()

    const scaffoldInk = Number(
      scaffoldRule?.body.match(/--conversation-scaffold-text:[^;]*var\(--ui-base\)\s+(\d+)%/)?.[1] ?? '0'
    )

    // Left at the app's 64% the faded row sits around 2.6:1 on a light pane.
    expect(scaffoldInk).toBeGreaterThanOrEqual(75)
    expect(scaffoldInk).toBeLessThanOrEqual(100)

    const fadeRule = frameRules.find(rule => /\[data-conversation-scaffold\]/.test(rule.selector))

    expect(fadeRule?.selector).toContain(':root[data-hermes-glass]')
    expect(fadeRule?.selector).toContain(':root[data-hermes-clear]')
    const opacity = Number(fadeRule?.body.match(/opacity:\s*([\d.]+)/)?.[1] ?? '0')

    expect(opacity).toBeGreaterThanOrEqual(0.8)
    expect(opacity).toBeLessThan(1)
  })
})