import { describe, expect, it } from 'vitest'

import { canApprove, checkPreservation } from './preservation'

const missing = (original: string, candidate: string) =>
  checkPreservation(original, candidate).issues.map(issue => [issue.kind, issue.severity, issue.value])

describe('checkPreservation — exact content blocks approval when dropped', () => {
  it('requires fenced code to survive byte for byte', () => {
    const original = 'Fix this:\n```ts\nconst a = 1\nconst b = a + 1\n```\nplease'
    const kept = 'Please fix the code below.\n```typescript\nconst a = 1\nconst b = a + 1\n```'
    const altered = 'Please fix the code.\n```ts\nconst a = 1\nconst b = a+1\n```'

    expect(checkPreservation(original, kept).blocking).toBe(0)
    expect(missing(original, altered)).toContainEqual(['code-block', 'blocking', 'const a = 1\nconst b = a + 1'])
  })

  it('requires inline code, commands and @references', () => {
    const original = 'Run `npm test -- --run` and compare @file:src/app.ts with @file:`my notes.md`'

    expect(
      checkPreservation(original, 'Compare @file:src/app.ts and @file:`my notes.md`, then run `npm test -- --run`.')
        .blocking
    ).toBe(0)
    expect(missing(original, 'Run the tests and compare the two files.')).toEqual(
      expect.arrayContaining([
        ['inline-code', 'blocking', 'npm test -- --run'],
        ['reference', 'blocking', '@file:src/app.ts'],
        ['reference', 'blocking', '@file:`my notes.md`']
      ])
    )
  })

  it('requires URLs without the sentence punctuation around them', () => {
    const original = 'Read https://example.com/docs?a=1&b=2. Then summarize.'

    expect(checkPreservation(original, 'Summarize https://example.com/docs?a=1&b=2 after reading it.').blocking).toBe(0)
    expect(missing(original, 'Summarize the docs page.')).toContainEqual([
      'url',
      'blocking',
      'https://example.com/docs?a=1&b=2'
    ])
  })

  it('requires file paths but not ordinary slashed words', () => {
    const original = 'Edit apps/desktop/src/main.tsx, ./run.sh, ~/notes/todo and README.md (and/or input/output).'
    const issues = missing(original, 'Edit the entry file and the readme.')

    expect(issues).toEqual(
      expect.arrayContaining([
        ['path', 'blocking', 'apps/desktop/src/main.tsx'],
        ['path', 'blocking', './run.sh'],
        ['path', 'blocking', '~/notes/todo'],
        ['path', 'blocking', 'README.md']
      ])
    )
    expect(issues.map(([, , value]) => value)).not.toContain('and/or')
    expect(issues.map(([, , value]) => value)).not.toContain('input/output')
  })

  it('requires model, agent and identifier names', () => {
    const original = 'Use claude-opus-5-5 or gpt-5.5 (not o3) with PromptRouter; set max_turns and HERMES_HOME.'
    const values = missing(original, 'Use the best model with the swarm and set the limits.').map(([, , v]) => v)

    expect(values).toEqual(
      expect.arrayContaining(['claude-opus-5-5', 'gpt-5.5', 'o3', 'PromptRouter', 'max_turns', 'HERMES_HOME'])
    )
    expect(
      checkPreservation(
        original,
        'With PromptRouter, use claude-opus-5-5 or gpt-5.5 (not o3). Set max_turns and HERMES_HOME.'
      ).blocking
    ).toBe(0)
  })

  it('requires numbers, and a longer number does not stand in for a shorter one', () => {
    const original = 'Keep it under 200 words and list 3 options.'

    expect(missing(original, 'List three options in under 200 words.')).toContainEqual(['number', 'blocking', '3'])
    expect(missing(original, 'List 30 options in under 200 words.')).toContainEqual(['number', 'blocking', '3'])
    expect(checkPreservation(original, 'List 3 options; stay under 200 words.').blocking).toBe(0)
  })
})

describe('checkPreservation — numbers inside words', () => {
  const original =
    'Create a video with HTML and math that has a 2d/3d artstyle teaching kids about how to brush your teeth.'

  it('keeps "2d/3d" when the rewrite keeps it word for word (the reported false positive)', () => {
    const refined =
      'Create a video using HTML and math, with a 2d/3d art style, that teaches kids how to brush their teeth.'

    expect(checkPreservation(original, refined).issues).toEqual([])
  })

  it('still blocks when the rewrite drops it, naming the whole word', () => {
    expect(missing(original, 'Create an animated video that teaches kids how to brush their teeth.')).toEqual(
      expect.arrayContaining([
        ['number', 'blocking', '2d'],
        ['number', 'blocking', '3d']
      ])
    )
  })

  it('treats a unit suffix as part of the number, ignoring its case', () => {
    expect(checkPreservation('Render at 60fps in 4K.', 'Render in 4k at 60FPS.').blocking).toBe(0)
    expect(missing('Render at 60fps.', 'Render at 30fps.')).toContainEqual(['number', 'blocking', '60fps'])
  })
})

describe('checkPreservation — heuristics ask for review instead of blocking', () => {
  it('flags dropped negations and requirement words as review items', () => {
    const report = checkPreservation(
      'Do not touch the tests and only edit one file.',
      'Edit one file and update the tests.'
    )

    expect(report.blocking).toBe(0)
    expect(report.issues).toEqual(
      expect.arrayContaining([
        { kind: 'negation', severity: 'review', value: '1' },
        { kind: 'requirement', severity: 'review', value: '1' }
      ])
    )
  })

  it('accepts an equivalent negation and ignores words inside code', () => {
    expect(checkPreservation("Don't use eval.", 'Avoid eval.').review).toBe(0)
    expect(
      checkPreservation('Fix ```\nif (!ok) return not_found\n```', 'Fix ```\nif (!ok) return not_found\n```').issues
    ).toEqual([])
  })

  it('treats hyphenated slugs as review items (they look like English compounds)', () => {
    expect(missing('Load the code-review-routing skill.', 'Load the routing skill.')).toEqual([
      ['name', 'review', 'code-review-routing']
    ])
  })
})

describe('canApprove', () => {
  it('never approves with a blocking issue, even when acknowledged', () => {
    expect(canApprove(checkPreservation('Open https://a.dev', 'Open the site'), true)).toBe(false)
  })

  it('needs an acknowledgement for review items only', () => {
    const report = checkPreservation('Never push to main.', 'Push carefully.')

    expect(canApprove(report, false)).toBe(false)
    expect(canApprove(report, true)).toBe(true)
    expect(canApprove(checkPreservation('Write a haiku', 'Write a short haiku'), false)).toBe(true)
  })
})
