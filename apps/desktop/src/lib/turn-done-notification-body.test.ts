import { describe, expect, it } from 'vitest'

import { TURN_DONE_BODY_FALLBACK, TURN_DONE_BODY_MAX, turnDoneNotificationBody } from './turn-done-notification-body'

// Mirrors the minified 0.20.0 Studio bundle helper quoted in #88488:
//   Xn(n?.content || e.title || "Message complete.", 140)
// That chain lived in the released dist/client bundle, not this repo's
// source; this app's own chain never read the title (see the helper's
// comment). Kept as the "never again" guard below.
function legacyStudioBody(sessionTitle: string, assistantContent?: string): string {
  const raw = assistantContent || sessionTitle || TURN_DONE_BODY_FALLBACK

  return raw.length <= TURN_DONE_BODY_MAX ? raw : raw.slice(0, TURN_DONE_BODY_MAX)
}

describe('turnDoneNotificationBody', () => {
  it('uses the assistant reply when present', () => {
    expect(turnDoneNotificationBody('The build is green.')).toBe('The build is green.')
  })

  it('renders Markdown as one plain-text preview without damaging literal punctuation or code', () => {
    const cases = [
      ['**Yep, fixed.** 🎉', 'Yep, fixed. 🎉'],
      ['__Bold__, *italic*, _italic_, and ~~old~~.', 'Bold, italic, italic, and old.'],
      [
        '## Result\n> **Done**: `file_name.ts`\n- Read [the report](https://example.com).',
        'Result Done: file_name.ts Read the report.'
      ],
      ['1. First\n2. Second\n\n---\n\nLast', 'First Second Last'],
      ['Read [**the report**](https://example.com/a_(b) "Report").', 'Read the report.'],
      ['Read [the report][result].\n\n[result]: https://example.com', 'Read the report.'],
      ['![Build passed](https://example.com/badge.png)', 'Build passed'],
      ['![Build passed][badge]\n\n[badge]: https://example.com/badge.png', 'Build passed'],
      ['| Result | Status |\n| --- | --- |\n| Build | **Passed** |', 'Result Status Build Passed'],
      ['First  \nsecond', 'First second'],
      ['Done <!-- internal note -->**now**.', 'Done now.'],
      ['```ts\nconst file_name = "**literal**"\n```', 'const file_name = "**literal**"'],
      ['~~~sh\nls *.ts\n~~~', 'ls *.ts'],
      ['Run `echo **literal**` and `` echo `code` ``.', 'Run echo **literal** and echo `code`.'],
      ['Path /tmp/file_name.ts costs $5 * 2.', 'Path /tmp/file_name.ts costs $5 * 2.'],
      ['Use *.ts files.', 'Use *.ts files.'],
      ['Use **/*.ts files.', 'Use **/*.ts files.'],
      ['Use `*.ts` and `**/*.ts`.', 'Use *.ts and **/*.ts.'],
      ['Use \\*literal\\* and snake_case_name; 2 * 3 * 4 = 24.', 'Use *literal* and snake_case_name; 2 * 3 * 4 = 24.'],
      ['Keep an unmatched * and **unfinished', 'Keep an unmatched * and **unfinished'],
      ['**完成** — [報告](https://example.com)', '完成 — 報告'],
      ['  First\n\n  second\tthird  ', 'First second third']
    ]

    expect(cases.map(([markdown]) => turnDoneNotificationBody(markdown))).toEqual(cases.map(([, text]) => text))
  })

  it('flattens the full reply before truncation and uses localized fallback for formatting-only replies', () => {
    expect(turnDoneNotificationBody(`**${'a'.repeat(200)}**`)).toBe('a'.repeat(TURN_DONE_BODY_MAX))
    expect(turnDoneNotificationBody(`[${'b'.repeat(200)}](https://example.com)`)).toBe('b'.repeat(TURN_DONE_BODY_MAX))
    expect(turnDoneNotificationBody('## \n---')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('## \n---', 'メッセージが完了しました。')).toBe('メッセージが完了しました。')
    expect(turnDoneNotificationBody('---', '   ')).toBe(TURN_DONE_BODY_FALLBACK)
  })

  it('truncates long replies to 140 characters', () => {
    const reply = 'a'.repeat(200)
    expect(turnDoneNotificationBody(reply)).toBe('a'.repeat(TURN_DONE_BODY_MAX))
    expect(turnDoneNotificationBody(reply).length).toBe(TURN_DONE_BODY_MAX)
  })

  it('falls back to a generic phrase when content is missing, not the session title (#88488)', () => {
    const sessionTitle = "what's the weather in Seoul?"

    expect(legacyStudioBody(sessionTitle, undefined)).toBe(sessionTitle)
    expect(legacyStudioBody(sessionTitle, '')).toBe(sessionTitle)

    expect(turnDoneNotificationBody(undefined)).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('   ')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody(undefined, TURN_DONE_BODY_FALLBACK)).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('', TURN_DONE_BODY_FALLBACK)).not.toBe(sessionTitle)
  })

  // The localized fallback is the only channel the i18n copy reaches the result
  // through; every earlier fallback assertion passed TURN_DONE_BODY_FALLBACK itself,
  // so deleting the whole fallback parameter kept the suite green (review follow-up).
  // These pin the real per-locale values from the i18n files so a broken fallback
  // path or an emptied locale string flips a test.
  it('uses the caller-supplied localized fallback, not the hardcoded English one', () => {
    expect(turnDoneNotificationBody('', '消息已完成。')).toBe('消息已完成。')
    expect(turnDoneNotificationBody(null, '訊息已完成。')).toBe('訊息已完成。')
    expect(turnDoneNotificationBody('   ', 'メッセージが完了しました。')).toBe('メッセージが完了しました。')
    expect(turnDoneNotificationBody(undefined, 'اكتملت الرسالة.')).toBe('اكتملت الرسالة.')
  })

  it('never falls back to the session title', () => {
    const sessionTitle = "what's the weather in Seoul?"

    // The helper takes its fallback from the caller; the only production call site
    // (use-message-stream/index.ts) passes translateNow('...turnDoneBody'), never the
    // title. With no fallback supplied the hardcoded phrase wins over the title.
    expect(turnDoneNotificationBody(undefined)).not.toBe(sessionTitle)
    expect(turnDoneNotificationBody('', TURN_DONE_BODY_FALLBACK)).not.toBe(sessionTitle)
  })

  it('truncates a localized fallback to the same 140-character cap', () => {
    expect(turnDoneNotificationBody(undefined, '完'.repeat(200)).length).toBe(TURN_DONE_BODY_MAX)
  })

  it('ignores a blank i18n fallback and still avoids an empty body', () => {
    expect(turnDoneNotificationBody('', '')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody(null, '   ')).toBe(TURN_DONE_BODY_FALLBACK)
  })
})
