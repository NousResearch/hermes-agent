import { tailBoundedRemend } from '@assistant-ui/react-streamdown'
import { describe, expect, it } from 'vitest'

import { completeDirectiveTailStart, remendPreservingTrailingDirective } from '@/lib/directive-remend-guard'
import { parseTranscriptDirective } from '@/lib/transcript-directives'

/**
 * The reported bug: a Follow-up panel rendered as raw
 * `::followup{...}*` text. The prompt `Dọn các worktree wt-* đã merge`
 * carries an unpaired `*`, so the incomplete-markdown repair appended a
 * closing `*` after the directive's `}` — and core's parser, which requires
 * the directive to be the whole paragraph, stopped matching.
 *
 * These tests pin the real repair function, not a stand-in.
 */

const REPORTED =
  '::followup{p1="Bắn một đơn PRINTHUB canary mới lên Sheet và theo trọn vòng tới tracking" ' +
  'p2="Đối chiếu chênh giá 23.89 và 11.89 của đơn 550728" ' +
  'p3="Dọn các worktree wt-* đã merge để tránh lạc commit" ' +
  'p4="Thêm log có ID job và đơn cho nhịp kéo Sheet để truy vết từng lượt"}'

const guard = (text: string) => remendPreservingTrailingDirective(text, tailBoundedRemend)

describe('remendPreservingTrailingDirective', () => {
  it('reproduces the corruption the guard exists to prevent', () => {
    // Guard rail on the guard: if upstream ever stops corrupting this, the
    // tests below stop proving anything and this failure says why.
    //
    // Only the CORRUPTION is asserted, not that the parse fails. The parser
    // now forgives trailing repair debris too, so these are two independent
    // layers over one bug: the guard keeps the text byte-exact, the parser
    // survives debris from anywhere else. Asserting a null parse here would
    // couple this test to the parser being strict, and it is deliberately not.
    const repaired = tailBoundedRemend(REPORTED)

    expect(repaired).not.toBe(REPORTED)
    expect(repaired.startsWith(REPORTED)).toBe(true)
    expect(repaired.slice(REPORTED.length)).toBe('*')
  })

  it('keeps the reported directive parseable', () => {
    const out = guard(REPORTED)

    expect(out).toBe(REPORTED)

    const parsed = parseTranscriptDirective(out)

    expect(parsed?.name).toBe('followup')
    expect(parsed?.attrs.p3).toBe('Dọn các worktree wt-* đã merge để tránh lạc commit')
  })

  it.each([
    ['asterisk', 'Dọn worktree wt-* đã merge'],
    ['double asterisk', 'Chạy **npm test'],
    ['underscore', 'Sửa file _config'],
    ['backtick', 'Chạy `npm test'],
    ['strikethrough', 'Bỏ ~~cũ'],
    ['bracket', 'Mở issue [123']
  ])('survives an unpaired %s inside a prompt', (_label, prompt) => {
    const directive = `::followup{p1="${prompt}"}`

    expect(guard(directive)).toBe(directive)
    expect(parseTranscriptDirective(guard(directive))?.attrs.p1).toBe(prompt)
  })

  it('still repairs the prose above the directive', () => {
    const text = `Đang chạy **dở dang\n\n${REPORTED}`
    const out = guard(text)

    expect(out.endsWith(REPORTED)).toBe(true)
    // The paragraph above keeps its repair (the dangling ** is closed).
    expect(out.slice(0, out.length - REPORTED.length)).toContain('**dở dang**')
  })

  it('tolerates trailing whitespace after the directive', () => {
    const text = `Xong.\n\n${REPORTED}\n`

    expect(guard(text).endsWith(`${REPORTED}\n`)).toBe(true)
  })

  it('leaves a still-streaming directive to the normal repair', () => {
    const partial = '::followup{p1="Dọn worktree wt-* đã me'

    expect(completeDirectiveTailStart(partial)).toBe(-1)
    expect(guard(partial)).toBe(tailBoundedRemend(partial))
  })

  it('is a no-op for messages without a directive', () => {
    const cases = [
      'Đang chạy **dở dang',
      'Không có directive nào ở đây.',
      'Chỉ là prose có :: hai dấu hai chấm giữa câu.',
      'echo "}" # a brace-terminated line that is not a directive'
    ]

    for (const text of cases) {
      expect(guard(text)).toBe(tailBoundedRemend(text))
    }
  })

  it('does not claim a directive that is not the whole last line', () => {
    const text = 'Xem thêm ::followup{p1="x"}'

    expect(completeDirectiveTailStart(text)).toBe(-1)
  })

  it('protects any directive name, not just followup', () => {
    const directive = '::preview{file="wt-*/demo.html"}'

    expect(guard(directive)).toBe(directive)
  })
})
