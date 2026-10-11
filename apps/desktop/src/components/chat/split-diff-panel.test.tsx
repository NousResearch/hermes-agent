import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { parseDiff } from './diff-lines'
import { SplitDiffPanel } from './split-diff-panel'
import { pairDiffLines } from './split-diff-rows'

afterEach(cleanup)

describe('split diffs', () => {
  it('aligns replacements and keeps surplus lines, empty lines and hunk gaps on their own sides', () => {
    const rows = pairDiffLines(
      parseDiff(
        [
          '--- a/file',
          '+++ b/file',
          '@@ -2,4 +2,3 @@',
          '-old one',
          '---literal',
          '+new one',
          ' context',
          '-old last',
          '+',
          '\\ No newline at end of file',
          '@@ -30,0 +29,1 @@',
          '+++literal',
          ''
        ].join('\n')
      )
    )

    expect(rows.map(row => [row.before?.text, row.after?.text])).toEqual([
      ['old one', 'new one'],
      ['--literal', undefined],
      ['context', 'context'],
      ['old last', ''],
      ['', ''],
      [undefined, '++literal']
    ])
    expect(rows[1].before?.oldNo).toBe(3)
    expect(rows[2].after?.newNo).toBe(3)
    expect(rows[3].after?.newNo).toBe(4)
    expect(rows[5].after?.newNo).toBe(29)
  })

  it('windows a large diff and synchronizes vertical scrolling while horizontal scrolling stays independent', async () => {
    const diff = '@@ -1,5000 +1,5000 @@\n' + Array.from({ length: 5000 }, (_, index) => ` line ${index}`).join('\n')
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <SplitDiffPanel diff={diff} />
      </I18nProvider>
    )
    const before = screen.getByRole('region', { name: 'Before' })
    const after = screen.getByRole('region', { name: 'After' })
    expect(before.textContent).not.toContain('line 2000')
    expect(before.textContent).toContain('line 0')
    fireEvent.scroll(before, { target: { scrollTop: 40000, scrollLeft: 40 } })
    expect(after.scrollTop).toBe(40000)
    expect(after.scrollLeft).toBe(0)
    await waitFor(() => expect(before.textContent).toContain('line 2000'))
    expect(after.textContent).toContain('line 2000')
    expect(before.textContent).not.toContain('line 0')
    fireEvent.scroll(after, { target: { scrollTop: 0, scrollLeft: 80 } })
    expect(before.scrollTop).toBe(0)
    expect(before.scrollLeft).toBe(40)
    await waitFor(() => expect(before.textContent).toContain('line 0'))
  })
})
