import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// RoutineRow narrow-pane layout contract (#91623):
// The Routines (Cronjobs) pane docks right at 250px. RoutineRow is a
// display:grid node inside the pane's grid list; grid items default to
// min-width:auto, so the row could not shrink below its min-content width.
// A nowrap job title pinned the row at ~284px inside the 250px pane, the
// overflow-hidden ScrollArea clipped the right edge, and the enable/disable
// Switch + delete button became invisible (title truncation also never
// engaged). Every grid/flex node in the row's width chain must carry
// min-w-0 so the title and metadata can actually shrink.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('RoutineRow root grid carries min-w-0 so the row can shrink inside the pane', () => {
  assert.match(source, /'group grid min-w-0 gap-1\.5 rounded-lg border border-\(--ui-stroke-secondary\) p-2\.5 transition-colors'/)
})

test('title/controls flex line carries min-w-0 (title truncate engages, Switch stays visible)', () => {
  assert.match(source, /className: 'flex min-w-0 items-center gap-2'/)
  // The old pinned-width form must not come back.
  assert.doesNotMatch(source, /jsxs\('div', \{\s*\n\s*className: 'flex items-center gap-2',\s*\n\s*children: \[\s*\n\s*jsx\('span', \{\s*\n\s*'aria-hidden': true,/)
})

test('metadata line carries min-w-0 so the next-run label can truncate, not clip', () => {
  assert.match(source, /className: 'flex min-w-0 items-center justify-between gap-2 pl-3\.5'/)
})

test('delete button is shrink-0 so the icon is never crushed in narrow panes', () => {
  assert.match(source, /'flex size-5 shrink-0 items-center justify-center rounded text-\(--ui-text-quaternary\)/)
})

test('legacy-routine warning strip carries min-w-0 (same grid item class of bug)', () => {
  assert.match(source, /'min-w-0 rounded-md border border-\(--ui-stroke-secondary\) px-2 py-1\.5 text-\[0\.65rem\] leading-4 text-\(--ui-accent\)'/)
})
