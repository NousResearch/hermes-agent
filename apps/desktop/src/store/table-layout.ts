import type { Codec } from '@/lib/persisted'
import { persistentAtom } from '@/lib/persisted'

const KEY = 'hermes.desktop.tableLayout.v1'

export const TABLE_LAYOUTS = ['fit', 'scroll'] as const
export type TableLayout = (typeof TABLE_LAYOUTS)[number]

const codec: Codec<TableLayout> = {
  decode: raw => (TABLE_LAYOUTS.includes(raw as TableLayout) ? (raw as TableLayout) : 'fit'),
  encode: value => value
}

export const $tableLayout = persistentAtom<TableLayout>(KEY, 'fit', codec)

export function setTableLayout(layout: TableLayout): void {
  $tableLayout.set(layout)
}

if (typeof document !== 'undefined') {
  $tableLayout.subscribe(layout => {
    document.documentElement.dataset.hermesTableLayout = layout
  })
}
