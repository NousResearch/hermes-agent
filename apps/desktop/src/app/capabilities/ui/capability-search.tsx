import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { SearchField } from '@/components/ui/search-field'

/** The one search field every Capabilities tab uses: the boxed SearchField
 *  with the tab's name as a scope badge and the tab's own actions (add, view
 *  toggle, …) at the end. Callers own the column it sits in and the query. */
export function CapabilitySearch({
  actions,
  hints,
  onChange,
  placeholder,
  scope,
  value
}: {
  actions?: ReactNode
  hints?: string[]
  onChange: (value: string) => void
  placeholder: string
  scope: string
  value: string
}) {
  return (
    <SearchField
      containerClassName="h-10 w-full min-w-0 rounded-2xl bg-(--dt-card) py-0"
      hints={hints}
      leadingContent={<Badge size="xs" variant="muted">{scope}</Badge>}
      onChange={onChange}
      placeholder={placeholder}
      trailingAction={
        actions ? (
          <div className="flex min-w-0 flex-wrap items-center justify-end gap-3 justify-self-end" data-catalog-actions>
            {actions}
          </div>
        ) : undefined
      }
      value={value}
      variant="box"
    />
  )
}
