import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { TabDropdown } from '@/components/ui/tab-dropdown'
import type { HermesReviewScope } from '@/global'
import { useI18n } from '@/i18n'
import { $reviewScope, clearReviewSelection, refreshReview } from '@/store/review'

function selectScope(id: HermesReviewScope) {
  $reviewScope.set(id)
  clearReviewSelection()
  void refreshReview()
}

/** Scope switcher row under the Review header. Label + tabs + icons can't share
 *  the 28px header in a pane that narrows to 10rem, so the tabs get their own
 *  row; below their natural width it collapses to the app's narrow-width tab
 *  dropdown rather than ellipsizing every option. */
export function ReviewScopeRow() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const scope = useStore($reviewScope)

  const options: { id: HermesReviewScope; label: string }[] = [
    { id: 'uncommitted', label: c.scopeUncommitted },
    { id: 'branch', label: c.scopeBranch },
    { id: 'lastTurn', label: c.scopeLastTurn }
  ]

  return (
    <div className="@container shrink-0 px-2 pb-1.5" data-suppress-pane-reveal-side="">
      <SegmentedControl<HermesReviewScope>
        className="hidden w-full auto-cols-[minmax(0,auto)] @[13.5rem]:grid [&>button]:px-2"
        onChange={selectScope}
        options={options}
        value={scope}
      />
      <div className="flex h-[1.5625rem] items-center pl-1.5 @[13.5rem]:hidden">
        <TabDropdown
          align="start"
          items={options.map(option => ({
            active: option.id === scope,
            id: option.id,
            label: option.label,
            onSelect: () => selectScope(option.id)
          }))}
        />
      </div>
    </div>
  )
}
