import { useStore } from '@nanostores/react'

import { FileDiffPanel } from '@/components/chat/diff-lines'
import { SplitDiffPanel } from '@/components/chat/split-diff-panel'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { $reviewDiffLayout, type ReviewDiffLayout } from '@/store/review-layout'

export function ReviewDiffLayoutControl() {
  const { t } = useI18n()
  const layout = useStore($reviewDiffLayout)
  const c = t.statusStack.coding

  return (
    <div className="shrink-0 px-2.5 pb-1.5" data-suppress-pane-reveal-side="">
      <SegmentedControl<ReviewDiffLayout>
        onChange={value => $reviewDiffLayout.set(value)}
        options={[
          { id: 'unified', label: c.diffUnified },
          { id: 'split', label: c.diffSplit }
        ]}
        value={layout}
      />
    </div>
  )
}

interface ReviewDiffProps {
  diff: string
  path: string
}

export function ReviewDiff({ diff, path }: ReviewDiffProps) {
  const layout = useStore($reviewDiffLayout)

  return layout === 'split' ? (
    <SplitDiffPanel diff={diff} key={path} path={path} />
  ) : (
    <FileDiffPanel className="mx-0 mb-0 h-full max-h-none" diff={diff} path={path} virtualized />
  )
}
