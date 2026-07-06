import { useMemo, useState } from 'react'

import { asRecords, asText, InvalidPlanBlock, parseJsonRecord, PlanBlockShell } from './plan-block-utils'
import type { RichFenceProps } from './types'

function tabBody(tab: Record<string, unknown>): string {
  const body = tab.content ?? tab.body ?? tab.markdown

  if (Array.isArray(body)) {
    return body.map(item => asText(item)).filter(Boolean).join('\n')
  }

  return asText(body)
}

export default function TabsRenderer({ code }: RichFenceProps) {
  const data = useMemo(() => parseJsonRecord(code), [code])
  const [activeIndex, setActiveIndex] = useState(0)

  if (!data) {
    return <InvalidPlanBlock code={code} kind="tabs" />
  }

  const title = asText(data.title, 'Tabs')
  const tabs = asRecords(data.tabs ?? data.items ?? data.options).filter(tab => asText(tab.title ?? tab.label))

  if (tabs.length === 0) {
    return <InvalidPlanBlock code={code} kind="tabs" />
  }

  const safeActiveIndex = Math.min(activeIndex, tabs.length - 1)
  const activeTab = tabs[safeActiveIndex]
  const activeTitle = asText(activeTab.title ?? activeTab.label, `Tab ${safeActiveIndex + 1}`)
  const activeBody = tabBody(activeTab)

  return (
    <PlanBlockShell title={title}>
      <div className="grid gap-3" data-testid="tabs-embed">
        <div className="flex flex-wrap gap-1 rounded-lg bg-muted p-1">
          {tabs.map((tab, index) => {
            const label = asText(tab.title ?? tab.label, `Tab ${index + 1}`)
            const selected = index === safeActiveIndex

            return (
              <button
                aria-selected={selected}
                className={
                  selected
                    ? 'rounded-md bg-background px-3 py-1.5 text-sm font-semibold shadow-xs'
                    : 'rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground'
                }
                key={`${label}-${index}`}
                onClick={() => setActiveIndex(index)}
                type="button"
              >
                {label}
              </button>
            )
          })}
        </div>
        <section className="rounded-lg border border-(--ui-stroke-tertiary) bg-muted/15 p-3">
          <div className="mb-2 text-sm font-semibold">{activeTitle}</div>
          <pre className="text-sm leading-relaxed whitespace-pre-wrap text-muted-foreground">{activeBody || 'No content'}</pre>
        </section>
      </div>
    </PlanBlockShell>
  )
}
