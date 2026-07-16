import { useStore } from '@nanostores/react'

import {
  $browserState,
  BROWSER_QC_DIMENSIONS,
  BROWSER_QC_EVIDENCE_MAX_LENGTH,
  BROWSER_QC_NOTE_MAX_LENGTH,
  type BrowserQcDimension,
  updateBrowserQc
} from '@/app/browser/store'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

const qcLabels = {
  clipping: 'browserQcClipping',
  color: 'browserQcColor',
  composition: 'browserQcComposition',
  contrast: 'browserQcContrast',
  referenceMatch: 'browserQcReferenceMatch',
  spacing: 'browserQcSpacing',
  typography: 'browserQcTypography'
} as const satisfies Record<BrowserQcDimension, keyof ReturnType<typeof useI18n>['t']['desktop']>

export function BrowserQcPane() {
  const { t } = useI18n()
  const copy = t.desktop
  const browserState = useStore($browserState)
  const activeTab = browserState.tabs.find(tab => tab.id === browserState.activeTabId) ?? null

  const qcStatusOptions = [
    { id: 'pass', label: copy.browserQcPass },
    { id: 'fail', label: copy.browserQcFail },
    { id: 'unchecked', label: copy.browserQcUnchecked }
  ] as const

  return (
    <section
      aria-label={copy.browserQc}
      className="flex h-full min-h-0 flex-col bg-(--ui-editor-surface-background)"
      data-testid="browser-qc-pane"
    >
      <header className="shrink-0 border-b border-(--ui-stroke-tertiary) px-3 py-2">
        <h1 className="text-xs font-semibold uppercase tracking-wide text-(--ui-text-secondary)">{copy.browserQc}</h1>
      </header>
      {activeTab ? (
        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          <div className="flex flex-col gap-4">
            {BROWSER_QC_DIMENSIONS.map(dimension => {
              const item = activeTab.qc[dimension]

              return (
                <div className="flex flex-col gap-2" key={dimension}>
                  <h2 className="text-xs font-medium">{copy[qcLabels[dimension]]}</h2>
                  <div aria-label={copy[qcLabels[dimension]]} role="group">
                    <SegmentedControl
                      onChange={status => updateBrowserQc(activeTab.id, dimension, { status })}
                      options={qcStatusOptions}
                      value={item.status}
                    />
                  </div>
                  <Input
                    aria-label={copy.browserQcNote}
                    maxLength={BROWSER_QC_NOTE_MAX_LENGTH}
                    onChange={event => updateBrowserQc(activeTab.id, dimension, { note: event.target.value })}
                    placeholder={copy.browserQcNote}
                    value={item.note}
                  />
                  <Input
                    aria-label={copy.browserQcEvidence}
                    maxLength={BROWSER_QC_EVIDENCE_MAX_LENGTH}
                    onChange={event => updateBrowserQc(activeTab.id, dimension, { evidence: event.target.value })}
                    placeholder={copy.browserQcEvidence}
                    value={item.evidence}
                  />
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        <p className="grid min-h-0 flex-1 place-items-center p-4 text-center text-sm text-(--ui-text-tertiary)">
          {copy.browserEmpty}
        </p>
      )}
    </section>
  )
}
