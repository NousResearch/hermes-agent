import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { paneMirror } from '@/app/chat/pane-mirror'
import { buildSandboxedHtmlDocument, sandboxedHtmlApprovalIdentity } from '@/app/chat/right-rail/sandboxed-html-approval'
import { SandboxedHtmlDocument } from '@/app/chat/right-rail/sandboxed-html-preview'
import { revealTreePane } from '@/components/pane-shell/tree/store'
import { useI18n } from '@/i18n'

import { bridgeForGeneratedView } from './bridge'
import {
  $generatedViews,
  $openGeneratedViews,
  closeGeneratedView,
  openGeneratedView,
  watchGeneratedViewDocuments
} from './store'

export const GENERATED_VIEW_PANE_PREFIX = 'generated-view'

function GeneratedViewPane({ id }: { id: string }) {
  const { t } = useI18n()
  const views = useStore($generatedViews)
  const view = views.find(candidate => candidate.manifest.id === id)
  const bridge = useMemo(() => (view ? bridgeForGeneratedView(view) : undefined), [view])

  if (!view) {
    return null
  }

  const authority = [
    t.artifacts.viewIsolationSummary,
    view.manifest.capabilities.length > 0
      ? `${t.artifacts.viewCapabilities}: ${view.manifest.capabilities.join(', ')}.`
      : `${t.artifacts.viewCapabilities}: —.`,
    view.manifest.bindings.length > 0
      ? `${t.artifacts.viewBindings}: ${view.manifest.bindings.join(', ')}.`
      : `${t.artifacts.viewBindings}: —.`
  ].join(' ')

  return (
    <div className="relative h-full min-h-0 w-full overflow-hidden bg-background">
      <SandboxedHtmlDocument
        bridge={bridge}
        digest={view.digest}
        documentSource={buildSandboxedHtmlDocument(view.html)}
        frameTitle={view.manifest.title}
        identity={sandboxedHtmlApprovalIdentity(view.connectionKey, view.manifestPath)}
        path={view.manifestPath}
        permissionSummary={authority}
        source={view.html}
        title={t.artifacts.viewApprovalTitle(view.manifest.title)}
      />
    </div>
  )
}

const watchPaneMirror = paneMirror({
  source: $openGeneratedViews,
  key: view => view.manifest.id,
  prefix: GENERATED_VIEW_PANE_PREFIX,
  minWidth: '20rem',
  title: id => $generatedViews.get().find(view => view.manifest.id === id)?.manifest.title ?? id,
  render: id => <GeneratedViewPane id={id} />,
  close: closeGeneratedView
})

let watching = false

export function watchGeneratedViewPanes(): void {
  if (watching) {
    return
  }

  watching = true
  // Wait for the first filesystem pass before the mirror's initial sync. An
  // eager empty sync would treat restored generated-view panes as stale and
  // remove their persisted layout slots before their documents are known.
  void watchGeneratedViewDocuments().finally(watchPaneMirror)
}

export function openGeneratedViewPane(id: string): void {
  openGeneratedView(id)
  queueMicrotask(() => revealTreePane(`${GENERATED_VIEW_PANE_PREFIX}:${id}`))
}
