import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createSplitPaneView, type PaneSessionView, PaneViewContext } from '@/app/chat/pane-view'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { clearSessionDraft } from '@/store/composer'
import { $activePaneId } from '@/store/split'

import { requestComposerInsert } from './focus'
import { useComposerDraft } from './hooks/use-composer-draft'

// Composer bus gating (design §3.6): with the split open BOTH mounted
// composers subscribe to the window-global bus as 'main', so an insert
// targeted 'main' (deep-link blueprint, terminal Cmd+L, review pane) must
// reach only the ACTIVE pane's draft engine. This drives the REAL
// useComposerDraft twice — main (default context) and split (provider) — and
// asserts delivery through each engine's draftRef + editor DOM.

type DraftApi = ReturnType<typeof useComposerDraft>

const EMPTY_REPOSITORY = ExportedMessageRepository.fromBranchableArray([])

function RuntimeShell({ children }: { children: ReactNode }) {
  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messageRepository: EMPTY_REPOSITORY,
    onCancel: async () => undefined,
    onEdit: async () => undefined,
    onNew: async () => undefined,
    onReload: async () => undefined,
    setMessages: () => undefined
  })

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

function DraftProbe({ onReady, testId }: { onReady: (api: DraftApi) => void; testId: string }) {
  const draft = useComposerDraft({
    activeQueueSessionKey: null,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null },
    sessionId: null
  })

  onReady(draft)

  return <div contentEditable data-testid={testId} ref={draft.editorRef} suppressContentEditableWarning />
}

function PaneComposer({
  onReady,
  testId,
  view
}: {
  onReady: (api: DraftApi) => void
  testId: string
  view?: PaneSessionView
}) {
  const probe = (
    <RuntimeShell>
      <DraftProbe onReady={onReady} testId={testId} />
    </RuntimeShell>
  )

  // Main pane = NO provider: the default context is the main bundle, exactly
  // like the production tree.
  return view ? <PaneViewContext.Provider value={view}>{probe}</PaneViewContext.Provider> : probe
}

async function dispatchInsert(text: string) {
  await act(async () => {
    requestComposerInsert(text, { mode: 'block', target: 'main' })
    // The bus defers dispatch by a macrotask.
    await new Promise(resolve => window.setTimeout(resolve, 10))
  })
}

describe('composer bus gating across panes', () => {
  let mainDraft!: DraftApi
  let splitDraft!: DraftApi
  let splitView: PaneSessionView

  beforeEach(() => {
    // The draft engine stashes per-session drafts (scope null for a fresh
    // draft) into localStorage on unmount — clear it so a prior test's draft
    // can't restore into this one's freshly mounted engines.
    window.localStorage.clear()
    clearSessionDraft(null)
    $activePaneId.set('main')
    splitView = createSplitPaneView()

    render(
      <>
        <PaneComposer onReady={api => (mainDraft = api)} testId="main-editor" />
        <PaneComposer onReady={api => (splitDraft = api)} testId="split-editor" view={splitView} />
      </>
    )
  })

  afterEach(() => {
    cleanup()
    $activePaneId.set('main')
  })

  it("delivers a 'main'-targeted insert only to the ACTIVE pane's draft engine", async () => {
    await dispatchInsert('hello main pane')

    expect(mainDraft.draftRef.current).toBe('hello main pane')
    expect(splitDraft.draftRef.current).toBe('')

    $activePaneId.set('split')
    await dispatchInsert('hello split pane')

    expect(splitDraft.draftRef.current).toBe('hello split pane')
    // The main engine kept its earlier draft — the split-targeted insert
    // never leaked into it.
    expect(mainDraft.draftRef.current).toBe('hello main pane')
  })

  it('paints the delivered insert into the active pane editor DOM only', async () => {
    $activePaneId.set('split')
    await dispatchInsert('/blueprint deploy')

    expect(document.querySelector('[data-testid="split-editor"]')?.textContent).toContain('/blueprint deploy')
    expect(document.querySelector('[data-testid="main-editor"]')?.textContent ?? '').toBe('')
  })
})
