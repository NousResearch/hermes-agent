import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ bound: false, poppedOut: false }))
vi.mock('@assistant-ui/react', () => ({ ComposerPrimitive: {
  Root: ({ children, ...props }: any) => <form {...props}>{children}</form>,
  Input: ({ children }: { children: ReactNode }) => children,
  Unstable_TriggerPopoverRoot: ({ children }: { children: ReactNode }) => children
} }))
vi.mock('@/app/chat/session-view', async () => { const { atom } = await import('nanostores');

 return { useSessionView: () => ({ $storedId: atom(null) }) } })
vi.mock('@/themes', () => ({ useTheme: () => ({ availableThemes: [], themeName: 'default' }) }))
vi.mock('./use-coding-workspace', () => ({ useCodingWorkspace: () => ({ visible: !state.bound, owner: { connectionId: 'local', profile: 'coder', draftKey: 'new' } }) }))
vi.mock('./coding-workspace-controls', () => ({ CodingWorkspaceControls: () => <button data-slot="coding-workspace-controls">Project picker</button> }))
vi.mock('./status-stack/coding-row', () => ({ CodingStatusRow: () => state.bound ? <button data-slot="coding-workspace-summary">Bound workspace</button> : null }))
vi.mock('./hooks/use-composer-popout', () => ({ useComposerPopout: () => ({ poppedOut: state.poppedOut, popoutAllowed: true, popoutPosition: { bottom: 0, right: 0 } }) }))
vi.mock('./hooks/use-composer-draft', () => ({ useComposerDraft: () => ({ activeQueueSessionKeyRef: { current: null }, draftRef: { current: '' }, editorRef: { current: null }, sessionIdRef: { current: null } }) }))
vi.mock('./hooks/use-composer-undo', () => ({ useComposerUndo: () => ({ resetUndoHistory: () => {} }) }))
vi.mock('./hooks/use-composer-queue', () => ({ useComposerQueue: () => ({ queuedPrompts: [] }) }))
vi.mock('./hooks/use-composer-metrics', () => ({ useComposerMetrics: () => ({ stacked: true }) }))
vi.mock('./hooks/use-composer-submit', () => ({ useComposerSubmit: () => ({}) }))
vi.mock('./hooks/use-composer-trigger', () => ({ useComposerTrigger: () => ({ triggerKeyConsumedRef: { current: false }, triggerItems: [] }), triggerKeyUpHandler: () => () => {} }))
vi.mock('./hooks/use-composer-drop', () => ({ useComposerDrop: () => ({}) }))
vi.mock('./hooks/use-composer-branch', () => ({ useComposerBranch: () => ({}) }))
vi.mock('./hooks/use-composer-voice', () => ({ useComposerVoice: () => ({ conversation: {} }) }))
vi.mock('./hooks/use-composer-url-dialog', () => ({ useComposerUrlDialog: () => ({}) }))
vi.mock('./hooks/use-at-completions', () => ({ useAtCompletions: () => ({}) }))
vi.mock('./hooks/use-slash-completions', () => ({ useSlashCompletions: () => ({}) }))
vi.mock('./hooks/use-emoji-completions', () => ({ useEmojiCompletions: () => ({}) }))
vi.mock('./hooks/use-micro-actions', () => ({ useComposerMicroActions: () => {} }))
vi.mock('./hooks/use-status-presence', () => ({ useSessionStatusPresence: () => false }))
vi.mock('./hooks/use-composer-esc-cancel', () => ({ useComposerEscCancel: () => {} }))
vi.mock('./hooks/use-composer-placeholder', () => ({ useComposerPlaceholder: () => 'Message' }))
vi.mock('./controls', () => ({ ComposerControls: () => null }))
vi.mock('./context-menu', () => ({ ContextMenu: () => null }))
vi.mock('./status-stack', () => ({ ComposerStatusStack: () => null }))
vi.mock('./micro-actions', () => ({ ActionBadges: () => null }))
vi.mock('./suggestion-pills', () => ({ SuggestionPills: () => null }))
vi.mock('./voice-activity', () => ({ VoiceActivity: () => null, VoicePlaybackActivity: () => null }))
vi.mock('./url-dialog', () => ({ UrlDialog: () => null }))
vi.mock('@/contrib/react/slot', () => ({ Slot: () => null }))

import type { ChatBarProps } from './types'

import { ChatBar } from './index'

afterEach(cleanup)
it.each([false, true])('keeps draft and bound workspace at the same external, non-draggable position (popout=%s)', poppedOut => {
  state.poppedOut = poppedOut
  state.bound = false
  const props = { busy: false, disabled: false, onCancel: vi.fn(), onSubmit: vi.fn(), state: {} } as unknown as ChatBarProps
  const view = render(<ChatBar {...props} />)
  const draft = view.container.querySelector('[data-slot="coding-workspace-controls"]')!
  expect(draft).toBeTruthy()
  expect(draft.closest('[data-slot="composer-surface"]')).toBeNull()
  expect(draft.closest('[data-slot="composer-root"]')).toBeNull()
  const row = draft.parentElement!
  expect(row.nextElementSibling?.getAttribute('data-slot')).toBe('composer-root')
  state.bound = true
  view.rerender(<ChatBar {...props} sessionId="bound" />)
  const summary = view.container.querySelector('[data-slot="coding-workspace-summary"]')!
  expect(summary.parentElement).toBe(row)
  expect(view.container.querySelectorAll('[data-slot="coding-workspace-summary"]')).toHaveLength(1)
  expect(summary.closest('[data-slot="composer-surface"]')).toBeNull()
  expect(summary.closest('[data-slot="composer-root"]')).toBeNull()
})
