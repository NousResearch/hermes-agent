import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $previewTabs,
  dropPreviewTabsForProfile,
  newBrowserTab,
  openPreview,
  previewTabOwnerProfile,
  type PreviewTarget,
  setPreviewScope
} from './preview'

function fileTarget(source: string): PreviewTarget {
  return { kind: 'file', label: source, path: source, previewKind: 'html', source, url: `file://${source}` }
}

describe('preview per-profile cookie scoping (#150283)', () => {
  afterEach(() => {
    setPreviewScope('default')
    window.localStorage.clear()
  })

  // F4 (SECURITY): the cookie jar lives in LOCAL userData. A LOCAL profile delete clears it; a REMOTE
  // profile delete (which shares only the NAME with a same-named local profile) must NOT, or deleting
  // a remote profile wipes the local profile's imported sign-ins.
  it('clears the local cookie jar for a local delete but not a remote one', () => {
    const drop = vi.fn(() => Promise.resolve({ ok: true }))
    const desktopWindow = window as unknown as { hermesDesktop?: { dropPreviewSigninsForProfile?: typeof drop } }
    const previous = desktopWindow.hermesDesktop
    desktopWindow.hermesDesktop = { dropPreviewSigninsForProfile: drop }

    try {
      dropPreviewTabsForProfile('shared-name', false) // remote delete
      expect(drop).not.toHaveBeenCalled()

      dropPreviewTabsForProfile('shared-name', true) // local delete
      expect(drop).toHaveBeenCalledWith({ profile: 'shared-name' })
      expect(drop).toHaveBeenCalledTimes(1)
    } finally {
      desktopWindow.hermesDesktop = previous
    }
  })

  // F2 (SECURITY): a session-less DRAFT tab in a named profile's workspace is owned by THAT profile,
  // not defaulted to `default` — otherwise its webview rides the shared default jar and the named
  // profile's draft workspace sees the default profile's sign-ins.
  it('owns a draft tab by its workspace profile, never defaulting a named profile to the shared jar', () => {
    try {
      setPreviewScope('work')
      newBrowserTab()
      const workTabId = $previewTabs.get().at(-1)!.id
      expect($previewTabs.get().at(-1)!.sessionId).toBeUndefined() // a true draft tab
      expect(previewTabOwnerProfile(workTabId)).toBe('work')

      setPreviewScope('default')
      newBrowserTab()
      const defaultTabId = $previewTabs.get().at(-1)!.id
      expect(previewTabOwnerProfile(defaultTabId)).toBe('default')

      // A tab not in any bucket resolves to the workspace on screen, never 'default' unconditionally.
      setPreviewScope('work')
      expect(previewTabOwnerProfile('url:never-stored')).toBe('work')
    } finally {
      setPreviewScope('default')
      dropPreviewTabsForProfile('work')
    }
  })

  // G1 (#150283, security): a draft FILE tab's id derives from the file URL alone (no profile,
  // no session), so the SAME html file opened as a draft under two profiles collides on one id.
  // Resolving the owner by a GLOBAL first-match-by-id search returned whichever bucket iterated
  // first, so the `default` draft resolved to `work`'s cookie jar and navigating from its Browser
  // bar used `work`'s sign-ins — regressing per-profile isolation. Each view must resolve to ITS
  // OWN workspace. (Red on base: the `default` draft resolves to `work`.)
  it('resolves a draft file tab shared across profiles to the viewed workspace, never a colliding bucket', () => {
    try {
      // Open the same html file as a draft under `work` first (so its bucket is stored first).
      setPreviewScope('work')
      openPreview(fileTarget('/shared/collide.html'))
      const workTabId = $previewTabs.get().at(-1)!.id
      expect($previewTabs.get().at(-1)!.sessionId).toBeUndefined() // a true draft tab

      // Open the SAME file as a draft under `default`: same derived id, different workspace.
      setPreviewScope('default')
      openPreview(fileTarget('/shared/collide.html'))
      const defaultTabId = $previewTabs.get().at(-1)!.id
      expect(defaultTabId).toBe(workTabId) // the collision the fix must tolerate

      // Viewing under `default`, the tab is `default`'s — NOT `work`'s (the first-stored bucket).
      expect(previewTabOwnerProfile(defaultTabId)).toBe('default')

      // And back under `work`, the same id resolves to `work`.
      setPreviewScope('work')
      expect(previewTabOwnerProfile(workTabId)).toBe('work')
    } finally {
      setPreviewScope('default')
      dropPreviewTabsForProfile('work')
    }
  })
})
