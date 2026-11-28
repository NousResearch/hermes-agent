import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { activePreviewTabFor } from '@/app/chat/right-rail/preview-active-tab'

import { $rightRailActiveTabId } from './layout'
import {
  $browserPages,
  $previewTabs,
  closeRightRail,
  openPreview,
  openPreviewPopupTab,
  type PreviewTarget,
  takePreviewPopupPost
} from './preview'
import { $selectedStoredSessionId } from './session'

function urlTarget(source: string): PreviewTarget {
  return { kind: 'url', label: source, source, url: source }
}

function reset() {
  $browserPages.set({})
  $selectedStoredSessionId.set(null)
  closeRightRail()
  window.localStorage.clear()
}

// A preview page's new tab (`_blank` link or form POST, `window.open`) opens as
// a preview tab of the same chat instead of the OS browser, and becomes the
// tab the agent's desktop_preview / drive_preview act on.
describe('openPreviewPopupTab', () => {
  beforeEach(reset)
  afterEach(reset)

  it('opens a second Browser tab owned like its opener and fronts it', () => {
    $selectedStoredSessionId.set('chat-a')
    openPreview(urlTarget('http://app.example.test/index.html'))
    const opener = $previewTabs.get()[0]

    const id = openPreviewPopupTab(opener.id, 'http://app.example.test/page2.html')

    expect(id).not.toBeNull()
    const tabs = $previewTabs.get()
    expect(tabs).toHaveLength(2)
    const popup = tabs.find(tab => tab.id === id)!
    expect(popup.sessionId).toBe(opener.sessionId)
    expect(popup.pinned).toBe(opener.pinned)
    expect(popup.target).toMatchObject({ kind: 'url', url: 'http://app.example.test/page2.html' })
    expect($rightRailActiveTabId.get()).toBe(id)
  })

  it('makes the new tab the one the agent tools act on', () => {
    $selectedStoredSessionId.set('chat-a')
    openPreview(urlTarget('http://app.example.test/index.html'))
    const opener = $previewTabs.get()[0]

    const id = openPreviewPopupTab(opener.id, 'http://app.example.test/page2.html')

    expect(activePreviewTabFor('chat-a')?.id).toBe(id)
  })

  it('starts a POST tab blank and hands its POST to the pane exactly once', () => {
    openPreview(urlTarget('https://admin.example.test/instances'))
    const opener = $previewTabs.get()[0]
    const body = new TextEncoder().encode('user=demo')

    const id = openPreviewPopupTab(opener.id, 'https://api.example.test/instances/1/login-as/2', {
      post: { body, contentType: 'application/x-www-form-urlencoded' }
    })!

    expect($previewTabs.get().find(tab => tab.id === id)?.target.url).toBe('about:blank')
    expect(takePreviewPopupPost(id)).toEqual({
      body,
      contentType: 'application/x-www-form-urlencoded',
      url: 'https://api.example.test/instances/1/login-as/2'
    })
    expect(takePreviewPopupPost(id)).toBeUndefined()
  })

  it('opens a background tab without taking focus', () => {
    openPreview(urlTarget('https://example.test/'))
    const opener = $previewTabs.get()[0]

    const id = openPreviewPopupTab(opener.id, 'https://example.test/other', { background: true })

    expect($previewTabs.get()).toHaveLength(2)
    expect($rightRailActiveTabId.get()).toBe(opener.id)
    expect(id).not.toBe(opener.id)
  })

  it('refuses a non-web URL or an unknown opener', () => {
    openPreview(urlTarget('https://example.test/'))
    const opener = $previewTabs.get()[0]

    expect(openPreviewPopupTab(opener.id, 'file:///etc/hosts')).toBeNull()
    expect(openPreviewPopupTab('url:browser-missing', 'https://example.test/x')).toBeNull()
    expect($previewTabs.get()).toHaveLength(1)
  })
})
