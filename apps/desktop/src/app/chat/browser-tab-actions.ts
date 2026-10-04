import { queryVisible } from '@/components/pane-shell/pane-visibility'
import { $browserWorkspaces, commandBrowserWorkspace } from '@/store/browser-workspaces'
import { openFindBar } from '@/store/find-in-page'
import { windowBrowserWorkspaceId } from '@/store/windows'

function focusSelectedPage(state: Awaited<ReturnType<typeof commandBrowserWorkspace>>) {
  window.requestAnimationFrame(() => {
    if (state && $browserWorkspaces.get()[state.id]?.activeTabId === state.activeTabId) {
      queryVisible<HTMLElement>('[data-browser-active] webview')?.focus()
    }
  })
}

export function runBrowserTabAction(action: string): boolean {
  const id = windowBrowserWorkspaceId()
  const state = id ? $browserWorkspaces.get()[id] : null

  if (!state || state.closed) {
    return false
  }

  if (action === 'view.findInPage') {
    openFindBar()

    return true
  }

  if (action === 'session.newTab') {
    void commandBrowserWorkspace({ kind: 'new' }).then(created => {
      // User-created tabs focus the address after React commits. A later
      // selection wins; background snapshot hydration never steals focus.
      window.requestAnimationFrame(() => {
        if (created && $browserWorkspaces.get()[created.id]?.activeTabId === created.activeTabId) {
          queryVisible<HTMLInputElement>('[data-browser-active] input[inputmode="url"]')?.focus()
        }
      })
    })

    return true
  }

  const active = state.activeTabId

  if (!active) {
    return false
  }

  if (action === 'view.closeTab') {
    void commandBrowserWorkspace({ kind: 'close', tabId: active }).then(focusSelectedPage)

    return true
  }

  if (action === 'session.next' || action === 'session.prev') {
    const index = state.tabs.findIndex(tab => tab.id === active)
    const offset = action === 'session.next' ? 1 : -1
    const tab = state.tabs[(index + offset + state.tabs.length) % state.tabs.length]

    if (tab) {
      void commandBrowserWorkspace({ kind: 'select', tabId: tab.id }).then(focusSelectedPage)
    }

    return true
  }

  return false
}
