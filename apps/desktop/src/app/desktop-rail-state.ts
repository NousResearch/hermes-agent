import { type BrowserTabId, isBrowserTabId } from '@/store/browser'
import type { RightRailTabId } from '@/store/layout'
import type { PreviewTarget } from '@/store/preview'

interface DesktopRailHasTabsInput {
  browserEnabled: boolean
  browserTabCount: number
  filePreviewTabCount: number
  previewTarget: PreviewTarget | null
}

interface CloseActiveDesktopRailTabActions {
  closeBrowserTab: (tabId: BrowserTabId) => void
  closePreviewTab: () => void
}

export function desktopRailHasTabs({ browserEnabled, browserTabCount, filePreviewTabCount, previewTarget }: DesktopRailHasTabsInput): boolean {
  return Boolean(previewTarget || filePreviewTabCount > 0 || (browserEnabled && browserTabCount > 0))
}

export function closeActiveDesktopRailTab(activeTabId: RightRailTabId, actions: CloseActiveDesktopRailTabActions): void {
  if (isBrowserTabId(activeTabId)) {
    actions.closeBrowserTab(activeTabId)

    return
  }

  actions.closePreviewTab()
}
