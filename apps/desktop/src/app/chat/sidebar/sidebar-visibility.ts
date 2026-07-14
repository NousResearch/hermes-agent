export interface SidebarAuxiliarySectionVisibility {
  searchActive: boolean
  workspaceGroupingActive: boolean
}

/**
 * Messaging-platform sections are auxiliary navigation, not a replacement for
 * Recents/Projects. They must remain visible while workspace grouping is active
 * so Slack/Telegram/etc. conversations do not disappear behind the Projects
 * toggle.
 */
export function shouldShowMessagingSections({ searchActive }: SidebarAuxiliarySectionVisibility): boolean {
  return !searchActive
}
