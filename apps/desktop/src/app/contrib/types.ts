import type { ComponentProps, ReactNode } from 'react'

import type { ChatView } from '../chat'
import type { ChatSidebar } from '../chat/sidebar'
import type { CommandCenterSection } from '../command-center'
import type { useGatewayRequest } from '../gateway/hooks/use-gateway-request'
import type { ModelMenuPanel } from '../shell/model-menu-panel'

export type GatewayRequester = ReturnType<typeof useGatewayRequest>['requestGateway']

/** The ChatSidebar handlers the controller owns — forwarded verbatim. */
export type SidebarActions = Pick<
  ComponentProps<typeof ChatSidebar>,
  | 'onArchiveSession'
  | 'onBranchSession'
  | 'onDeleteSession'
  | 'onLoadMoreMessaging'
  | 'onLoadMoreProfileSessions'
  | 'onLoadMoreSessions'
  | 'onManageCronJob'
  | 'onNavigate'
  | 'onNewSessionInWorkspace'
  | 'onNewSessionSplit'
  | 'onResumeSession'
  | 'onTriggerCronJob'
>

/** The ChatView handlers the controller owns — forwarded verbatim. */
export type ChatActions = Pick<
  ComponentProps<typeof ChatView>,
  | 'onAddContextRef'
  | 'onAddUrl'
  | 'onAttachDroppedItems'
  | 'onAttachImageBlob'
  | 'onBranchInNewChat'
  | 'onCancel'
  | 'onDeleteSelectedSession'
  | 'onDismissError'
  | 'onEdit'
  | 'onPasteClipboardImage'
  | 'onPickFiles'
  | 'onPickFolders'
  | 'onPickImages'
  | 'onReload'
  | 'onRemoveAttachment'
  | 'onRestoreToMessage'
  | 'onRetryResume'
  | 'onSteer'
  | 'onSubmit'
  | 'onThreadMessagesChange'
  | 'onToggleSelectedPin'
  | 'onTranscribeAudio'
>

/**
 * The complete controller-owned callback surface. One object, one stable
 * identity for the app's life — its fields are mutated in place each render,
 * so surfaces bound to it never re-render on identity churn but always invoke
 * the latest closure.
 */
export interface WiringActions extends SidebarActions, ChatActions {
  /** The live gateway instance (held in a controller ref). Surfaces recapture
   *  it by subscribing to `$gatewayState`, so no gateway prop needs threading. */
  getGateway: () => ComponentProps<typeof ChatView>['gateway']
  openAgents: () => void
  openCommandCenterSection: (section: CommandCenterSection) => void
  requestGateway: GatewayRequester
  selectModel: ComponentProps<typeof ModelMenuPanel>['onSelectModel']
  toggleCommandCenter: () => void
}

/** The four wired surfaces the controller publishes; `WiredPane` renders one by
 *  key inside a registered pane / chrome slot. */
export interface WiringApi {
  sidebar: ReactNode
  chatRoutes: ReactNode
  terminal: ReactNode
  statusbar: ReactNode
}
