// The plugin authoring contract types; the SDK barrel re-exports them.

import type { ReactNode } from 'react'

export type {
  HermesPlugin,
  PetMessageTone,
  PetSayOptions,
  PluginAppActionId,
  PluginAppActionInfo,
  PluginContext,
  PluginContribution,
  PluginNativeNotificationInput,
  PluginNotificationAction,
  PluginOs,
  PluginPet,
  PluginRestOptions,
  PluginRunActionResult,
  PluginSettingsPage,
  PluginSettingsSubpage,
  PluginStorage
} from '@/contrib/plugin'

/** `host.openWorkspace` options: a main-area tab a plugin renders. */
export interface PluginWorkspaceOptions {
  dock?: { before?: null | string; pane: string; pos: 'bottom' | 'center' | 'left' | 'right' | 'top' }
  headerVeto?: boolean
  minWidth?: string
  onClose?: () => void
  render: () => ReactNode
  /** The tab's live label, a component that subscribes for itself (a renamed room): it changes
   *  without registering the tab again or bringing it to the front. Absent, the tab shows `title`. */
  tabTitle?: () => ReactNode
  /** The same label as text, read when a menu opens or a drag starts. Absent, `title`. */
  tabTitleText?: () => string
  /** The label sampled when the tab opens: the fallback on desktops without `tabTitle`. */
  title?: string
  uncloseable?: boolean
}
