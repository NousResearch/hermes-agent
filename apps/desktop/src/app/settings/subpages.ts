import {
  Archive,
  Bell,
  Box,
  Brain,
  CircleLetterA,
  Cpu,
  Download,
  FileImage,
  FileText,
  FolderOpen,
  Globe,
  type IconComponent,
  Keyboard,
  KeyRound,
  Lock,
  MessageCircle,
  Mic,
  Monitor,
  Network,
  Palette,
  PawPrint,
  Settings2,
  ShieldLock,
  Terminal,
  Trash2,
  Users,
  Volume2,
  Wrench
} from '@/lib/icons'

import { APPEARANCE_SUBPAGES, appearanceSubpageForSetting } from './appearance-subpages'
import { CONFIG_SUBPAGES, configSubpageForField } from './config-subpages'
import { OTHER_SUBPAGES } from './other-subpages'
import type { SettingsView } from './types'

export interface SettingsSubpage {
  id: string
  labelKey: string
}

const SUBPAGE_ICONS: Record<string, IconComponent> = {
  appearanceTheme: Palette, appearanceTypography: CircleLetterA,
  appearanceWindowLayout: Monitor, appearanceChatDisplay: MessageCircle,
  appearancePet: PawPrint, appearanceGeneral: Settings2,
  modelMain: Box, modelAuxiliary: Cpu, modelMoa: Users, modelFallbacks: Box,
  chatBehavior: MessageCircle, chatAttachments: FileImage,
  workspaceProjects: FolderOpen, workspaceShell: Terminal, workspaceFiles: FileText,
  safetyApprovals: ShieldLock, safetyPrivacy: Lock, safetyCheckpoints: Archive,
  browserProfile: Globe, browserNetwork: Network,
  memoryPersistent: Brain, memoryContext: FileText,
  voiceConversation: Mic, voiceTranscription: FileText, voiceSpeech: Volume2,
  advancedRuntime: Cpu, advancedTools: Wrench, advancedTerminal: Terminal,
  advancedOutput: FileText, advancedDelegation: Users, advancedDesktop: Monitor,
  gatewayConnection: Monitor, gatewayDevices: Network, gatewayManagedUpdates: Download,
  keyboardShortcuts: Keyboard, screenCapture: FileImage,
  notificationAlerts: Bell, notificationSounds: Volume2,
  archivedSessions: Archive, defaultDirectory: FolderOpen,
  vaultCredentials: KeyRound, vaultSources: Lock, appUpdates: Download, uninstall: Trash2
}

export function settingsSubpageIcon(page: SettingsSubpage, fallback: IconComponent): IconComponent {
  return SUBPAGE_ICONS[page.labelKey] ?? fallback
}

export function settingsSubpages(view: SettingsView): readonly SettingsSubpage[] {
  if (view === 'config:appearance') {
    return APPEARANCE_SUBPAGES
  }

  if (view.startsWith('config:')) {
    return CONFIG_SUBPAGES[view.slice('config:'.length)] ?? []
  }

  return OTHER_SUBPAGES[view] ?? []
}

/** Shared by search serialization and saved links that predate subpages. */
export function settingsSubpageForTarget(view: SettingsView, field?: string, setting?: string): string | undefined {
  if (view === 'config:appearance' && setting) {
    return appearanceSubpageForSetting(setting)
  }

  if (view.startsWith('config:') && field) {
    return configSubpageForField(view.slice('config:'.length), field)
  }

  return undefined
}

const LEGACY_SUBPAGE_TARGETS: Record<string, { params: string[]; page: string }> = {
  'config:model': { params: ['aux'], page: 'auxiliary' },
  sessions: { params: ['session'], page: 'archived' },
  vault: { params: ['kind', 'label', 'origin'], page: 'credentials' }
}

export function settingsSubpageForLegacyLink(view: SettingsView, params: URLSearchParams): string | undefined {
  const target = LEGACY_SUBPAGE_TARGETS[view]

  return target?.params.some(param => params.get(param))
    ? target.page
    : settingsSubpageForTarget(view, params.get('field') ?? undefined, params.get('setting') ?? undefined)
}

export function resolveSettingsSubpage(view: SettingsView, params: URLSearchParams): string | undefined {
  const pages = settingsSubpages(view)
  const requested = settingsSubpageForLegacyLink(view, params) ?? params.get('page')

  return pages.find(page => page.id === requested)?.id ?? pages[0]?.id
}
