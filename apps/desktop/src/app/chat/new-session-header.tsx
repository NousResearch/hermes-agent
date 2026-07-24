import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { titlebarHeaderBaseClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { ProfileTag } from './profile-tag'

interface NewSessionHeaderProps {
  cwd: string
  profile: null | string
  projectName: null | string
  showProfileTag: boolean
}

interface FreshSessionDraftState {
  activeSessionId: null | string
  isRoutedSessionView: boolean
  selectedSessionId: null | string
}

export function isFreshSessionDraft({
  activeSessionId,
  isRoutedSessionView,
  selectedSessionId
}: FreshSessionDraftState): boolean {
  return !selectedSessionId && !activeSessionId && !isRoutedSessionView
}

function workspaceLeaf(cwd: string): string {
  const normalized = cwd.trim().replace(/[\\/]+$/, '')

  return normalized.split(/[\\/]/).filter(Boolean).pop() || normalized
}

export function NewSessionHeader({ cwd, profile, projectName, showProfileTag }: NewSessionHeaderProps) {
  const { t } = useI18n()
  const path = cwd.trim()
  const workspace = projectName?.trim() || workspaceLeaf(path) || t.sidebar.noProject

  return (
    <header className={titlebarHeaderBaseClass}>
      <div className={cn(titlebarHeaderTitleClass, 'flex items-center')}>
        {showProfileTag && <ProfileTag className="pointer-events-auto mr-1.5" profile={profile} />}
        <div className="flex min-w-0 items-center gap-1.5 text-[0.75rem] leading-none">
          <span className="shrink-0 font-medium text-(--ui-text-secondary)">{t.sidebar.nav['new-session']}</span>
          <span aria-hidden className="text-(--ui-text-quaternary)">
            ·
          </span>
          <Tip label={path || t.sidebar.noProject} side="bottom">
            <span
              className="pointer-events-auto min-w-0 truncate rounded-[3px] bg-(--ui-control-background) px-1.5 py-1 text-[0.6875rem] text-(--ui-text-tertiary) [-webkit-app-region:no-drag] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              tabIndex={0}
            >
              {workspace}
            </span>
          </Tip>
        </div>
      </div>
    </header>
  )
}
