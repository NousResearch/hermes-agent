import { useStore } from '@nanostores/react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'
import { $remoteSessions } from '@/store/remote-sessions'

import { SidebarPanelLabel } from '../../shell/sidebar-label'

interface SidebarRemoteSessionsSectionProps {
  label: string
  onResumeSession: (sessionId: string) => void
  onToggle: () => void
  open: boolean
}

// "Live on other devices": sessions discovered via presence on a peer gateway
// (Phase 2b). Clicking one resumes it — useSessionActions.resumeSession dials
// the advertised endpoint and the existing chat view streams it like a local
// session. Renders nothing when there are no remote sessions, so single-device
// users (or anyone without presence sync) never see it.
export function SidebarRemoteSessionsSection({ label, onResumeSession, onToggle, open }: SidebarRemoteSessionsSectionProps) {
  const remotes = useStore($remoteSessions)

  if (remotes.length === 0) {
    return null
  }

  return (
    <SidebarGroup className="shrink-0 p-0">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          <SidebarPanelLabel>{label}</SidebarPanelLabel>
          <span className="text-[0.625rem] leading-none text-(--ui-text-quaternary)">{remotes.length}</span>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      </div>
      {open && (
        <SidebarGroupContent>
          <div className="grid gap-px">
            {remotes.map(remote => {
              const active = remote.status === 'working' || remote.status === 'busy'
              return (
                <button
                  className="group grid min-h-[2.375rem] w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] items-center gap-1.5 rounded-md bg-transparent py-1 pl-2 pr-2 text-left transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background)"
                  key={remote.sessionId}
                  onClick={() => onResumeSession(remote.sessionId)}
                  title={remote.host ? `${remote.title} · ${remote.host}` : remote.title}
                  type="button"
                >
                  <span className="grid w-3.5 shrink-0 place-items-center">
                    <span
                      aria-hidden="true"
                      className={cn(
                        'rounded-full',
                        active
                          ? "relative size-1.5 bg-orange-500 shadow-[0_0_0.625rem_color-mix(in_srgb,#f97316_60%,transparent)] before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-orange-500 before:opacity-70 before:content-['']"
                          : 'size-1 bg-(--ui-text-quaternary) opacity-80'
                      )}
                    />
                  </span>
                  <span className="flex min-w-0 flex-col">
                    <span className="block truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground">
                      {remote.title}
                    </span>
                    {remote.host && (
                      <span className="block truncate text-[0.625rem] leading-tight text-(--ui-text-quaternary)">
                        {remote.host}
                      </span>
                    )}
                  </span>
                </button>
              )
            })}
          </div>
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}
