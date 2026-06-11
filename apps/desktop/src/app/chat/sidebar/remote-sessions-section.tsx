import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $remoteDevices, $remoteSessions, type RemoteDevice } from '@/store/remote-sessions'

import { SidebarSectionHeader } from './section-header'
import { formatSidebarRowAge } from './session-row'

interface SidebarRemoteSessionsSectionProps {
  label: string
  // Localized "New session in {target}" — reused from the workspace-group label
  // so create-from-anywhere needs no new strings.
  newSessionLabel: (target: string) => string
  onCreateOnDevice: (endpoint: string) => void
  onResumeSession: (sessionId: string) => void
  onToggle: () => void
  open: boolean
}

// A friendly device name for the create row: the presence record's host, else
// the endpoint's host:port (so an unnamed peer still reads as more than a raw
// ws URL).
function deviceLabel(device: RemoteDevice): string {
  if (device.host) {
    return device.host
  }

  try {
    return new URL(device.endpoint).host || device.endpoint
  } catch {
    return device.endpoint
  }
}

function remoteStatusKind(status: string): 'active' | 'next' | 'starting' | 'user' {
  const normalized = status.trim().toLowerCase()

  if (normalized === 'working' || normalized === 'busy') {
    return 'active'
  }

  if (
    normalized === 'waiting' ||
    normalized.includes('input') ||
    normalized.includes('user') ||
    normalized.includes('authorize')
  ) {
    return 'user'
  }

  if (normalized === 'starting') {
    return 'starting'
  }

  return 'next'
}

// "Live on other devices": sessions discovered via presence on a peer gateway
// (Phase 2b). Clicking one resumes it — useSessionActions.resumeSession dials
// the advertised endpoint and the existing chat view streams it like a local
// session. Each reachable peer also gets a "New session" row (Phase 3:
// create-from-anywhere) that creates a fresh session ON that device. Renders
// nothing when there are no remote sessions, so single-device users (or anyone
// without presence sync) never see it.
export function SidebarRemoteSessionsSection({
  label,
  newSessionLabel,
  onCreateOnDevice,
  onResumeSession,
  onToggle,
  open
}: SidebarRemoteSessionsSectionProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const remotes = useStore($remoteSessions)
  const devices = useStore($remoteDevices)

  if (remotes.length === 0) {
    return null
  }

  return (
    <SidebarGroup className="shrink-0 p-0">
      <SidebarSectionHeader label={label} meta={String(remotes.length)} onToggle={onToggle} open={open} />
      {open && (
        <SidebarGroupContent>
          <div className="grid gap-px">
            {remotes.map(remote => {
              const statusKind = remoteStatusKind(remote.status)
              const active = statusKind === 'active'
              const needsInput = statusKind === 'user'
              const meta =
                statusKind === 'active'
                  ? ''
                  : statusKind === 'user'
                    ? formatSidebarRowAge(remote.updatedAt, r)
                    : statusKind === 'starting'
                      ? r.sessionStarting
                      : r.nextAction
              const metaTitle =
                statusKind === 'user'
                  ? r.waitingForAnswer
                  : statusKind === 'starting'
                    ? r.sessionStarting
                    : statusKind === 'next'
                      ? r.waitingForNextAction
                      : r.sessionRunning

              return (
                <button
                  className="group grid min-h-[1.625rem] w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1.5 rounded-md bg-transparent py-0.5 pl-2 pr-2 text-left transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background)"
                  data-remote-session-row
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
                          : needsInput
                            ? 'quest-glow size-1.5 bg-amber-500'
                          : 'size-1 bg-(--ui-text-quaternary) opacity-80'
                      )}
                    />
                  </span>
                  <span className="block min-w-0 truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground">
                    {remote.title}
                  </span>
                  {meta && (
                    <span
                      className={cn(
                        'min-w-6 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary)',
                        needsInput && 'text-amber-400/90'
                      )}
                      data-remote-session-status
                      title={metaTitle}
                    >
                      {meta}
                    </span>
                  )}
                </button>
              )
            })}
            {devices.map(device => {
              const target = deviceLabel(device)

              return (
                <button
                  className="group grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] items-center gap-1.5 rounded-md bg-transparent py-1 pl-2 pr-2 text-left text-(--ui-text-tertiary) transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:text-foreground"
                  key={`new:${device.endpoint}`}
                  onClick={() => onCreateOnDevice(device.endpoint)}
                  title={newSessionLabel(target)}
                  type="button"
                >
                  <span className="grid w-3.5 shrink-0 place-items-center">
                    <Codicon name="add" size="0.8125rem" />
                  </span>
                  <span className="block truncate text-[0.8125rem] font-normal">{newSessionLabel(target)}</span>
                </button>
              )
            })}
          </div>
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}
