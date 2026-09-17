import { useStore } from '@nanostores/react'

import { sessionDotClassName } from '@/app/chat/session-status-dot'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { desktopGit } from '@/lib/desktop-git'
import { cn } from '@/lib/utils'
import {
  $sidebarCardRows,
  $sidebarFiltersActive,
  $sidebarGrouping,
  $sidebarListGroupIds,
  $sidebarOrdering,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarRowMeta,
  $sidebarShowAllSessions,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  $sidebarViewCustomized,
  $sidebarWorkspaceNodeOpen,
  resetSidebarView,
  setSidebarCardRows,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarShowAllSessions,
  setSidebarShowArchived,
  setWorkspaceNodesOpen,
  type SidebarGrouping,
  type SidebarOrdering,
  type SidebarRowMeta,
  toggleSidebarPrFilter,
  toggleSidebarProfileFilter,
  toggleSidebarProjectFilter,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from '@/store/layout'
import {
  $profiles,
  $showAllProfiles,
  normalizeProfileKey,
  requestProfileCreate,
  toggleShowAllProfiles
} from '@/store/profile'
import { $profileRailVisible, toggleProfileRailVisible } from '@/store/profile-rail-prefs'
import { runImportProfileFlow } from '@/store/profile-share'
import { $projectTree } from '@/store/projects'
import type { PullRequestBucket } from '@/store/pull-requests'
import { $unreadFinishedSessionIds, markAllSessionsRead } from '@/store/session'
import type { SessionStatusBucket } from '@/store/session-dot-state'
import { $sessionsHaveCost } from '@/store/sidebar-archive'

/** Which `sidebar.filter.option` entry labels a row. The tables below carry the
 *  key; the menu resolves it against the active locale at render, so a language
 *  switch relabels every row without touching the tables. */
type FilterOptionKey = keyof Translations['sidebar']['filter']['option']

interface Option<T extends string = string> {
  /** A status dot's full className, from the row's own vocabulary. */
  dot?: string
  icon?: string
  id: T
  /** English source text — also the fallback for a not-yet-translated key. */
  label: string
  labelKey?: FilterOptionKey
}

const GROUPINGS: Option<SidebarGrouping>[] = [
  { icon: 'clock', id: 'date', label: 'Updated', labelKey: 'updated' },
  { icon: 'root-folder', id: 'project', label: 'Project', labelKey: 'project' },
  { icon: 'pulse', id: 'status', label: 'Status', labelKey: 'status' },
  { icon: 'account', id: 'profile', label: 'Profile', labelKey: 'profile' }
]

const ORDERINGS: Option<SidebarOrdering>[] = [
  { icon: 'clock', id: 'updated', label: 'Updated', labelKey: 'updated' },
  { icon: 'add', id: 'created', label: 'Created', labelKey: 'created' },
  { icon: 'pulse', id: 'status', label: 'Status', labelKey: 'status' },
  { icon: 'symbol-numeric', id: 'tokens', label: 'Tokens', labelKey: 'tokens' },
  { icon: 'credit-card', id: 'cost', label: 'Cost', labelKey: 'cost' },
  { icon: 'list-ordered', id: 'manual', label: 'Manual', labelKey: 'manual' }
]

const ROW_META: Option<SidebarRowMeta>[] = [
  { icon: 'clock', id: 'updated', label: 'Updated', labelKey: 'updated' },
  { icon: 'comment', id: 'preview', label: 'Preview', labelKey: 'preview' },
  { icon: 'symbol-numeric', id: 'tokens', label: 'Tokens', labelKey: 'tokens' },
  { icon: 'credit-card', id: 'cost', label: 'Cost', labelKey: 'cost' },
  { icon: 'git-pull-request', id: 'pr', label: 'PR', labelKey: 'pr' },
  { icon: 'account', id: 'profile', label: 'Profile', labelKey: 'profile' }
]

const PR_FILTERS: Option<PullRequestBucket>[] = [
  { icon: 'git-pull-request', id: 'open', label: 'Open', labelKey: 'open' },
  { icon: 'git-pull-request-draft', id: 'draft', label: 'Draft', labelKey: 'draft' },
  { icon: 'git-merge', id: 'merged', label: 'Merged', labelKey: 'merged' },
  { icon: 'git-pull-request-closed', id: 'closed', label: 'Closed', labelKey: 'closed' },
  { icon: 'circle-slash', id: 'none', label: 'No PR', labelKey: 'none' }
]

const STATUS_FILTERS: Option<SessionStatusBucket>[] = [
  { dot: sessionDotClassName('needs-input'), id: 'needs-input', label: 'Needs input', labelKey: 'needsInput' },
  { dot: sessionDotClassName('working'), id: 'working', label: 'Working', labelKey: 'working' },
  { dot: sessionDotClassName('unread'), id: 'unread', label: 'Unread', labelKey: 'unread' },
  { dot: sessionDotClassName('draft'), id: 'draft', label: 'Draft', labelKey: 'draft' },
  {
    dot: cn(sessionDotClassName('idle'), 'bg-(--ui-text-quaternary)'),
    id: 'idle',
    label: 'Idle',
    labelKey: 'idle'
  }
]

function OptionGlyph({ option }: { option: Option }) {
  if (option.dot) {
    return <span aria-hidden="true" className={cn('shrink-0', option.dot)} />
  }

  return option.icon ? <Codicon className="text-(--ui-text-tertiary)" name={option.icon} size="0.8125rem" /> : null
}

/** Every option row — single or multi select — leaves the menu open, so a whole
 *  view can be set up in one pass. Only the actions at the bottom dismiss it. */
const keepOpen = (event: Event) => event.preventDefault()

function OptionCheckbox({ checked, onCheck, option }: { checked: boolean; onCheck: () => void; option: Option }) {
  return (
    <DropdownMenuCheckboxItem
      checked={checked}
      onSelect={event => {
        keepOpen(event)
        onCheck()
      }}
    >
      <OptionGlyph option={option} />
      {option.label}
    </DropdownMenuCheckboxItem>
  )
}

function OptionRadio({ option }: { option: Option }) {
  return (
    <DropdownMenuRadioItem onSelect={keepOpen} value={option.id}>
      <OptionGlyph option={option} />
      {option.label}
    </DropdownMenuRadioItem>
  )
}

export function SidebarFilterMenu({ className }: { className?: string }) {
  const { t } = useI18n()
  const grouping = useStore($sidebarGrouping)
  const ordering = useStore($sidebarOrdering)
  const rowMeta = useStore($sidebarRowMeta)
  const cardRows = useStore($sidebarCardRows)
  const profileRailVisible = useStore($profileRailVisible)
  const showAllSessions = useStore($sidebarShowAllSessions)
  const statusFilter = useStore($sidebarStatusFilter)
  const projectFilter = useStore($sidebarProjectFilter)
  const profileFilter = useStore($sidebarProfileFilter)
  const showAllProfiles = useStore($showAllProfiles)
  const profileNames = useStore($profiles).map(profile => normalizeProfileKey(profile.name))
  const narrowsByProfile = showAllProfiles && profileNames.length > 1
  const prFilter = useStore($sidebarPrFilter)
  const showArchived = useStore($sidebarShowArchived)
  const filtersActive = useStore($sidebarFiltersActive)
  const viewCustomized = useStore($sidebarViewCustomized)
  const nodeOpen = useStore($sidebarWorkspaceNodeOpen)
  const listGroupIds = useStore($sidebarListGroupIds)
  const projects = useStore($projectTree)
  const hasCost = useStore($sessionsHaveCost)
  const unreadIds = useStore($unreadFinishedSessionIds)
  // PR state comes from `gh` on whichever machine holds the checkout — Electron
  // locally, the gateway's REST mirror remotely. Resolved per render, not once
  // at module load: switching to a remote profile swaps the bridge underneath.
  const prAvailable = Boolean(desktopGit()?.review?.prList)

  // Fold the level in view: project rows, or the date/status buckets. Project
  // rows default open, so "all collapsed" means every one of them has been
  // explicitly shut. Never sweeps Pinned or Cron.
  const foldIds =
    grouping === 'project'
      ? projects.map(project => project.id)
      : grouping === 'date' || grouping === 'status'
        ? listGroupIds
        : []

  const foldCollapsed = foldIds.length > 0 && foldIds.every(id => nodeOpen[id] === false)

  const filter = t.sidebar.filter
  /** The option tables carry keys, not sentences — resolve them here so the
   *  whole menu follows the active locale. */
  const localized = <T extends string>(options: Option<T>[]): Option<T>[] =>
    options.map(option => (option.labelKey ? { ...option, label: filter.option[option.labelKey] } : option))

  const groupings = localized(GROUPINGS).map(option =>
    option.id === 'profile' ? { ...option, label: t.sidebar.gatewayGroups.grouping } : option
  )

  const groupingLabel = groupings.find(option => option.id === grouping)?.label

  // Two options are conditional: dragging a row is what picks manual, so it
  // only appears as a way back out once there's a hand-picked order to leave;
  // and cost is hidden until some session actually reports spend.
  const orderings = localized(
    ORDERINGS.filter(option => {
      if (option.id === 'manual') {
        return ordering === 'manual'
      }

      return option.id !== 'cost' || hasCost || ordering === 'cost'
    })
  )

  const rowMetaOptions = localized(
    ROW_META.filter(option => {
      if (option.id === 'cost') {
        return hasCost || rowMeta.includes('cost')
      }

      // Preview is a card line; the one-line row has nowhere to put it.
      if (option.id === 'preview') {
        return cardRows
      }

      return option.id !== 'pr' || prAvailable
    })
  )

  const prFilters = localized(PR_FILTERS)
  const statusFilters = localized(STATUS_FILTERS)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={filter.aria}
          className={cn(
            className,
            'data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100',
            // Active filters read as "this control is engaged", the same way the
            // open menu does — never as an accent, which the sidebar reserves
            // for a session that is actually doing something.
            filtersActive && 'bg-(--ui-control-active-background) text-foreground opacity-100'
          )}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="list-filter" size="0.75rem" />
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="min-w-52">
        <DropdownMenuGroup>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger hideChevron>
              {filter.grouping}
              <span className="ml-auto flex items-center gap-1 pl-4 text-(--ui-text-tertiary)">
                {groupingLabel}
                <Codicon name="chevron-right" size="1rem" />
              </span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup
                onValueChange={value => setSidebarGrouping(value as SidebarGrouping)}
                value={grouping}
              >
                {groupings.map(option => (
                  <OptionRadio key={option.id} option={option} />
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>

          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{filter.ordering}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup
                onValueChange={value => setSidebarOrdering(value as SidebarOrdering)}
                value={ordering}
              >
                {orderings.map(option => (
                  <OptionRadio key={option.id} option={option} />
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>

          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{filter.show}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {rowMetaOptions.map(option => (
                <OptionCheckbox
                  checked={rowMeta.includes(option.id)}
                  key={option.id}
                  onCheck={() => toggleSidebarRowMeta(option.id)}
                  option={option}
                />
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>

          {grouping === 'project' && (
            <OptionCheckbox
              checked={showAllSessions}
              onCheck={() => setSidebarShowAllSessions(!showAllSessions)}
              option={{ icon: 'list-unordered', id: 'all-sessions', label: t.sidebar.projects.showAllSessions }}
            />
          )}

          {/* A render variant, not a grouping: three-line cards (project · age /
              title / model · size) compose with whichever grouping is active. */}
          <OptionCheckbox
            checked={cardRows}
            onCheck={() => setSidebarCardRows(!cardRows)}
            option={{ icon: 'inbox', id: 'card-rows', label: filter.inboxStyle }}
          />

          {/* The colored strip at the sidebar foot. Off, the statusbar grows a
              profile dropdown beside the gateway switcher, so nobody loses the
              door — this is for people whose profiles are bots, not workspaces. */}
          <OptionCheckbox
            checked={profileRailVisible}
            onCheck={toggleProfileRailVisible}
            option={{ icon: 'organization', id: 'profile-rail', label: t.sidebar.profileRail }}
          />
        </DropdownMenuGroup>

        <DropdownMenuSeparator />

        <DropdownMenuGroup>
          <DropdownMenuLabel>{filter.groupLabel}</DropdownMenuLabel>

          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{filter.status}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {statusFilters.map(option => (
                <OptionCheckbox
                  checked={statusFilter.includes(option.id)}
                  key={option.id}
                  onCheck={() => toggleSidebarStatusFilter(option.id)}
                  option={option}
                />
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>

          {/* `gh` only exists where the checkout does, so on a remote backend
              this submenu never appears rather than filtering everything out. */}
          {prAvailable && (
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>{filter.pullRequest}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {prFilters.map(option => (
                  <OptionCheckbox
                    checked={prFilter.includes(option.id)}
                    key={option.id}
                    onCheck={() => toggleSidebarPrFilter(option.id)}
                    option={option}
                  />
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )}

          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{filter.profile}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="max-h-80 overflow-y-auto">
              {/* Scoped to one profile the rail is already the filter, so the
                  per-profile boxes only appear where they can narrow something.
                  The actions below stand on their own. */}
              {narrowsByProfile && (
                <>
                  {profileNames.map(name => (
                    <OptionCheckbox
                      checked={profileFilter.includes(name)}
                      key={name}
                      onCheck={() => toggleSidebarProfileFilter(name)}
                      option={{ icon: 'account', id: name, label: name }}
                    />
                  ))}
                  <DropdownMenuSeparator />
                </>
              )}
              <DropdownMenuItem onSelect={requestProfileCreate}>{t.profiles.newProfile}</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => void runImportProfileFlow()}>
                {t.profiles.importProfile}
              </DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>

          {projects.length > 1 && (
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>{filter.project}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="max-h-80 overflow-y-auto">
                {projects.map(project => (
                  <OptionCheckbox
                    checked={projectFilter.includes(project.id)}
                    key={project.id}
                    onCheck={() => toggleSidebarProjectFilter(project.id)}
                    option={{
                      icon: project.isNoProject ? 'home' : 'root-folder',
                      id: project.id,
                      // Home is synthetic, so its label is ours to translate.
                      label: project.isNoProject ? t.sidebar.projects.home : project.label
                    }}
                  />
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )}

          {/* Off by default: one profile's sessions are what the rail selected.
              Nothing to widen to until a second profile exists — but stay
              visible while it's on, or deleting your way back down to one
              profile would strand the sidebar in a mode nothing can leave (the
              rail hides its switcher at one profile too). */}
          {(profileNames.length > 1 || showAllProfiles) && (
            <OptionCheckbox
              checked={showAllProfiles}
              onCheck={toggleShowAllProfiles}
              option={{ id: 'all-profiles', label: t.profiles.allProfiles }}
            />
          )}

          <OptionCheckbox
            checked={showArchived}
            onCheck={() => setSidebarShowArchived(!showArchived)}
            option={{ id: 'archived', label: filter.archived }}
          />

          {/* One way back rather than two near-identical ones: this drops the
              grouping and sort too, which "clear filters" left behind. */}
          {viewCustomized && <DropdownMenuItem onSelect={resetSidebarView}>{filter.reset}</DropdownMenuItem>}
        </DropdownMenuGroup>

        <DropdownMenuSeparator />

        {foldIds.length > 0 && (
          <DropdownMenuItem onSelect={() => setWorkspaceNodesOpen(foldIds, foldCollapsed)}>
            {foldCollapsed ? filter.expandAll : filter.collapseAll}
          </DropdownMenuItem>
        )}
        <DropdownMenuItem disabled={unreadIds.length === 0} onSelect={markAllSessionsRead}>
          {t.sidebar.markAllRead}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
