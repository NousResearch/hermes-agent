import { useStore } from '@nanostores/react'
import { useState } from 'react'

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
import { desktopGit } from '@/lib/desktop-git'
import { cn } from '@/lib/utils'
import {
  $sidebarCardRows,
  $sidebarFiltersActive,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarRowMeta,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  $sidebarViewCustomized,
  $sidebarWorkspaceNodeOpen,
  resetSidebarView,
  setSidebarCardRows,
  setSidebarGrouping,
  setSidebarOrdering,
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
import { runImportProfileFlow } from '@/store/profile-share'
import { $projectTree } from '@/store/projects'
import type { PullRequestBucket } from '@/store/pull-requests'
import { $unreadFinishedSessionIds, markAllSessionsRead } from '@/store/session'
import type { SessionStatusBucket } from '@/store/session-dot-state'
import { $sessionsHaveCost } from '@/store/sidebar-archive'
import {
  $activeSavedSidebarViewId,
  $savedSidebarViews,
  applySavedSidebarView,
  type SavedSidebarView,
  savedSidebarViewRequiresProfileSwitch
} from '@/store/sidebar-views'

import { SidebarViewDialog, type SidebarViewDialogState } from './sidebar-view-dialog'

interface Option<T extends string = string> {
  /** A status dot's full className, from the row's own vocabulary. */
  dot?: string
  icon?: string
  id: T
  label: string
}

type OptionSpec<T extends string = string> = Omit<Option<T>, 'label'>

const GROUPINGS: OptionSpec<SidebarGrouping>[] = [
  { icon: 'clock', id: 'date' },
  { icon: 'list-flat', id: 'none' },
  { icon: 'root-folder', id: 'project' },
  { icon: 'pulse', id: 'status' },
  { icon: 'account', id: 'profile' }
]

const ORDERINGS: OptionSpec<SidebarOrdering>[] = [
  { icon: 'clock', id: 'updated' },
  { icon: 'add', id: 'created' },
  { icon: 'pulse', id: 'status' },
  { icon: 'symbol-numeric', id: 'tokens' },
  { icon: 'credit-card', id: 'cost' },
  { icon: 'list-ordered', id: 'manual' }
]

const ROW_META: OptionSpec<SidebarRowMeta>[] = [
  { icon: 'clock', id: 'updated' },
  { icon: 'comment', id: 'preview' },
  { icon: 'symbol-numeric', id: 'tokens' },
  { icon: 'credit-card', id: 'cost' },
  { icon: 'git-pull-request', id: 'pr' },
  { icon: 'account', id: 'profile' }
]

const PR_FILTERS: OptionSpec<PullRequestBucket>[] = [
  { icon: 'git-pull-request', id: 'open' },
  { icon: 'git-pull-request-draft', id: 'draft' },
  { icon: 'git-merge', id: 'merged' },
  { icon: 'git-pull-request-closed', id: 'closed' },
  { icon: 'circle-slash', id: 'none' }
]

const STATUS_FILTERS: OptionSpec<SessionStatusBucket>[] = [
  { dot: sessionDotClassName('needs-input'), id: 'needs-input' },
  { dot: sessionDotClassName('working'), id: 'working' },
  { dot: sessionDotClassName('unread'), id: 'unread' },
  { dot: sessionDotClassName('draft'), id: 'draft' },
  { dot: cn(sessionDotClassName('idle'), 'bg-(--ui-text-quaternary)'), id: 'idle' }
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
  const copy = t.sidebar.viewMenu
  const [dialog, setDialog] = useState<SidebarViewDialogState | null>(null)
  const grouping = useStore($sidebarGrouping)
  const ordering = useStore($sidebarOrdering)
  const rowMeta = useStore($sidebarRowMeta)
  const cardRows = useStore($sidebarCardRows)
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
  const projects = useStore($projectTree)
  const hasCost = useStore($sessionsHaveCost)
  const unreadIds = useStore($unreadFinishedSessionIds)
  const savedViews = useStore($savedSidebarViews).views
  const activeSavedViewId = useStore($activeSavedSidebarViewId)
  const activeSavedView = savedViews.find(view => view.id === activeSavedViewId)
  // PR state comes from `gh` on whichever machine holds the checkout — Electron
  // locally, the gateway's REST mirror remotely. Resolved per render, not once
  // at module load: switching to a remote profile swaps the bridge underneath.
  const prAvailable = Boolean(desktopGit()?.review?.prList)
  // Project rows default open, so "all collapsed" means every one of them has
  // been explicitly shut.
  const projectsCollapsed = projects.length > 0 && projects.every(project => nodeOpen[project.id] === false)

  const groupings: Option<SidebarGrouping>[] = GROUPINGS.map(option => ({
    ...option,
    label: copy.groupings[option.id]
  }))

  const orderingOptions: Option<SidebarOrdering>[] = ORDERINGS.map(option => ({
    ...option,
    label: copy.orderings[option.id]
  }))

  const rowMetaOptionsTranslated: Option<SidebarRowMeta>[] = ROW_META.map(option => ({
    ...option,
    label: copy.metadata[option.id]
  }))

  const prFilterOptions: Option<PullRequestBucket>[] = PR_FILTERS.map(option => ({
    ...option,
    label: copy.pullRequestFilters[option.id]
  }))

  const statusFilterOptions: Option<SessionStatusBucket>[] = STATUS_FILTERS.map(option => ({
    ...option,
    label: copy.statusFilters[option.id]
  }))

  const groupingLabel = groupings.find(option => option.id === grouping)?.label

  // Two options are conditional: dragging a row is what picks manual, so it
  // only appears as a way back out once there's a hand-picked order to leave;
  // and cost is hidden until some session actually reports spend.
  const orderings = orderingOptions.filter(option => {
    if (option.id === 'manual') {
      return ordering === 'manual'
    }

    return option.id !== 'cost' || hasCost || ordering === 'cost'
  })

  const rowMetaOptions = rowMetaOptionsTranslated.filter(option => {
    if (option.id === 'cost') {
      return hasCost || rowMeta.includes('cost')
    }

    // Preview is a card line; the one-line row has nowhere to put it.
    if (option.id === 'preview') {
      return cardRows
    }

    return option.id !== 'pr' || prAvailable
  })

  const editView = (kind: 'delete' | 'rename' | 'update', view: SavedSidebarView) => setDialog({ kind, view })

  const selectView = (view: SavedSidebarView) => {
    if (savedSidebarViewRequiresProfileSwitch(view)) {
      setDialog({ kind: 'apply', view })
    } else {
      applySavedSidebarView(view.id)
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={copy.filters}
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
            {savedViews.length > 0 && (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger hideChevron>
                  {copy.savedViews}
                  <span className="ml-auto flex max-w-28 items-center gap-1 pl-4 text-(--ui-text-tertiary)">
                    <span className="truncate">{activeSavedView?.name ?? '—'}</span>
                    <Codicon name="chevron-right" size="1rem" />
                  </span>
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {savedViews.map(view => (
                    <DropdownMenuSub key={view.id}>
                      <DropdownMenuSubTrigger>
                        <span className="flex w-3 shrink-0 items-center justify-center">
                          {activeSavedViewId === view.id && <Codicon name="check" size="0.75rem" />}
                        </span>
                        <span className="max-w-48 truncate">{view.name}</span>
                      </DropdownMenuSubTrigger>
                      <DropdownMenuSubContent>
                        <DropdownMenuItem onSelect={() => selectView(view)}>
                          {copy.useView}
                        </DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => editView('update', view)}>{copy.updateCurrent}</DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => editView('rename', view)}>{copy.rename}</DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onSelect={() => editView('delete', view)} variant="destructive">
                          {copy.delete}
                        </DropdownMenuItem>
                      </DropdownMenuSubContent>
                    </DropdownMenuSub>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            )}

            <DropdownMenuItem onSelect={() => setDialog({ kind: 'save' })}>
              <Codicon name="save" size="0.8125rem" />
              {copy.saveCurrent}
            </DropdownMenuItem>

            <DropdownMenuSeparator />

            <DropdownMenuSub>
              <DropdownMenuSubTrigger hideChevron>
                {copy.grouping}
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
              <DropdownMenuSubTrigger>{copy.ordering}</DropdownMenuSubTrigger>
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
              <DropdownMenuSubTrigger>{copy.show}</DropdownMenuSubTrigger>
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

            {/* A render variant, not a grouping: three-line cards (project · age /
              title / model · size) compose with whichever grouping is active. */}
            <OptionCheckbox
              checked={cardRows}
              onCheck={() => setSidebarCardRows(!cardRows)}
              option={{ icon: 'inbox', id: 'card-rows', label: copy.inboxStyle }}
            />
          </DropdownMenuGroup>

          <DropdownMenuSeparator />

          <DropdownMenuGroup>
            <DropdownMenuLabel>{copy.filters}</DropdownMenuLabel>

            <DropdownMenuSub>
              <DropdownMenuSubTrigger>{copy.status}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {statusFilterOptions.map(option => (
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
                <DropdownMenuSubTrigger>{copy.pullRequest}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {prFilterOptions.map(option => (
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
              <DropdownMenuSubTrigger>{copy.profile}</DropdownMenuSubTrigger>
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
                <DropdownMenuSubTrigger>{copy.project}</DropdownMenuSubTrigger>
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
              option={{ id: 'archived', label: copy.archived }}
            />

            {/* One way back rather than two near-identical ones: this drops the
              grouping and sort too, which "clear filters" left behind. */}
            {viewCustomized && <DropdownMenuItem onSelect={resetSidebarView}>{copy.resetToDefaults}</DropdownMenuItem>}
          </DropdownMenuGroup>

          <DropdownMenuSeparator />

          {/* Only the project rows fold, and only when they're what you're
            looking at — sweeping Pinned and Cron shut alongside them is not
            what "collapse all" means here. Their lanes underneath keep their
            own state, so re-opening a project shows it as you left it. */}
          {grouping === 'project' && projects.length > 0 && (
            <DropdownMenuItem
              onSelect={() =>
                setWorkspaceNodesOpen(
                  projects.map(project => project.id),
                  projectsCollapsed
                )
              }
            >
              {projectsCollapsed ? copy.expandAll : copy.collapseAll}
            </DropdownMenuItem>
          )}
          <DropdownMenuItem disabled={unreadIds.length === 0} onSelect={markAllSessionsRead}>
            {copy.markAllRead}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <SidebarViewDialog dialog={dialog} onClose={() => setDialog(null)} />
    </>
  )
}
