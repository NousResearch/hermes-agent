import {
  Button,
  cn,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  SearchField,
  Tip
} from '@hermes/plugin-sdk'

import { connectionName, groupSelectionElsewhere, seatableGroupBots, useGroupCapability } from './canonical-group-capabilities'
import { botSourceStatus } from './data'
import type { useBots } from './i18n'
import { setActivityToasts } from './roster-actions'
import { clearRosterOrder, hasAnyCustomRosterOrder, type RosterOrderMap } from './roster-order'
import { GatewayKindGlyph } from './roster-sections'
import type { rosterGatewayOptions } from './roster-sections'
import type { RosterActivityFilter, RosterKindFilter, RosterRow } from './types'

interface NewGroupChatItemProps {
  label: string
  /** The dialog's "these bots all live on <connection>" note. */
  elsewhereNote: (connection: string) => string
  onSelect: () => void
  roster: RosterRow[]
}

/** Mounted only while the menu is open, so the capability read happens on
 *  demand. Offered only when the dialog it opens can create a room: two
 *  seatable bots by the same predicate the dialog's picker uses, and not all of
 *  them on ONE other connection (every selection there is refused, so the item
 *  says why instead of opening a dead Create button). */
function NewGroupChatItem({ elsewhereNote, label, onSelect, roster }: NewGroupChatItemProps) {
  const { connectionId, mode, seatable } = useGroupCapability(true)
  const seatableBots = seatableGroupBots(roster, { connectionId, mode, seatable })
  const elsewhere = mode === 'canonical' && groupSelectionElsewhere(seatableBots, connectionId)

  return (
    <DropdownMenuItem disabled={seatableBots.length < 2 || elsewhere} onSelect={onSelect}>
      <Codicon className="mr-1.5" name="organization" />
      {elsewhere ? (
        <span className="grid gap-0.5">
          <span>{label}</span>
          <span className="text-xs text-(--ui-text-secondary)" role="note">
            {elsewhereNote(connectionName(seatableBots[0]))}
          </span>
        </span>
      ) : (
        label
      )}
    </DropdownMenuItem>
  )
}

interface renderRosterToolbarProps {
  b: ReturnType<typeof useBots>
  activityToasts: boolean
  activeSourceRoster?: RosterRow[]
  /** Full roster; New Group Chat seats only `seatableGroupBots` of it. */
  roster: RosterRow[]
  setCreateOpen: (value: boolean) => void
  setGroupCreateOpen: (value: boolean) => void
  setSectionDialog: (
    value: null | { bot?: RosterRow; mode: 'create' } | { id: string; mode: 'rename'; name: string }
  ) => void
  showRosterTools: boolean
  showRosterSearch: boolean
  showRosterFilters: boolean
  query: string
  setQuery: (value: string) => void
  activeFilterCount: number
  gatewayOptions: ReturnType<typeof rosterGatewayOptions>
  rowKindFilter: RosterKindFilter
  setRowKindFilter: (value: RosterKindFilter) => void
  activityFilter: RosterActivityFilter
  setActivityFilter: (value: RosterActivityFilter) => void
  gatewayFilter: string
  setGatewayFilter: (value: string) => void
  rosterOrder?: RosterOrderMap
}

export function renderRosterToolbar({
  b,
  activityToasts,
  roster,
  setCreateOpen,
  setGroupCreateOpen,
  setSectionDialog,
  showRosterTools,
  showRosterSearch,
  showRosterFilters,
  query,
  setQuery,
  activeFilterCount,
  gatewayOptions,
  rowKindFilter,
  setRowKindFilter,
  activityFilter,
  setActivityFilter,
  gatewayFilter,
  setGatewayFilter,
  rosterOrder
}: renderRosterToolbarProps) {
  return (
    <>
      <div className="flex items-center justify-between gap-2 px-2.5 pt-2.5 pb-1.5">
        <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)">
          {b.roster.title}
        </span>
        <div className="flex items-center gap-0.5">
          <Tip
            label={activityToasts ? b.roster.toastsOn : b.roster.toastsOff}
          >
            <Button
              className="rounded-md text-(--ui-text-tertiary) hover:text-foreground"
              onClick={() => setActivityToasts(!activityToasts)}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name={activityToasts ? 'bell' : 'bell-slash'} />
            </Button>
          </Tip>
          <DropdownMenu>
            <Tip label={b.roster.newMenu}>
              <DropdownMenuTrigger asChild>
                <Button
                  aria-label={b.roster.newBotOrGroup}
                  className="rounded-md text-(--ui-text-tertiary) hover:text-foreground"
                  size="icon-xs"
                  variant="ghost"
                >
                  <Codicon name="add" />
                </Button>
              </DropdownMenuTrigger>
            </Tip>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={() => setCreateOpen(true)}>
                <Codicon className="mr-1.5" name="hubot" />
                {b.bot.newTitle}
              </DropdownMenuItem>
              <NewGroupChatItem
                elsewhereNote={b.crossGateway.selectionElsewhere}
                label={b.group.newTitle}
                onSelect={() => setGroupCreateOpen(true)}
                roster={roster}
              />
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => setSectionDialog({ mode: 'create' })}>
                <Codicon className="mr-1.5" name="new-folder" />
                {b.sections.newSection}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {showRosterTools ? (
        <div className="flex min-w-0 items-center gap-1 px-2.5 pb-1.5">
          {showRosterSearch ? (
            <SearchField
              aria-label={b.roster.search}
              containerClassName={cn('min-w-0 flex-1', query ? 'opacity-100!' : 'opacity-50 focus-within:opacity-100')}
              inputClassName="w-full text-[0.75rem] placeholder:text-(--ui-text-tertiary)"
              key={'roster-search'}
              onChange={setQuery}
              placeholder={b.roster.searchPlaceholder}
              value={query}
            />
          ) : (
            <span className="min-w-0 flex-1" key={'roster-search-spacer'} />
          )}
          {showRosterFilters ? (
            <DropdownMenu key={'roster-filters'}>
              <Tip label={activeFilterCount ? b.roster.filtersActive(activeFilterCount) : b.roster.filter}>
                <DropdownMenuTrigger asChild>
                  <Button
                    aria-label={activeFilterCount ? b.roster.filtersActive(activeFilterCount) : b.roster.filter}
                    className={cn(
                      'size-7 shrink-0 rounded-md text-(--ui-text-tertiary) hover:text-foreground',
                      activeFilterCount && 'text-(--ui-accent)'
                    )}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="list-filter" />
                  </Button>
                </DropdownMenuTrigger>
              </Tip>
              <DropdownMenuContent align="end">
                {(
                  [
                    ['all', b.roster.botsAndGroups],
                    ['bots', b.roster.botsOnly],
                    ['groups', b.roster.groupsOnly]
                  ] as [RosterKindFilter, string][]
                ).map(([value, label]) => (
                  <DropdownMenuItem key={`kind:${value}`} onSelect={() => setRowKindFilter(value)}>
                    <span className="min-w-0 flex-1">{label}</span>
                    {rowKindFilter === value ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                {(
                  [
                    ['all', b.roster.anyActivity],
                    ['active', b.roster.activeNow],
                    ['recent', b.roster.recentlyActive],
                    ['older', b.roster.older]
                  ] as [RosterActivityFilter, string][]
                ).map(([value, label]) => (
                  <DropdownMenuItem key={`activity:${value}`} onSelect={() => setActivityFilter(value)}>
                    <span className="min-w-0 flex-1">{label}</span>
                    {activityFilter === value ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ))}
                {gatewayOptions.length > 1 ? <DropdownMenuSeparator /> : null}
                {gatewayOptions.length > 1 ? (
                  <DropdownMenuItem onSelect={() => setGatewayFilter('all')}>
                    <Codicon className="mr-1.5" name="globe" />
                    <span className="min-w-0 flex-1">{b.roster.allGateways}</span>
                    {gatewayFilter === 'all' ? <Codicon name="check" /> : null}
                  </DropdownMenuItem>
                ) : null}
                {gatewayOptions.length > 1
                  ? gatewayOptions.map(option => {
                      const status = botSourceStatus({
                        sourceError: option.error,
                        sourceReachable: option.reachable
                      })

                      return (
                        <DropdownMenuItem
                          key={option.connectionId}
                          onSelect={() => setGatewayFilter(option.connectionId)}
                        >
                          <GatewayKindGlyph
                            className={cn('mr-1.5', !status.available && 'text-amber-600 dark:text-amber-300')}
                            kind={option.kind}
                          />
                          <span className="min-w-0 flex-1 truncate">{option.label || option.connectionId}</span>
                          <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
                            {option.count}
                          </span>
                          {gatewayFilter === option.connectionId ? <Codicon name="check" /> : null}
                        </DropdownMenuItem>
                      )
                    })
                  : []}
                {activeFilterCount ? <DropdownMenuSeparator /> : null}
                {activeFilterCount ? (
                  <DropdownMenuItem
                    onSelect={() => {
                      setRowKindFilter('all')
                      setActivityFilter('all')
                      setGatewayFilter('all')
                    }}
                  >
                    {b.roster.clearFilters}
                  </DropdownMenuItem>
                ) : null}
                {rosterOrder && hasAnyCustomRosterOrder(rosterOrder) ? (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onSelect={() => clearRosterOrder()}>
                      <Codicon className="mr-1.5" name="clear-all" />
                      {b.sections.resetOrder}
                    </DropdownMenuItem>
                  </>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
        </div>
      ) : null}
    </>
  )
}
