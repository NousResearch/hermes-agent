import {
  atom,
  Button,
  cn,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  host,
  icons,
  type PluginStorage,
  ToggleRow,
  usePluginI18n,
  useValue,
} from '@hermes/plugin-sdk'

/** Storage key for the Appearance-page toggle. Defaults to on (today's behaviour). */
export const SHOW_PICKER_KEY = 'showPicker'

/** Whether the picker is shown in the composer toolbar. Persisted via the
 *  plugin's own storage (the kanban `storage.get/set` pattern). */
export const $showPicker = atom<boolean>(true)

/** Hydrate `$showPicker` from storage and keep storage in sync. Returns a
 *  disposer the host runs on unload/disable (via `ctx.onDispose`). */
export function bindVisibility(storage: PluginStorage): () => void {
  $showPicker.set(storage.get(SHOW_PICKER_KEY, true))

  return $showPicker.listen(value => storage.set(SHOW_PICKER_KEY, value))
}

// The model pill's exact chrome (`model-pill.tsx` PILL): the operator asked
// for the same design, so the class list matches verbatim.
const PILL = cn(
  'h-(--composer-control-size) min-w-0 shrink gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

const SIDEBAR_BUTTON =
  'text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground'

/** The composer-toolbar project picker: an action menu, not a status readout.
 *  Picking a project starts a FRESH draft anchored at that project root — the
 *  same door as the sidebar's project “new session”. It never touches the
 *  current conversation or any profile/global default. Active profile only:
 *  the host verbs throw while viewing all profiles, and the picker says so.
 *
 *  Reactive: the tree arrives over an async gateway call after the composer
 *  mounts, so the rows come from `host.projects.$list` (a `useValue`
 *  subscription), never a one-shot `list()` at mount — that read is empty
 *  forever when the tree hasn't landed yet. `list()` is still called each
 *  render as the all-profiles gate: it throws exactly when the scope (not
 *  the data) is the reason the list is empty. Strings come from this plugin's
 *  own locale bundle (`./i18n`), so the control follows the app language.
 *
 *  Hidden entirely while the Appearance-page toggle is off, and hidden once
 *  the composer is inside a conversation — it renders only on a fresh draft
 *  (no focused session yet). */
export function ProjectPicker({ surface = 'composer' }: { surface?: 'composer' | 'sidebar' }) {
  const projects = useValue(host.projects.$list)
  const show = useValue($showPicker)
  const focusedSessionId = useValue(host.state.focusedSessionId)
  const t = usePluginI18n('project-picker')
  const inSidebar = surface === 'sidebar'

  if (!show || (!inSidebar && focusedSessionId)) {
    return null
  }

  let blocked = false

  try {
    host.projects.list()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (/all profiles/i.test(message)) {
      blocked = true
    } else {
      return null
    }
  }

  if (blocked) {
    return (
      <span
        className="truncate px-1 text-[0.6875rem] text-(--ui-text-tertiary)"
        title={t('picker.blockedTitle')}
      >
        {t('picker.blocked')}
      </span>
    )
  }

  if (projects.length === 0) {
    return null
  }

  const pick = (projectId: string) => {
    try {
      host.projects.openNewSession({ projectId })
    } catch (error) {
      host.notify({
        kind: 'error',
        message: error instanceof Error ? error.message : t('picker.error'),
      })
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {inSidebar ? (
          <Button
            aria-label={t('picker.sidebarLabel')}
            className={SIDEBAR_BUTTON}
            size="icon-xs"
            title={t('picker.sidebarLabel')}
            type="button"
            variant="ghost"
          >
            <icons.FolderOpen className="size-3.5" />
          </Button>
        ) : (
          <Button
            aria-label={t('picker.label')}
            className={PILL}
            title={t('picker.title')}
            type="button"
            variant="ghost"
          >
            <span className="truncate">{t('picker.placeholder')}</span>
            <icons.ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        )}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" side={inSidebar ? 'right' : 'top'} sideOffset={8}>
        {projects.map(project => (
          <DropdownMenuItem key={project.id} onSelect={() => pick(project.id)}>
            <span className="truncate">{project.label}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** The Appearance-page toggle for the picker — the app's own `ToggleRow`, so
 *  it lines up with the surrounding settings rows. */
export function ProjectPickerSettings() {
  const show = useValue($showPicker)
  const t = usePluginI18n('project-picker')

  return (
    <ToggleRow
      checked={show}
      description={t('settings.description')}
      label={t('settings.label')}
      onChange={on => $showPicker.set(on)}
    />
  )
}
