import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ProjectAppearancePicker } from '@/app/chat/sidebar/projects/project-appearance'
import { Button } from '@/components/ui/button'
import { Codicon, codiconIcon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import {
  $defaultProjectId,
  $projects,
  $projectsRpcAvailable,
  deleteProject,
  openProjectAddFolder,
  openProjectCreate,
  openProjectRename,
  removeProjectFolder,
  setDefaultProject,
  setProjectAppearance,
  setProjectFolderPrimary
} from '@/store/projects'
import type { ProjectInfo } from '@/types/hermes'

import { Pill, SettingsSection } from './primitives'

// Settings → Workspace → Projects: central management for the saved,
// per-profile projects. Every mutation routes through the projects store —
// the same functions the sidebar's kebab menu drives — so the optimistic
// cache/rollback contract and profile scoping stay in one place. Create,
// rename and add-folder arm the single sidebar-mounted ProjectDialog through
// the $projectDialog atom; this panel never mounts a second dialog.
export function WorkspaceProjectsSettings() {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const projects = useStore($projects)
  const defaultId = useStore($defaultProjectId)
  const rpcAvailable = useStore($projectsRpcAvailable)

  // One in-flight write at a time. The key names the write for readability,
  // but run() no-ops on any second call while one is in flight — so every
  // run-controlled control locks for the duration, not just the affected row's.
  const [busyKey, setBusyKey] = useState<null | string>(null)

  const [confirm, setConfirm] = useState<
    null | { folderPath?: string; kind: 'delete-project' | 'remove-folder'; project: ProjectInfo }
  >(null)

  const run = async (key: string, work: () => Promise<unknown>, fallback: string) => {
    if (busyKey) {
      return
    }

    setBusyKey(key)

    try {
      await work()
    } catch (err) {
      notifyError(err, fallback)
    } finally {
      setBusyKey(null)
    }
  }

  const visible = projects.filter(project => !project.archived)

  return (
    <SettingsSection
      aside={
        rpcAvailable !== false && visible.length > 0 ? (
          <Button onClick={openProjectCreate} size="sm" type="button">
            <Codicon name="add" size="0.75rem" />
            {p.newButton}
          </Button>
        ) : undefined
      }
      icon={SECTION_ICON}
      title={t.settings.subpages.workspaceProjects}
    >
      {rpcAvailable === false ? (
        <p className="rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
          {p.staleBackend}
        </p>
      ) : visible.length === 0 ? (
        <div className="flex flex-col items-start gap-2 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-4">
          <span className="text-[length:var(--conversation-text-font-size)] text-(--ui-text-secondary)">
            {p.noProjects}
          </span>
          <Button onClick={openProjectCreate} size="sm" type="button" variant="ghost">
            <Codicon name="add" size="0.75rem" />
            {p.newButton}
          </Button>
        </div>
      ) : (
        <ul className="flex flex-col gap-2" role="list">
          {visible.map(project => (
            <ProjectCard
              busyKey={busyKey}
              defaultId={defaultId}
              key={project.id}
              onConfirm={setConfirm}
              project={project}
              run={run}
            />
          ))}
        </ul>
      )}

      <ConfirmDialog
        confirmLabel={confirm?.kind === 'remove-folder' ? p.removeFolder : p.menuDelete}
        description={confirm?.kind === 'remove-folder' ? p.removeFolderConfirm : p.deleteConfirm}
        destructive
        dismissOnConfirm
        onClose={() => setConfirm(null)}
        onConfirm={() => {
          const target = confirm

          if (!target) {
            return
          }

          if (target.kind === 'remove-folder') {
            return run(
              `remove-folder:${target.project.id}:${target.folderPath}`,
              () => removeProjectFolder(target.project.id, target.folderPath ?? ''),
              p.removeFolderFailed
            )
          }

          return run(`delete:${target.project.id}`, () => deleteProject(target.project.id), p.menuDelete)
        }}
        open={confirm !== null}
        title={
          confirm?.kind === 'remove-folder'
            ? p.removeFolder
            : `${p.menuDelete} "${confirm?.project.name ?? ''}"?`
        }
      />
    </SettingsSection>
  )
}

function iconFor(name: string): IconComponent {
  return codiconIcon(name)
}

const SECTION_ICON = iconFor('folder-library')

interface ProjectCardProps {
  project: ProjectInfo
  defaultId: null | string
  busyKey: null | string
  run: (key: string, work: () => Promise<unknown>, fallback: string) => Promise<void>
  onConfirm: (
    next: null | { folderPath?: string; kind: 'delete-project' | 'remove-folder'; project: ProjectInfo }
  ) => void
}

function ProjectCard({ project, defaultId, busyKey, run, onConfirm }: ProjectCardProps) {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const [appearanceOpen, setAppearanceOpen] = useState(false)
  const rowBusy = busyKey?.startsWith(`delete:${project.id}`) ?? false
  // run() ignores any second mutation while a write is in flight, so a control
  // left enabled during that window is a dead click — lock them all.
  const anyBusy = busyKey !== null
  const isDefault = defaultId === project.id
  const folders = project.folders ?? []

  return (
    <li
      aria-label={project.name}
      className="rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2.5"
      role="listitem"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="grid size-6 shrink-0 place-items-center"
          data-testid="project-glyph"
          style={project.color ? { color: project.color } : undefined}
        >
          {project.color && !project.icon ? (
            <span aria-hidden="true" className="size-2 rounded-full" style={{ backgroundColor: project.color }} />
          ) : (
            <Codicon name={project.icon || 'folder-library'} size="1rem" />
          )}
        </span>
        <span className="min-w-0 flex-1 truncate text-[length:var(--conversation-text-font-size)] font-medium">
          {project.name}
        </span>
        {isDefault && <Pill>{p.defaultForNewChats}</Pill>}
        <Switch
          aria-label={`${p.defaultForNewChats} ${project.name}`}
          checked={isDefault}
          disabled={anyBusy}
          onCheckedChange={on => {
            void run(`default:${project.id}`, () => setDefaultProject(on ? project.id : null), p.setDefaultFailed)
          }}
        />
        <div className="flex shrink-0 items-center gap-1">
          <Button
            disabled={rowBusy}
            onClick={() => openProjectRename({ id: project.id, name: project.name })}
            size="sm"
            type="button"
            variant="ghost"
          >
            {p.menuRename}
          </Button>
          <Popover onOpenChange={setAppearanceOpen} open={appearanceOpen}>
            <PopoverTrigger asChild>
              <Button disabled={rowBusy} size="sm" type="button" variant="ghost">
                {p.menuAppearance}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-2">
              <ProjectAppearancePicker
                color={project.color}
                icon={project.icon}
                noColorLabel={p.noColor}
                onColor={color => {
                  void run(
                    `appearance:${project.id}`,
                    () =>
                      setProjectAppearance(
                        {
                          color: project.color,
                          icon: project.icon,
                          id: project.id,
                          isAuto: false,
                          label: project.name,
                          path: project.primary_path
                        },
                        { color }
                      ),
                    t.common.failed
                  )
                }}
                onIcon={icon => {
                  void run(
                    `appearance:${project.id}`,
                    () =>
                      setProjectAppearance(
                        {
                          color: project.color,
                          icon: project.icon,
                          id: project.id,
                          isAuto: false,
                          label: project.name,
                          path: project.primary_path
                        },
                        { icon }
                      ),
                    t.common.failed
                  )
                }}
              />
            </PopoverContent>
          </Popover>
          <Button
            disabled={rowBusy}
            onClick={() => openProjectAddFolder({ id: project.id, name: project.name })}
            size="sm"
            type="button"
            variant="ghost"
          >
            {p.addFolder}
          </Button>
          <Button
            disabled={rowBusy}
            onClick={() => onConfirm({ kind: 'delete-project', project })}
            size="sm"
            type="button"
            variant="ghost"
          >
            {p.menuDelete}
          </Button>
        </div>
      </div>

      {folders.length > 0 && (
        <ul className="mt-1.5 flex flex-col gap-0.5">
          {folders.map(folder => {
            const folderBusy =
              busyKey === `remove-folder:${project.id}:${folder.path}` ||
              busyKey === `primary:${project.id}:${folder.path}`

            return (
              <li
                className="flex min-w-0 items-center gap-2 rounded-md px-1 py-0.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)"
                key={folder.path}
              >
                <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.75rem" />
                <span className="min-w-0 flex-1 truncate" title={folder.path}>
                  {folder.path}
                </span>
                {folder.is_primary && (
                  <span className="shrink-0 text-[0.625rem] uppercase text-(--ui-text-quaternary)">
                    {p.primaryBadge}
                  </span>
                )}
                {!folder.is_primary && folders.length > 1 && (
                  <Tip label={`${t.common.set} ${p.primaryBadge}`}>
                    <Button
                      aria-label={`${t.common.set} ${p.primaryBadge}`}
                      className="size-5 text-(--ui-text-quaternary) hover:text-foreground"
                      disabled={anyBusy || folderBusy}
                      onClick={() =>
                        void run(
                          `primary:${project.id}:${folder.path}`,
                          () => setProjectFolderPrimary(project.id, folder.path),
                          p.setPrimaryFailed
                        )
                      }
                      size="icon-xs"
                      type="button"
                      variant="ghost"
                    >
                      <Codicon name="target" size="0.75rem" />
                    </Button>
                  </Tip>
                )}
                <Tip label={p.removeFolder}>
                  <Button
                    aria-label={p.removeFolder}
                    className={cn('size-5 text-(--ui-text-quaternary) hover:text-foreground')}
                    disabled={anyBusy || folderBusy}
                    onClick={() =>
                      onConfirm({ folderPath: folder.path, kind: 'remove-folder', project })
                    }
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name="close" size="0.75rem" />
                  </Button>
                </Tip>
              </li>
            )
          })}
        </ul>
      )}
    </li>
  )
}
