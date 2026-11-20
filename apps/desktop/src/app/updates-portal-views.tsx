import './discovery/discovery.css'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import type { DesktopUpdateStage, DesktopUpdateStatus } from '@/global'
import { useI18n } from '@/i18n'
import type { UpdateApplyState } from '@/store/updates'

/** The rows a card shows before it says how many more there are. */
const PITCH_ROWS = 5

/** The updates dialog's width: the redesign's pitch and progress take the conversation column (E2,
 *  E3); every other view, and classic, keep the dialog's own. */
export function updatesDialogClass(
  portal: boolean,
  phase: string,
  status: DesktopUpdateStatus | null,
  updateAvailable: boolean,
  isBackend: boolean
): string {
  const installable = Boolean(status?.supported && !status.error) && (isBackend || !status?.retirement)
  const pitching = phase === 'idle' && updateAvailable && installable

  return portal && (phase === 'applying' || pitching) ? 'updates-portal-dialog' : 'max-w-sm'
}

interface PortalUpdatePitchProps {
  title: string
  body: string
  /** The commit changelog's lines, for when there are no release notes. */
  changelog: readonly string[]
  /** The fetched release notes (items undefined while loading) and where they live. */
  notes: { items?: readonly string[]; kind?: string; url: null | string }
  /** A stable release this checkout already contains: a move back, which brings no changes. */
  rollbackTag: null | string
  onInstall: () => void
  onLater: () => void
}

/** What the update brings, as rows: the notes, else the changelog; nothing while the notes load or
 *  for a move back. */
function pitchItems({ changelog, notes, rollbackTag }: PortalUpdatePitchProps): readonly string[] {
  if (rollbackTag || (notes.url && notes.items === undefined)) {
    return []
  }

  return notes.items?.length ? notes.items : changelog
}

/**
 * The redesign's update details (Figma E2 18677:29477) in the updates dialog: the heading, what the
 * update brings as numbered rows in the welcome's card, Update now and Maybe later, and the release
 * notes link. The rows are the real notes (or changelog), never a fixed list.
 */
export function PortalUpdatePitch(props: PortalUpdatePitchProps) {
  const { body, notes, onInstall, onLater, rollbackTag, title } = props
  const { t } = useI18n()
  const items = pitchItems(props)
  const notesUrl = rollbackTag ? null : notes.url
  const notesLabel = notes.kind === 'release' ? t.updates.releaseNotes : t.settings.about.channel.allChanges
  const installLabel = rollbackTag ? t.settings.about.channel.rollbackAction(rollbackTag) : t.updates.updateNow
  const shown = items.slice(0, PITCH_ROWS)
  const more = items.length - shown.length

  return (
    <div className="discovery-welcome updates-portal">
      <div className="discovery-welcome-heading">
        <DialogTitle className="discovery-welcome-title">{title}</DialogTitle>
        <DialogDescription className="discovery-welcome-subtitle" data-tone="regular">
          {body}
        </DialogDescription>
      </div>
      {shown.length > 0 && (
        <ol className="discovery-welcome-card discovery-check-in-rows">
          {shown.map((item, index) => (
            <li className="updates-portal-row" key={`${index}:${item}`}>
              <span className="discovery-welcome-place-name updates-portal-row-text" title={item}>
                {item}
              </span>
              <span aria-hidden className="discovery-welcome-place-meta">
                {String(index + 1).padStart(2, '0')}
              </span>
            </li>
          ))}
        </ol>
      )}
      {more > 0 && <p className="discovery-welcome-footnote">{t.updates.moreChanges(more)}</p>}
      <div className="discovery-welcome-actions">
        <Button className="discovery-welcome-action" onClick={onInstall}>
          {installLabel}
        </Button>
        <Button className="discovery-welcome-action" onClick={onLater} variant="secondary">
          {t.updates.maybeLater}
        </Button>
      </div>
      {notesUrl && (
        <a
          className="discovery-welcome-footnote updates-portal-link"
          href={notesUrl}
          onClick={event => {
            event.preventDefault()
            void window.hermesDesktop?.openExternal?.(event.currentTarget.href)
          }}
          rel="noreferrer"
          target="_blank"
        >
          {notesLabel}
          <Codicon aria-hidden className="discovery-rename-arrow" name="arrow-right" />
        </a>
      )}
    </div>
  )
}

interface PortalUpdateApplyingProps {
  apply: UpdateApplyState
  label: string
  body: string
  closeNote: string
}

/** The updater's stages this run has reached, in order, each with its newest message. */
function reachedStages(apply: UpdateApplyState): { stage: DesktopUpdateStage; message: string }[] {
  const reached = new Map<DesktopUpdateStage, string>()

  for (const entry of apply.log) {
    reached.set(entry.stage, entry.message)
  }

  if (!reached.has(apply.stage)) {
    reached.set(apply.stage, apply.message)
  }

  return [...reached].map(([stage, message]) => ({ stage, message }))
}

/**
 * The redesign's install progress (Figma E3 18677:29591): "Updating Hermes…" as the heading and each
 * stage the updater has reached as a row, done or under way, with the real progress under them. Figma's
 * rows name the person's work (chats, agents, routines, machines); the updater reports its own
 * stages, not per-area progress, so the rows are those.
 */
export function PortalUpdateApplying({ apply, body, closeNote, label }: PortalUpdateApplyingProps) {
  const { t } = useI18n()
  const stages = reachedStages(apply)

  const percent =
    typeof apply.percent === 'number' && Number.isFinite(apply.percent)
      ? Math.max(2, Math.min(100, Math.round(apply.percent)))
      : null

  return (
    <div className="discovery-welcome updates-portal">
      <div className="discovery-welcome-heading">
        <DialogTitle className="discovery-welcome-title">{label}</DialogTitle>
        <DialogDescription className="discovery-welcome-subtitle" data-tone="regular">
          {body}
        </DialogDescription>
      </div>
      <ul className="discovery-welcome-card discovery-check-in-rows">
        {stages.map(({ message, stage }, index) => {
          const current = index === stages.length - 1

          return (
            <li className="updates-portal-row" data-current={current || undefined} key={stage}>
              <span className="discovery-welcome-place-text">
                <span className="discovery-welcome-place-name">{t.updates.stages[stage] ?? stage}</span>
                {message.trim() && <span className="discovery-welcome-place-line">{message.trim()}</span>}
              </span>
              <Codicon aria-hidden className="updates-portal-state" name={current ? 'loading' : 'check'} />
            </li>
          )
        })}
      </ul>
      <div className="updates-portal-progress">
        <Progress aria-label={label} indeterminate={percent === null} value={percent === null ? 0 : percent / 100} />
      </div>
      <p className="discovery-welcome-footnote">{closeNote}</p>
    </div>
  )
}
