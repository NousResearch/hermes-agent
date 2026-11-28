import type { ReadableAtom } from 'nanostores'

import type { DesktopUpdateStage } from '@/global'
import { getRuntimeI18nLocale } from '@/i18n/runtime'

/** Stages the update overlay can be closed from. */
type SettledUpdateStage = 'idle' | 'manual' | 'guiSkew' | 'error'

/** Every stage that can hold the overlay on a spinner (`done` does until the
 *  apply call resolves). A new DesktopUpdateStage lands here unless settled. */
export type BusyUpdateStage = Exclude<DesktopUpdateStage, SettledUpdateStage>

const MINUTE = 60_000

/** How long each busy stage may go WITHOUT progress (a new stage, message,
 *  percent or log entry) before the UI stops claiming it is running. A busy
 *  stage missing here is a type error, so no state can spin forever. Expiry
 *  kills nothing: the hand-off script or the backend owns the process, the UI
 *  only owns its spinner. */
export const BUSY_STAGE_STALL_MS = {
  prepare: 5 * MINUTE,
  // The pinned artifact fails after 3 connections without a byte, 120 s each.
  fetch: 10 * MINUTE,
  // The remote backend action's inactivity bound (#137848).
  pull: 6 * MINUTE,
  pydeps: 15 * MINUTE,
  // Above the hand-off scripts' own 10-minute step watchdogs, so theirs speaks first.
  update: 15 * MINUTE,
  rebuild: 20 * MINUTE,
  // A handed-off app should have quit; a backend should have come back.
  restart: 4 * MINUTE,
  done: 5 * MINUTE
} as const satisfies Record<BusyUpdateStage, number>

export function isBusyUpdateStage(stage: DesktopUpdateStage): stage is BusyUpdateStage {
  return Object.hasOwn(BUSY_STAGE_STALL_MS, stage)
}

// Module-level copy: the bundled locale catalogs are past their size cap.
const STALLED_COPY: Record<string, string> = {
  en: 'The update stopped reporting progress. It may still finish in the background; the update log has the details.',
  de: 'Das Update meldet keinen Fortschritt mehr. Es kann im Hintergrund noch abschließen; Details stehen im Update-Log.',
  es: 'La actualización dejó de informar progreso. Puede terminar en segundo plano; el registro de actualización tiene los detalles.',
  fr: 'La mise à jour ne signale plus de progression. Elle peut encore se terminer en arrière-plan ; le journal de mise à jour contient les détails.',
  ja: 'アップデートの進行状況が届かなくなりました。バックグラウンドで完了する可能性があります。詳細はアップデートログを確認してください。',
  ru: 'Обновление перестало сообщать о ходе работы. Оно может завершиться в фоне; подробности — в журнале обновления.',
  zh: '更新已停止报告进度。它可能仍会在后台完成；详情见更新日志。',
  'zh-hant': '更新已停止回報進度。它可能仍會在背景完成；詳情請見更新記錄。',
  ar: 'توقف التحديث عن الإبلاغ عن التقدم. قد يكتمل في الخلفية؛ التفاصيل في سجل التحديث.'
}

export function stalledUpdateMessage(): string {
  return STALLED_COPY[getRuntimeI18nLocale()] ?? STALLED_COPY.en
}

interface StageProgress {
  applying: boolean
  stage: DesktopUpdateStage
  message: string
  percent: number | null
  log: readonly unknown[]
}

/** Arms the busy stage's stall timer and re-arms it on every progress change;
 *  `onStall` runs once per expiry. Returns the unsubscribe. */
export function watchUpdateStageDeadlines(
  store: ReadableAtom<StageProgress>,
  onStall: (stage: BusyUpdateStage) => void
): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined
  let fingerprint = ''
  // The log is capped (its length stops growing), so a new line is detected by
  // the identity of the newest entry: every ingested report appends a new one.
  let newest: unknown

  const unsubscribe = store.subscribe((state: StageProgress): void => {
    const busy = (state.applying || state.stage === 'restart') && isBusyUpdateStage(state.stage)
    const next = busy ? [state.stage, state.message, state.percent].join('\0') : ''
    const entry = busy ? state.log.at(-1) : undefined

    if (next === fingerprint && entry === newest) {
      return
    }

    fingerprint = next
    newest = entry
    clearTimeout(timer)

    if (busy && isBusyUpdateStage(state.stage)) {
      const stage = state.stage
      timer = setTimeout((): void => {
        fingerprint = ''
        newest = undefined
        onStall(stage)
      }, BUSY_STAGE_STALL_MS[stage])
    }
  })

  return (): void => {
    clearTimeout(timer)
    unsubscribe()
  }
}
