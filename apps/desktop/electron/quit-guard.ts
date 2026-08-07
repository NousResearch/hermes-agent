// Quitting with a turn in flight kills the backend mid-tool-call: the work is
// lost, and anything the agent had half-written to disk stays half-written.
// Renderers publish what they're running; the main process asks before it lets
// that go. The decision + copy live here (pure, testable) so main.ts only owns
// the IPC and the dialog call.
//
// Copy is localized to the renderer's chosen display language (pushed to the
// main process via the same channel as the tray menu). The main process has no
// i18n of its own, so we keep a small table here; defaults to English.

const MAX_LISTED = 4

export interface ActiveWork {
  /** Titles of sessions running a turn. Untitled sessions contribute a count only. */
  titles: string[]
  /** Running turns, including untitled ones — always >= titles.length. */
  count: number
}

export const NO_ACTIVE_WORK: ActiveWork = { count: 0, titles: [] }

/** Coerce an IPC payload from an untrusted renderer into an ActiveWork. */
export function normalizeActiveWork(payload: unknown): ActiveWork {
  if (!payload || typeof payload !== 'object') {
    return NO_ACTIVE_WORK
  }

  const raw = payload as { count?: unknown; titles?: unknown }

  const titles = Array.isArray(raw.titles)
    ? raw.titles
        .filter((title): title is string => typeof title === 'string')
        .map(title => title.trim())
        .filter(Boolean)
    : []

  const count = typeof raw.count === 'number' && Number.isFinite(raw.count) ? Math.max(0, Math.floor(raw.count)) : 0

  return { count: Math.max(count, titles.length), titles }
}

/** Merge every window's report into one. Windows can show the same session. */
export function mergeActiveWork(reports: Iterable<ActiveWork>): ActiveWork {
  const titles: string[] = []
  let count = 0

  for (const report of reports) {
    count = Math.max(count, report.count)

    for (const title of report.titles) {
      if (!titles.includes(title)) {
        titles.push(title)
      }
    }
  }

  return { count: Math.max(count, titles.length), titles }
}

// ── Localization ──────────────────────────────────────────────────────────
// The quit-confirm dialog is a main-process native dialog, so it has no access
// to the renderer's i18n catalog. Keep a small table here, keyed by the same
// locale the tray menu uses (pushed via setTrayMenuLocale).

type QuitGuardLocale = 'en' | 'zh' | 'zh-hant' | 'ja' | 'ar'

const QUIT_GUARD_STRINGS: Record<QuitGuardLocale, {
  messageOne: string
  messageMany: (n: number) => string
  warn: string
  moreOne: string
  moreMany: (n: number) => string
  buttons: [string, string]
}> = {
  en: {
    messageOne: 'Hermes is still working on 1 chat.',
    messageMany: n => `Hermes is still working on ${n} chats.`,
    warn: 'Quitting stops the agent mid-turn. Any work it has not finished writing is lost.',
    moreOne: '• 1 more',
    moreMany: n => `• ${n} more`,
    buttons: ['Keep Running', 'Quit Anyway']
  },
  zh: {
    messageOne: 'Hermes 正在处理 1 个对话。',
    messageMany: n => `Hermes 正在处理 ${n} 个对话。`,
    warn: '退出会中断 agent 的当前轮次，未写入的工作将丢失。',
    moreOne: '• 还有 1 个',
    moreMany: n => `• 还有 ${n} 个`,
    buttons: ['继续运行', '仍然退出']
  },
  'zh-hant': {
    messageOne: 'Hermes 正在處理 1 個對話。',
    messageMany: n => `Hermes 正在處理 ${n} 個對話。`,
    warn: '退出會中斷 agent 的當前輪次，未寫入的工作將遺失。',
    moreOne: '• 還有 1 個',
    moreMany: n => `• 還有 ${n} 個`,
    buttons: ['繼續運行', '仍然退出']
  },
  ja: {
    messageOne: 'Hermes は 1 件のチャットを処理中です。',
    messageMany: n => `Hermes は ${n} 件のチャットを処理中です。`,
    warn: '終了すると agent の処理中のターンが中断され、未保存の作業が失われます。',
    moreOne: '• さらに 1 件',
    moreMany: n => `• さらに ${n} 件`,
    buttons: ['続行', '強制終了']
  },
  ar: {
    messageOne: 'ما زال هيرميس يعمل على محادثة واحدة.',
    messageMany: n => `ما زال هيرميس يعمل على ${n} محادثات.`,
    warn: 'سيؤدي الخروج إلى إيقاف الوكيل أثناء دورته الحالية. ستفقد أي أعمال لم يكتبها بعد.',
    moreOne: '• المزيد 1',
    moreMany: n => `• المزيد ${n}`,
    buttons: ['استمرار التشغيل', 'الخروج على أي حال']
  }
}

function quitGuardLocaleFor(locale: string): QuitGuardLocale {
  return (locale === 'zh' || locale === 'zh-hant' || locale === 'ja' || locale === 'ar') ? (locale as QuitGuardLocale) : 'en'
}

export interface QuitPrompt {
  message: string
  detail: string
  buttons: [string, string]
}

/**
 * The confirmation to show, or null when quitting should just proceed.
 *
 * `quittingForHandoff` covers the update / swap / uninstall relaunches: those
 * are the app replacing itself, not the user walking away, and a modal there
 * would strand the detached script waiting on a PID that never exits.
 *
 * `locale` is the renderer's chosen display language (defaults to English).
 */
export function quitPromptFor(work: ActiveWork, quittingForHandoff: boolean, locale = 'en'): null | QuitPrompt {
  if (quittingForHandoff || work.count < 1) {
    return null
  }

  const strings = QUIT_GUARD_STRINGS[quitGuardLocaleFor(locale)]
  const listed = work.titles.slice(0, MAX_LISTED)
  const remaining = work.count - listed.length
  const lines = listed.map(title => `• ${title}`)

  if (remaining > 0) {
    lines.push(remaining === 1 ? strings.moreOne : strings.moreMany(remaining))
  }

  return {
    buttons: strings.buttons,
    detail: [
      lines.join('\n'),
      lines.length > 0 ? '' : null,
      strings.warn
    ]
      .filter(line => line !== null)
      .join('\n')
      .trim(),
    message: work.count === 1 ? strings.messageOne : strings.messageMany(work.count)
  }
}
