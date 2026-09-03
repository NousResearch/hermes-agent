// Strings for native (main-process) UI surfaces — dialogs, tray menus, and any
// other OS-level chrome the renderer can't localize itself.
//
// Why a dedicated module instead of reusing the renderer's `TRANSLATIONS`
// catalog: the catalog (`src/i18n/*`) pulls in renderer-only modules (e.g.
// `en.ts` imports `@/app/settings/constants`), so it can't be bundled into the
// main-process entry without dragging unneeded code across the boundary. Every
// native UI surface reads from THIS single table via `appLocale`, so adding a
// new main-process dialog means one more entry here — not a whole new table.

import type { ActiveWork } from './quit-guard'

export type NativeLocale = 'en' | 'zh' | 'zh-hant' | 'ja' | 'ar'

const SUPPORTED: NativeLocale[] = ['en', 'zh', 'zh-hant', 'ja', 'ar']

/** Normalize an arbitrary locale string (e.g. `zh-CN`, `en-US`) to a table key. */
export function normalizeNativeLocale(locale: string | undefined | null): NativeLocale {
  if (!locale) {
    return 'en'
  }

  const base = locale.toLowerCase().split('-')[0]

  return (SUPPORTED as string[]).includes(base) ? (base as NativeLocale) : 'en'
}

// The renderer pushes its display language here whenever it changes.
let appLocale: NativeLocale = 'en'

export function setAppLocale(locale: string | undefined | null): void {
  appLocale = normalizeNativeLocale(locale)
}

export function getAppLocale(): NativeLocale {
  return appLocale
}

interface QuitConfirmCopy {
  messageOne: string
  messageMany: (n: number) => string
  warn: string
  moreOne: string
  moreMany: (n: number) => string
  buttons: [string, string]
}

const QUIT_CONFIRM: Record<NativeLocale, QuitConfirmCopy> = {
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

const MAX_LISTED = 4

export interface QuitConfirmPrompt {
  message: string
  detail: string
  buttons: [string, string]
}

/** Localized quit-confirmation copy, or null when there's nothing in flight. */
export function quitConfirmPrompt(work: ActiveWork, quittingForHandoff: boolean): null | QuitConfirmPrompt {
  if (quittingForHandoff || work.count < 1) {
    return null
  }

  const strings = QUIT_CONFIRM[appLocale]
  const listed = work.titles.slice(0, MAX_LISTED)
  const remaining = work.count - listed.length
  const lines = listed.map(title => `• ${title}`)

  if (remaining > 0) {
    lines.push(remaining === 1 ? strings.moreOne : strings.moreMany(remaining))
  }

  return {
    buttons: strings.buttons,
    detail: [lines.join('\n'), lines.length > 0 ? '' : null, strings.warn]
      .filter(line => line !== null)
      .join('\n')
      .trim(),
    message: work.count === 1 ? strings.messageOne : strings.messageMany(work.count)
  }
}
