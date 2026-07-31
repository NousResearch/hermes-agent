export type SelectionActionLocale = 'en' | 'ar' | 'ja' | 'zh' | 'zh-hant'

interface SelectionActionLabels {
  lookUp: string
  readAloud: string
  translate: string
}

const DEFAULT_SELECTION_ACTION_LOCALE: SelectionActionLocale = 'en'

const SELECTION_ACTION_LABELS = {
  en: {
    lookUp: 'Look Up',
    readAloud: 'Read Aloud',
    translate: 'Translate…'
  },
  ar: {
    lookUp: 'بحث',
    readAloud: 'قراءة بصوت عال',
    translate: 'ترجمة…'
  },
  ja: {
    lookUp: '調べる',
    readAloud: '読み上げ',
    translate: '翻訳…'
  },
  zh: {
    lookUp: '查询',
    readAloud: '朗读',
    translate: '翻译…'
  },
  'zh-hant': {
    lookUp: '查詢',
    readAloud: '朗讀',
    translate: '翻譯…'
  }
} satisfies Record<SelectionActionLocale, SelectionActionLabels>

const CHAT_MESSAGE_SELECTION_SELECTOR =
  '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'

export interface SelectionFrame {
  executeJavaScript: (script: string) => Promise<unknown>
  isDestroyed: () => boolean
}

interface SelectionWindow {
  isDestroyed: () => boolean
  webContents: {
    isDestroyed: () => boolean
    mainFrame: SelectionFrame
  }
}

interface SelectionActionWindow {
  isDestroyed: () => boolean
  webContents: {
    send: (channel: string, text: string) => void
    showDefinitionForSelection: () => void
  }
}

export interface SelectionActionItem {
  click?: () => void
  label: string
}

interface SelectionParams {
  isEditable: boolean
  selectionText: string
}

export interface SelectionProbeResult {
  authorized: boolean
  locale: SelectionActionLocale
}

export interface SelectionAuthorization extends SelectionProbeResult {
  current: boolean
}

export type SelectionProbe = (frame: SelectionFrame, expectedText: string) => Promise<SelectionProbeResult>

function deniedSelectionProbe(): SelectionProbeResult {
  return { authorized: false, locale: DEFAULT_SELECTION_ACTION_LOCALE }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function normalizeSelectionActionLocale(locale: unknown): SelectionActionLocale {
  if (
    typeof locale !== 'string' ||
    !Object.prototype.hasOwnProperty.call(SELECTION_ACTION_LABELS, locale)
  ) {
    return DEFAULT_SELECTION_ACTION_LOCALE
  }

  return locale as SelectionActionLocale
}

export async function probeChatMessageSelection(
  frame: SelectionFrame,
  expectedText: string
): Promise<SelectionProbeResult> {
  if (!frame || frame.isDestroyed() || !expectedText) {
    return deniedSelectionProbe()
  }

  try {
    const result = await frame.executeJavaScript(`
      (() => {
        const expectedText = ${JSON.stringify(expectedText)}
        const locale = globalThis.document?.documentElement?.lang
        const selection = globalThis.getSelection?.()

        if (
          !selection ||
          selection.rangeCount === 0 ||
          selection.isCollapsed ||
          selection.toString() !== expectedText
        ) {
          return { authorized: false, locale }
        }

        const messageRootFor = node => {
          const element =
            node?.nodeType === globalThis.Node?.ELEMENT_NODE ? node : (node?.parentElement ?? null)

          return element?.closest(${JSON.stringify(CHAT_MESSAGE_SELECTION_SELECTOR)}) ?? null
        }
        const anchorRoot = messageRootFor(selection.anchorNode)
        const focusRoot = messageRootFor(selection.focusNode)

        return { authorized: Boolean(anchorRoot && anchorRoot === focusRoot), locale }
      })()
    `)

    if (!isRecord(result) || result.authorized !== true) {
      return deniedSelectionProbe()
    }

    return {
      authorized: true,
      locale: normalizeSelectionActionLocale(result.locale)
    }
  } catch {
    return deniedSelectionProbe()
  }
}

export function createChatSelectionAuthorizer(
  window: SelectionWindow,
  probe: SelectionProbe = probeChatMessageSelection
) {
  let generation = 0

  return async (
    frame: SelectionFrame | null | undefined,
    expectedText: string,
    enabled = true
  ): Promise<SelectionAuthorization> => {
    const ownGeneration = ++generation

    if (
      !enabled ||
      !frame ||
      frame.isDestroyed() ||
      window.isDestroyed() ||
      window.webContents.isDestroyed() ||
      frame !== window.webContents.mainFrame
    ) {
      return {
        ...deniedSelectionProbe(),
        current: ownGeneration === generation
      }
    }

    let probeResult: unknown

    try {
      probeResult = await probe(frame, expectedText)
    } catch {
      probeResult = deniedSelectionProbe()
    }

    const normalizedProbe =
      isRecord(probeResult) && probeResult.authorized === true
        ? {
            authorized: true,
            locale: normalizeSelectionActionLocale(probeResult.locale)
          }
        : deniedSelectionProbe()

    const current =
      ownGeneration === generation &&
      !window.isDestroyed() &&
      !window.webContents.isDestroyed() &&
      !frame.isDestroyed() &&
      frame === window.webContents.mainFrame

    const authorized = current && normalizedProbe.authorized

    return {
      authorized,
      current,
      locale: authorized ? normalizedProbe.locale : DEFAULT_SELECTION_ACTION_LOCALE
    }
  }
}

export function shouldOfferSelectionActions(params: SelectionParams, authorized: boolean): boolean {
  return Boolean(params.selectionText?.trim()) && !params.isEditable && authorized
}

export function buildSelectionActionItems(
  window: SelectionActionWindow,
  selectionText: string,
  isMac: boolean,
  locale: unknown = DEFAULT_SELECTION_ACTION_LOCALE
): SelectionActionItem[] {
  const labels = SELECTION_ACTION_LABELS[normalizeSelectionActionLocale(locale)]

  const items: SelectionActionItem[] = [
    {
      label: labels.readAloud,
      click: () => {
        if (!window.isDestroyed()) {
          window.webContents.send('hermes:selection-speech:read', selectionText)
        }
      }
    }
  ]

  if (isMac) {
    items.push({
      label: labels.lookUp,
      click: () => {
        if (!window.isDestroyed()) {
          window.webContents.showDefinitionForSelection()
        }
      }
    })
  }

  items.push({
    label: labels.translate,
    click: () => {
      if (!window.isDestroyed()) {
        window.webContents.send('hermes:selection-translate:open', selectionText)
      }
    }
  })

  return items
}
