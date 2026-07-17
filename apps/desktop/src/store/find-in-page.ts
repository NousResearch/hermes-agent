import { atom } from 'nanostores'

export interface FindInPageState {
  active: boolean
  query: string
  matchOrdinal: number
  matchCount: number
}

export const $findInPage = atom<FindInPageState>({
  active: false,
  query: '',
  matchOrdinal: 0,
  matchCount: 0
})

export function openFindBar(): void {
  $findInPage.set({ active: true, query: '', matchOrdinal: 0, matchCount: 0 })
}

export function closeFindBar(): void {
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
  void window.hermesDesktop?.stopFindInPage()
}

export function setFindQuery(query: string): void {
  const prev = $findInPage.get()
  $findInPage.set({ ...prev, query })

  if (!query) {
    void window.hermesDesktop?.stopFindInPage()
    $findInPage.set({ ...prev, query: '', matchOrdinal: 0, matchCount: 0 })

    return
  }

  void window.hermesDesktop?.findInPage(query, { forward: true, findNext: false })
}

export function findNext(): void {
  const { query } = $findInPage.get()

  if (query) {
    void window.hermesDesktop?.findInPage(query, { forward: true, findNext: true })
  }
}

export function findPrevious(): void {
  const { query } = $findInPage.get()

  if (query) {
    void window.hermesDesktop?.findInPage(query, { forward: false, findNext: true })
  }
}

/** Called by the preload bridge when `found-in-page` fires on webContents. */
export function updateFindResults(activeMatch: number, count: number): void {
  const prev = $findInPage.get()
  $findInPage.set({ ...prev, matchOrdinal: activeMatch, matchCount: count })
}

export function initFindInPageListener(): (() => void) | undefined {
  return window.hermesDesktop?.onFoundInPage?.((result: { activeMatchOrdinal: number; count: number }) => {
    updateFindResults(result.activeMatchOrdinal, result.count)
  })
}
