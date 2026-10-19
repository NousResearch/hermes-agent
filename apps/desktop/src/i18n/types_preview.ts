export interface PreviewTranslations {
  tab: string
  pin: string
  unpin: string
  closePane: string
  loading: string
  unavailable: string
  missingTarget: string
  missingTitle: string
  missingBody: (label: string) => string
  opening: string
  hide: string
  openPreview: string
  openInBrowser: string
  openInExternal: string
  popIn: string
  popOut: string
  linkHint: string
  sourceLineTitle: string
  source: string
  renderedPreview: string
  diff: string
  unknownSize: string
  binaryTitle: string
  binaryBody: (label: string) => string
  largeTitle: string
  largeBody: (label: string, size: string) => string
  previewAnyway: string
  truncated: string
  table: {
    summary: (rows: number, columns: number) => string
    clipped: (rows: number, columns: number) => string
    empty: string
    unterminatedQuote: string
    column: (index: number) => string
    regionLabel: (label: string) => string
  }
  noInlineTitle: string
  noInlineBody: (mimeType: string) => string
  edit: string
  editing: string
  unsavedChanges: string
  saveFailed: (message: string) => string
  saveScopeChanged: string
  diskChangedTitle: string
  diskChangedBody: string
  overwrite: string
  discardReload: string
  console: {
    deselect: string
    select: string
    copyFailed: string
    copyEntry: string
    sendEntry: string
    messages: (count: number) => string
    resize: string
    title: string
    selected: (count: number) => string
    sendToChat: string
    copySelected: string
    copyAll: string
    copy: string
    clear: string
    empty: string
    promptHeader: string
    sentTitle: string
    sentMessage: (count: number) => string
  }
  web: {
    appFailedToBoot: string
    serverNotFound: string
    remoteLoopback: string
    failedToLoad: string
    tryAgain: string
    restarting: string
    askRestart: string
    lookingRestart: (taskId: string) => string
    restartingTitle: string
    restartingMessage: string
    startRestartFailed: (message: string) => string
    restartFailed: string
    hideConsole: string
    showConsole: string
    hideDevTools: string
    openDevTools: string
    goBack: string
    goForward: string
    reload: string
    address: string
    addressPlaceholder: string
    blankPageBody: string
    finishedRestarting: (message?: string) => string
    failedRestarting: (message: string) => string
    unknownError: string
    restartedTitle: string
    reloadingNow: string
    restartFailedTitle: string
    restartFailedMessage: string
    stillWorking: string
    workspaceReloading: string
    fileChanged: (url: string) => string
    filesChanged: (count: number, url: string) => string
    watchFailed: (message: string) => string
    moduleMimeDescription: string
    loadFailedConsole: (code: number | undefined, message: string) => string
    unreachableDescription: string
    openTarget: (url: string) => string
    fallbackTitle: string
    annotate: string
    annotateOn: string
    annotateNeedPage: string
    annotateFailed: string
    commenting: string
    addComments: (count: number) => string
    commentPlaceholder: string
    commentTitle: (n: number) => string
    saveComment: string
    cancelComment: string
  }
}
