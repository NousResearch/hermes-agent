// The composer's copy; `Translations` references this type.
export interface ComposerTranslations {
  quoteAddToChat: string
  message: string
  wakingProfile: (profile: string) => string
  placeholderStarting: string
  placeholderReconnecting: string
  placeholderFollowUp: string
  newSessionPlaceholders: readonly string[]
  followUpPlaceholders: readonly string[]
  startVoice: string
  openDirective: string
  queueMessage: string
  steer: string
  redirect: string
  stop: string
  send: string
  speaking: string
  transcribing: string
  thinking: string
  muted: string
  listening: string
  muteMic: string
  unmuteMic: string
  stopListening: string
  stopShort: string
  endConversation: string
  endShort: string
  stopDictation: string
  transcribingDictation: string
  voiceControls: string
  voiceEngine: string
  voiceEngineChained: string
  voiceEngineLive: string
  voiceEngineLiveNeedsKey: string
  voiceEngineChangeFailed: string
  voiceEngineChainedShort: string
  voiceEngineLiveShort: string
  voiceDictation: string
  speakReplies: string
  stopSpeakingReplies: string
  wakeWord: (phrase: string) => string
  wakeWordListening: (phrase: string) => string
  wakeWordOff: (phrase: string) => string
  wakeWordPausedVoice: (phrase: string) => string
  lookupLoading: string
  lookupNoMatches: string
  lookupTry: string
  lookupOr: string
  commonCommands: string
  hotkeys: string
  helpFooter: string
  commandDescs: Record<string, string>
  hotkeyDescs: Record<string, string>
  attachUrlTitle: string
  attachUrlDesc: string
  urlPlaceholder: string
  urlHintPre: string
  attach: string
  queued: (count: number) => string
  queuedPaused: (count: number) => string
  attachmentOnly: string
  emptyTurn: string
  hiddenQueued: string
  attachments: (count: number) => string
  editingInComposer: string
  editingQueuedInComposer: string
  restoredDraftNotice: string
  restoredDraftUndo: string
  /** The local-setup offer above the input after the first finished task. */
  queueEdit: string
  queueExpand: string
  queueCollapse: string
  queueSendNext: string
  queueSend: string
  queueSteer: string
  queueDelete: string
  queueLostNote: string
  restoreImageDraft: string
  queueCancelFailed: string
  queueLostDiscard: string
  queueLostDiscardTip: string
  queueResume: string
  queueResumeTip: string
  queueStuckTitle: string
  queueStuckBody: string
  queueDroppedTitle: string
  queueDroppedBody: string
  terminalSelectionMissingTitle: string
  terminalSelectionMissingBody: string
  queuedTerminalSelectionExpiredBody: string
  previewUnavailable: string
  previewLabel: (label: string) => string
  couldNotPreview: (label: string) => string
  removeAttachment: (label: string) => string
  dictating: string
  preparingAudio: string
  speakingResponse: string
  readingAloud: string
  themeSuggestions: string
  noMatchingThemes: string
  themeTryPre: string
  themeTryPost: string
  attachLabel: string
  files: string
  folder: string
  images: string
  pasteImage: string
  url: string
  promptSnippets: string
  tipPre: string
  tipPost: string
  snippetsTitle: string
  snippetsDesc: string
  snippets: Record<string, { label: string; description: string; text: string }>
  dropFiles: string
  dropSession: string
  mcpSuggestions: {
    label: (server: string) => string
    tip: (keyword: string) => string
    connecting: (server: string) => string
    cancelTip: string
    added: (server: string) => string
    addedTip: string
    connectFailed: (server: string) => string
  }
  skillSuggestions: {
    label: (skill: string) => string
    tip: (skill: string) => string
    done: (skill: string) => string
    doneTip: string
  }
  githubSuggestions: {
    label: string
    tip: string
    done: string
    doneTip: string
  }
  repairSuggestions: {
    label: (server: string) => string
    tip: (server: string) => string
    working: (server: string) => string
    workingTip: string
    done: (server: string) => string
    doneTip: string
    failed: (server: string) => string
  }
  cronSuggestions: {
    label: string
    tip: (phrase: string) => string
    prefix: string
    done: string
    doneTip: string
  }
}
