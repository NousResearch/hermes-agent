import type { ErrorCodeKey } from '@/lib/error-surface'

/** One error-card entry: a short title and one plain sentence. Either may
 *  take the failing provider's display name (falls back to "the AI service"). */
export interface ErrorCardCopy {
  title: string | ((provider: string) => string)
  body: string | ((provider: string) => string)
}

export type ToolTitleKey =
  | 'browser_click'
  | 'browser_fill'
  | 'browser_navigate'
  | 'browser_snapshot'
  | 'browser_take_screenshot'
  | 'browser_type'
  | 'clarify'
  | 'cronjob'
  | 'edit_file'
  | 'execute_code'
  | 'image_generate'
  | 'list_files'
  | 'memory'
  | 'patch'
  | 'read_file'
  | 'search_files'
  | 'session_search_recall'
  | 'terminal'
  | 'todo'
  | 'vision_analyze'
  | 'web_extract'
  | 'web_search'
  | 'write_file'

interface ToolTitleCopy {
  done: string
  pending: string
  pendingAction: string
}

export interface AssistantTranslations {
  assistant: {
    thread: {
      loadingSession: string
      showEarlier: string
      loadingResponse: string
      loadingLocalModel: (model: string) => string
      processingPrompt: string
      resumeWhenBackgroundDone: (count: number) => string
      thinking: string
      thought: string
      thoughtBriefly: string
      thoughtFor: (duration: string) => string
      turnDuration: (duration: string) => string
      today: (time: string) => string
      yesterday: (time: string) => string
      copy: string
      refresh: string
      moreActions: string
      branchNewChat: string
      react: string
      dismissError: string
      /** Layer titles for the structured error card (agent/error_surface.py).
       *  `generic` is the fallback when the backend sent no descriptor. */
      errorLayers: {
        auth: string
        billing: string
        disk: string
        endpoint: string
        gateway: string
        generic: string
        provider: string
        runtime: string
        streaming: string
      }
      /** One plain sentence per layer — what happened and what to do — shown
       *  when the failure code has no dedicated entry in `errorCodes`. */
      errorLayerBodies: {
        auth: string
        billing: string
        disk: string
        endpoint: string
        gateway: string
        generic: string
        provider: string
        runtime: string
        streaming: string
      }
      /** Per failure code (agent/error_classifier.py FailoverReason values plus
       *  the gateway's site codes): a title and one plain sentence saying what
       *  happened and what to do. Function entries take the provider label. */
      errorCodes: Record<ErrorCodeKey, ErrorCardCopy>
      /** Auth layer, keyed on how the provider is credentialed. The OAuth
       *  body is `errorOauthExpired` (already translated per locale). */
      errorAuthKinds: { api_key: ErrorCardCopy; oauth: Pick<ErrorCardCopy, 'title'> }
      /** Collapsed "Details" line holding the raw provider/gateway text. */
      errorDetails: string
      /** Stands in for the provider name when the descriptor carries none. */
      errorGenericProvider: string
      /** Global toast title for a mid-turn gateway `error` event. */
      errorToastTitle: string
      errorRetry: string
      errorLimitResets: (time: string) => string
      /** Escape hatch when Retry would only reproduce SESSION_NOT_OWNED (#106217). */
      errorStartNewSession: string
      errorSwitchProvider: string
      errorChooseModel: string
      errorCompressConversation: string
      errorCompressFailed: string
      errorOpenHermesFolder: string
      errorOpenHermesFolderFailed: string
      errorUpdateApiKey: string
      /** One-click recovery for an expired/revoked OAuth grant: re-runs that
       *  provider's sign-in flow (auth layer, authKind 'oauth'). */
      errorSignInAgain: (provider: string) => string
      /** Free-tier refusals: opens the free sign-in dialog (signing in is free and lifts the refusal). */
      errorSignInFreeTier: string
      /** Explains WHY the turn failed for an OAuth 401 — the raw body
       *  ("HTTP 401: User not found.") doesn't say "sign in again". */
      errorOauthExpired: (provider: string) => string
      errorOpenLogs: string
      errorOpenLogsFailed: string
      errorOpenDesktopLogs: string
      errorCopyDiagnostics: string
      errorSendDiagnostics: string
      filesChanged: (count: number) => string
      reviewChanges: string
      readAloudFailed: string
      preparingAudio: string
      stopReading: string
      readAloud: string
      editMessage: string
      expandMessage: string
      scrollToBottom: string
      stop: string
      restorePrevious: string
      restoreCheckpoint: string
      restoreFromHere: string
      restoreTitle: string
      restoreBody: string
      restoreConfirm: string
      restoreNext: string
      goForward: string
      sendEdited: string
      attachingFile: string
    }
    approval: {
      gatewayDisconnected: string
      sendFailed: string
      reconnect: string
      timedOutSystemLine: string
      openSafetySettings: string
      run: string
      command: string
      moreOptions: string
      allowSession: string
      alwaysAllowMenu: string
      jumpToApproval: string
      reject: string
      alwaysTitle: string
      alwaysDescription: (pattern: string) => string
      alwaysAllow: string
    }
    clarify: {
      notReady: string
      gatewayDisconnected: string
      sendFailed: string
      loadingQuestion: string
      other: string
      placeholder: string
      skip: string
      skipped: string
      continueLabel: string
      confirmAndContinueLabel: string
      answeredBadge: string
      questionProgress: (answered: number, total: number) => string
      lateAnswer: (question: string, choice: string) => string
      lateAnswerTip: string
      lateAnswerHint: string
    }
    mcpSetup: {
      installTitle: string
      enableTitle: string
      authorizeTitle: string
      installAction: string
      enableAction: string
      authorizeAction: string
      installed: (server: string) => string
      enabled: (server: string) => string
      authorized: (server: string) => string
      failed: (server: string) => string
      toolCount: (count: number) => string
      notInCatalog: (server: string) => string
      envRequired: string
      sendFailed: string
      reloadFailed: string
      gatewayDisconnected: string
    }
    tool: {
      copyCode: string
      renderingImage: string
      copyOutput: string
      copyCommand: string
      copyContent: string
      copyUrl: string
      copyResults: string
      copyQuery: string
      copyFile: string
      copyPath: string
      failedCalls: (count: number) => string
      skillActivity: {
        loading: string
        loaded: string
        loadFailed: string
        readingResource: string
        readResource: string
        resourceFailed: string
        listing: string
        listed: string
        listFailed: string
        unavailable: string
      }
      outputAlt: string
      rawResponse: string
      copyActivity: string
      recoveredOne: string
      recoveredMany: (count: number) => string
      failedOne: string
      failedMany: (count: number) => string
      statusRunning: string
      statusError: string
      statusRecovered: string
      statusDone: string
      /** Over-budget / rejected memory write title — not "Saved to memory". */
      resultUnavailable: string
      memoryWriteNoted: string
      actions: {
        read: string
        reading: string
        opened: string
        opening: string
        failedToOpen: string
        searched: string
        searching: string
        ran: string
        running: string
        ranCode: string
        runningCode: string
      }
      prefixes: {
        browser: string
        web: string
      }
      titleTemplates: {
        actionCommand: (action: string, command: string) => string
        actionQuoted: (action: string, value: string) => string
        actionTarget: (action: string, target: string) => string
        prefixedDone: (prefix: string, action: string) => string
        runningPrefixedTool: (prefix: string, action: string) => string
        runningTool: (action: string) => string
      }
      titles: Record<ToolTitleKey, ToolTitleCopy>
    }
  }
}
