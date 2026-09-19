import type { Translations } from './types'

export const enAssistant = {
  assistant: {
    thread: {
      loadingSession: 'Loading session',
      showEarlier: 'Show earlier messages',
      loadingResponse: 'Hermes is loading a response',
      loadingLocalModel: model => `Loading ${model} into memory`,
      processingPrompt: 'Processing prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Will resume when the background task finishes'
          : `Will resume when ${count} background tasks finish`,
      thinking: 'Thinking',
      thought: 'Thought',
      thoughtBriefly: 'Thought briefly',
      thoughtFor: duration => `Thought for ${duration}`,
      turnDuration: duration => `This turn took ${duration}`,
      today: time => `Today, ${time}`,
      yesterday: time => `Yesterday, ${time}`,
      copy: 'Copy',
      refresh: 'Refresh',
      moreActions: 'More actions',
      branchNewChat: 'Branch in new chat',
      react: 'React',
      dismissError: 'Dismiss error',
      errorLayers: {
        auth: 'Sign-in problem',
        billing: 'Out of credits',
        disk: 'Disk full',
        endpoint: "Can't reach your model server",
        gateway: 'Hermes hit a problem',
        generic: "Hermes couldn't finish this reply",
        provider: 'The AI service returned an error',
        runtime: 'Hermes hit a problem',
        streaming: 'The reply was cut off'
      },
      errorLayerBodies: {
        auth: 'The AI service rejected your sign-in. Check the credentials for this provider, then send your message again.',
        billing: 'Your account has no credits left for this provider. Top up or switch provider, then send again.',
        disk: 'Your disk is full, so Hermes could not save this conversation. Free some space, then retry.',
        endpoint: "Hermes can't reach your custom model server. Check that it is running, then send your message again.",
        gateway: 'Hermes hit an internal problem starting this reply. Send your message again; if it keeps happening, send diagnostics.',
        generic: 'Something went wrong while Hermes was replying. Retry, or copy the details if it keeps happening.',
        provider: 'The AI service could not complete this request. Retry in a moment or switch provider.',
        runtime: 'Hermes hit an internal problem starting this reply. Send your message again; if it keeps happening, send diagnostics.',
        streaming: 'The connection dropped before the reply finished. Retry to send it again.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider} rejected your sign-in`,
          body: provider => `The credentials saved for ${provider} were not accepted. Fix them in Settings or switch provider, then send your message again.`
        },
        auth_permanent: {
          title: provider => `${provider} rejected your sign-in`,
          body: provider =>
            `The credentials saved for ${provider} are invalid or were revoked. Update them or switch provider, then send your message again.`
        },
        billing: {
          title: 'Out of credits',
          body: provider => `Your ${provider} account has no credits left. Top up or switch provider, then send again.`
        },
        rate_limit: {
          title: 'The AI service is busy',
          body: provider => `${provider} is limiting requests right now. Wait a minute, then retry.`
        },
        upstream_rate_limit: {
          title: 'The AI service is busy',
          body: provider => `${provider} is limiting requests right now. Wait a minute, then retry.`
        },
        overloaded: {
          title: 'The AI service is overloaded',
          body: provider => `${provider} is having problems right now. Retry in a moment or switch provider.`
        },
        server_error: {
          title: 'The AI service had a problem',
          body: provider => `${provider} returned a server error. Retry in a moment or switch provider.`
        },
        timeout: {
          title: 'The reply timed out',
          body: provider => `${provider} did not answer in time. Retry to send it again.`
        },
        stream_drop: {
          title: 'The reply was cut off',
          body: 'The connection dropped before the reply finished. Retry to send it again.'
        },
        upstream_blocked: {
          title: 'A firewall blocked the request',
          body: provider =>
            `A firewall or CDN in front of ${provider} blocked the request before it reached the model — your key is probably fine. Set a User-Agent header via the provider's extra_headers in Settings, or switch provider, then send your message again.`
        },
        ssl_cert_verification: {
          title: 'Secure connection failed',
          body: provider =>
            `Hermes could not verify the secure connection to ${provider}. Check your network or proxy settings, or switch provider, then send your message again.`
        },
        context_overflow: {
          title: 'This conversation is too long',
          body: 'The conversation no longer fits the model. Compress it or start a new chat, then send again.'
        },
        payload_too_large: {
          title: 'This message is too large',
          body: 'The request was too big for the model. Compress the conversation or start a new chat, then send again.'
        },
        model_not_found: {
          title: 'This model is not available',
          body: provider => `${provider} does not offer this model on your account. Choose another model, then send your message again.`
        },
        provider_policy_blocked: {
          title: 'This model is blocked by your account settings',
          body: provider =>
            `${provider} would not route this request under your account's data or privacy settings. Choose another model or switch provider.`
        },
        content_policy_blocked: {
          title: 'The AI service declined this request',
          body: provider => `${provider} would not answer this message. Edit it and send again.`
        },
        format_error: {
          title: 'The AI service rejected the request',
          body: provider =>
            `${provider} did not accept how this request was built. Switch provider or send diagnostics so we can look into it.`
        },
        truncated: {
          title: 'The reply was cut short',
          body: 'The model stopped before finishing. Retry to get a complete reply.'
        },
        invalid_response: {
          title: 'The AI service sent an unreadable reply',
          body: provider => `${provider} returned something Hermes could not read. Retry in a moment.`
        },
        empty_response: {
          title: 'The AI service sent an empty reply',
          body: provider => `${provider} returned nothing for this message. Retry in a moment.`
        },
        loop_error: {
          title: 'Hermes got stuck in a loop',
          body: 'The reply kept repeating the same steps, so Hermes stopped it. Retry, or start a new chat if it happens again.'
        },
        SESSION_NOT_OWNED: {
          title: 'This chat is open somewhere else',
          body: 'This chat is currently open in another Hermes window or terminal. Close it there and send your message again, or start a new chat here.'
        },
        disk_full: {
          title: 'Disk full',
          body: 'Your disk is full, so Hermes could not save this conversation. Free some space, then retry.'
        },
        // Nous free tier. The body is normally the backend's own sentence (it names the wait
        // and the way forward); these bodies stand in for an older backend that sent none.
        free_tier_disabled: {
          title: 'Using Hermes without signing in is switched off right now',
          body: "Sign in with a Nous account to keep chatting, it's free."
        },
        free_tier_rate_limited: {
          title: "You've used up the allowance for chatting without signing in",
          body: "It refreshes shortly. Sign in with a Nous account for a bigger allowance, it's free."
        },
        free_tier_at_capacity: {
          title: 'Chatting without signing in is really busy right now',
          body: "Sign in to skip the queue, it's free, or try again in a little while."
        },
        free_tier_model_not_free: {
          title: "That model isn't available without signing in",
          body: "Hermes uses the free model for now. Sign in with a Nous account for more models, it's free."
        },
        free_tier_route: {
          title: "Hermes couldn't reach the free model on this route",
          body: "Sign in with a Nous account, it's free, or check the NOUS_INFERENCE_BASE_URL setting."
        },
        free_tier_outage: {
          title: 'The free model is having trouble responding right now',
          body: 'Try sending your message again in a minute.'
        },
        free_tier_refused: {
          title: "Hermes couldn't send that without signing in",
          body: 'Signing in with a Nous account is free.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} rejected your API key`,
          body: provider => `The key saved for ${provider} is invalid or was revoked. Update it, then retry.`
        },
        oauth: {
          title: provider => `Your ${provider} sign-in expired`
        }
      },
      errorDetails: 'Details',
      errorGenericProvider: 'The AI service',
      errorToastTitle: "Hermes couldn't finish the reply",
      errorRetry: 'Retry',
      errorLimitResets: time => `Limit resets at ${time}`,
      errorStartNewSession: 'Start new session',
      errorSwitchProvider: 'Switch provider',
      errorChooseModel: 'Choose a model',
      errorCompressConversation: 'Compress conversation',
      errorCompressFailed: 'Could not compress the conversation',
      errorOpenHermesFolder: 'Open Hermes folder',
      errorOpenHermesFolderFailed: 'Could not open the Hermes folder',
      errorUpdateApiKey: 'Update API key',
      errorSignInAgain: provider => `Sign in to ${provider} again`,
      errorSignInFreeTier: 'Sign in with a Nous account',
      errorOauthExpired: provider =>
        `Your ${provider} sign-in has expired or was revoked. Sign in again to keep chatting.`,
      errorOpenLogs: 'Open logs',
      errorOpenLogsFailed: 'Could not open the logs folder',
      errorOpenDesktopLogs: 'Open Desktop logs',
      errorCopyDiagnostics: 'Copy error details',
      errorSendDiagnostics: 'Send diagnostics',
      filesChanged: count => (count === 1 ? '1 file changed' : `${count} files changed`),
      reviewChanges: 'Review',
      readAloudFailed: 'Read aloud failed',
      preparingAudio: 'Preparing audio...',
      stopReading: 'Stop reading',
      readAloud: 'Read aloud',
      editMessage: 'Edit message',
      expandMessage: 'Expand message',
      scrollToBottom: 'Scroll to bottom',
      stop: 'Stop',
      restorePrevious: 'Restore previous checkpoint',
      restoreCheckpoint: 'Restore checkpoint',
      restoreFromHere: 'Restore checkpoint — rerun from this prompt',
      restoreTitle: 'Restore to this checkpoint?',
      restoreBody:
        'Everything after this prompt is removed from the conversation, and the prompt runs again from here.',
      restoreConfirm: 'Restore & rerun',
      restoreNext: 'Restore next checkpoint',
      goForward: 'Go forward',
      sendEdited: 'Send edited message',
      attachingFile: 'Attaching…'
    },
    approval: {
      gatewayDisconnected:
        'Hermes is offline right now. The command is still waiting for your answer (until the approval timeout). Reconnect, then send it again.',
      sendFailed: 'Could not send your answer',
      reconnect: 'Reconnect',
      timedOutSystemLine:
        'Approval timed out — the command was not run. Ask Hermes to try again, or raise the limit in Settings → Safety → Approval timeout.',
      openSafetySettings: 'Open Safety settings',
      run: 'Run',
      command: 'Command',
      moreOptions: 'More approval options',
      allowSession: 'Allow this session',
      alwaysAllowMenu: 'Always allow…',
      jumpToApproval: 'Approval needed',
      reject: 'Reject',
      alwaysTitle: 'Always allow this command?',
      alwaysDescription: pattern =>
        `This adds the “${pattern}” pattern to your permanent allowlist (~/.hermes/config.yaml). Hermes won’t ask again for commands like this — in this session or any future one.`,
      alwaysAllow: 'Always allow'
    },
    clarify: {
      notReady: 'Clarify request is not ready yet',
      gatewayDisconnected: 'Hermes is offline right now. Reconnect, then send it again.',
      sendFailed: 'Could not send clarify response',
      loadingQuestion: 'Loading question…',
      other: 'Other (type your answer)',
      placeholder: 'Type your answer…',
      skip: 'Skip',
      skipped: 'Skipped',
      continueLabel: 'Continue',
      confirmAndContinueLabel: 'Confirm and continue',
      answeredBadge: 'Answered',
      questionProgress: (answered, total) => `${answered} of ${total} answered`,
      lateAnswer: (question, choice) => `Re: "${question}" — my answer: ${choice}`,
      lateAnswerTip: 'Draft this answer as a follow-up message',
      lateAnswerHint: 'This prompt is no longer waiting. Pick an option to draft it as a follow-up message.'
    },
    mcpSetup: {
      installTitle: 'Add MCP servers',
      enableTitle: 'Enable MCP servers',
      authorizeTitle: 'Authorize MCP servers',
      installAction: 'Install',
      enableAction: 'Enable',
      authorizeAction: 'Authorize',
      installed: server => `Installed ${server}`,
      enabled: server => `Enabled ${server}`,
      authorized: server => `Authorized ${server}`,
      failed: server => `Setup failed for ${server}`,
      toolCount: count => (count === 1 ? '1 tool' : `${count} tools`),
      notInCatalog: server => `“${server}” is not in the MCP catalog`,
      envRequired: 'Fill in the required credentials first',
      sendFailed: 'Could not send MCP setup response',
      reloadFailed: 'Server saved, but reloading MCP tools failed — they load next session',
      gatewayDisconnected: 'Hermes is offline right now. Reconnect, then send it again.'
    },
    tool: {
      copyCode: 'Copy code',
      renderingImage: 'Rendering image',
      copyOutput: 'Copy output',
      copyCommand: 'Copy command',
      copyContent: 'Copy content',
      copyUrl: 'Copy URL',
      copyResults: 'Copy results',
      copyQuery: 'Copy query',
      copyFile: 'Copy file',
      copyPath: 'Copy path',
      failedCalls: (count: number) => `${count} tool call${count === 1 ? '' : 's'} failed`,
      skillActivity: {
        loading: 'Loading skill',
        loaded: 'Loaded skill',
        loadFailed: 'Failed to load skill',
        readingResource: 'Reading skill resource',
        readResource: 'Read skill resource',
        resourceFailed: 'Failed to read skill resource',
        listing: 'Listing skills',
        listed: 'Listed skills',
        listFailed: 'Failed to list skills',
        unavailable: 'Skill result unavailable'
      },
      outputAlt: 'Tool output',
      rawResponse: 'Raw response',
      copyActivity: 'Copy activity',
      recoveredOne: 'Recovered after 1 failed step',
      recoveredMany: count => `Recovered after ${count} failed steps`,
      failedOne: '1 step failed',
      failedMany: count => `${count} steps failed`,
      statusRunning: 'Running',
      statusError: 'Error',
      statusRecovered: 'Recovered',
      statusDone: 'Done',
      resultUnavailable: 'Result unavailable',
      memoryWriteNoted: 'Memory write noted',
      actions: {
        read: 'Read',
        reading: 'Reading',
        opened: 'Opened',
        opening: 'Opening',
        failedToOpen: 'Failed to open',
        searched: 'Searched',
        searching: 'Searching',
        ran: 'Ran',
        running: 'Running',
        ranCode: 'Ran code',
        runningCode: 'Scripting'
      },
      prefixes: {
        browser: 'Browser',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `Running ${prefix.toLowerCase()} ${action.toLowerCase()}`,
        runningTool: action => `Running ${action.toLowerCase()}`
      },
      titles: {
        browser_click: { done: 'Clicked page element', pending: 'Clicking page element', pendingAction: 'Clicking' },
        browser_fill: { done: 'Filled form field', pending: 'Filling form field', pendingAction: 'Filling' },
        browser_navigate: { done: 'Opened page', pending: 'Opening page', pendingAction: 'Opening' },
        browser_snapshot: {
          done: 'Captured page snapshot',
          pending: 'Capturing page snapshot',
          pendingAction: 'Capturing'
        },
        browser_take_screenshot: {
          done: 'Captured screenshot',
          pending: 'Capturing screenshot',
          pendingAction: 'Capturing'
        },
        browser_type: { done: 'Typed on page', pending: 'Typing on page', pendingAction: 'Typing' },
        clarify: { done: 'Asked a question', pending: 'Asking a question', pendingAction: 'Asking' },
        cronjob: { done: 'Cron job', pending: 'Scheduling cron job', pendingAction: 'Scheduling' },
        edit_file: { done: 'Edited file', pending: 'Editing file', pendingAction: 'Editing' },
        execute_code: { done: 'Ran code', pending: 'Scripting', pendingAction: 'Scripting' },
        image_generate: { done: 'Generated image', pending: 'Generating image', pendingAction: 'Generating' },
        list_files: { done: 'Listed files', pending: 'Listing files', pendingAction: 'Listing' },
        memory: { done: 'Saved to memory', pending: 'Saving to memory', pendingAction: 'Saving' },
        patch: { done: 'Patched file', pending: 'Patching file', pendingAction: 'Patching' },
        read_file: { done: 'Read file', pending: 'Reading file', pendingAction: 'Reading' },
        search_files: { done: 'Searched files', pending: 'Searching files', pendingAction: 'Searching' },
        session_search_recall: {
          done: 'Searched session history',
          pending: 'Searching session history',
          pendingAction: 'Searching'
        },
        terminal: { done: 'Ran command', pending: 'Running command', pendingAction: 'Running' },
        todo: { done: 'Updated todos', pending: 'Updating todos', pendingAction: 'Updating' },
        vision_analyze: { done: 'Analyzed image', pending: 'Analyzing image', pendingAction: 'Analyzing' },
        web_extract: { done: 'Read webpage', pending: 'Reading webpage', pendingAction: 'Reading' },
        web_search: { done: 'Searched web', pending: 'Searching web', pendingAction: 'Searching' },
        write_file: { done: 'Edited file', pending: 'Editing file', pendingAction: 'Editing' }
      }
    }
  }
} satisfies Pick<Translations, 'assistant'>
