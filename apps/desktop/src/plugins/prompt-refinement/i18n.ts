/**
 * Plugin-scoped copy (registered through ctx.i18n, never core en.ts). English
 * ships now; other locales fall back to it until translated.
 */

import type { PluginLocaleBundles } from '@hermes/plugin-sdk'

export const PLUGIN_ID = 'prompt-refinement'

const en = {
  name: 'Prompt Refinement',
  description: 'Turn your prompt into a deeper, expert version of the same task before it is sent.',
  control: {
    pill: {
      off: 'Refine off',
      manual: 'Refine: Approval-First',
      auto: 'Refine: Auto'
    },
    label: (current: string) => `Prompt refinement settings (${current})`,
    tip: 'Prompt refinement: mode, depth and style',
    more: 'More settings…'
  },
  drawer: {
    label: 'Prompt refinement',
    titleSend: 'Review before sending',
    refining: 'Refining your prompt',
    refiningSend: 'Refining your prompt before sending',
    sendOriginal: 'Send original',
    chosenDepth: (depth: string) => `Depth: ${depth}, chosen automatically`,
    attentionIssues: 'Not sent automatically: the refined prompt may have lost or changed something from yours.',
    attentionFailed: 'Not sent: refinement didn’t finish.',
    yourPrompt: 'Your prompt',
    editing: 'Edit the refined prompt',
    unchanged: 'No changes suggested. Your prompt already covers it.',
    attachments: (count: number) =>
      count === 1 ? '1 attachment goes with it unchanged' : `${count} attachments go with it unchanged`,
    added: 'What was added and why',
    analysis: {
      professional: 'Professional standard',
      rawMaterial: 'Raw material',
      methods: 'Methods and tools',
      verification: 'Outside check'
    },
    kept: 'Everything from your prompt is kept',
    approveSend: 'Approve and send',
    edit: 'Edit',
    doneEditing: 'Done editing',
    cancel: 'Cancel',
    retry: 'Try again',
    shortcuts: '⌘↵ approve · Esc cancel',
    blockingTitle: 'Missing from the refined prompt',
    blockingHint: 'Approve stays off until these are back. Fix them with Edit, or use your original.',
    intentTitle: 'Asks for more than your prompt did',
    intentHint: 'Approve only if you want this. Otherwise remove it with Edit, or send your original.',
    reviewTitle: 'Check before approving',
    reviewHint: 'These may only have been rephrased.',
    acknowledge: 'I checked the refined prompt: nothing important was lost or added',
    kinds: {
      'code-block': 'Code block',
      'inline-code': 'Inline code',
      reference: 'Reference',
      url: 'URL',
      path: 'File path',
      identifier: 'Name or identifier',
      number: 'Number',
      name: 'Hyphenated name',
      destructive: 'Deletes or overwrites something',
      'file-change': 'Changes files or code',
      install: 'Installs something',
      external: 'Contacts someone or publishes',
      spending: 'Spends money',
      permission: 'Touches permissions, credentials or security',
      delegation: 'Changes delegation (subagents, swarms)',
      'from-context': 'Carries text in from the chat',
      negation: (n: number) => `${n} fewer negation word${n === 1 ? '' : 's'} (not, never, without…)`,
      requirement: (n: number) => `${n} fewer requirement word${n === 1 ? '' : 's'} (must, only, exactly…)`
    }
  },
  failure: {
    title: 'Couldn’t refine this prompt',
    offline: 'Hermes isn’t connected right now.',
    unavailable:
      'No refinement model is available. Pick one under Settings › Model › Auxiliary, or check the provider sign-in.',
    unsupported: 'This Hermes backend doesn’t support prompt refinement. Update Hermes to use it.',
    'slot-missing':
      'Prompt refinement’s backend plugin isn’t turned on. Run “hermes plugins enable prompt-refinement”, then restart Hermes and try again.',
    timeout: 'The refinement model took too long to answer.',
    empty: 'The refinement model returned nothing.',
    'too-long': 'This prompt is too long to refine. You can still send it as written.',
    canceled: 'Refinement was canceled.',
    failed: 'Something went wrong while refining.',
    malformed: 'The refinement model answered in a form that couldn’t be read.'
  },
  notify: {
    scopeChanged: 'Refinement canceled because you switched chats. Your draft was kept.',
    noHost: 'Couldn’t open the review, so nothing was sent. Your draft was kept.',
    busy: 'Another prompt is already being refined.',
    error: 'Refinement stopped unexpectedly, so nothing was sent. Your draft was kept.'
  },
  settings: {
    title: 'Prompt Refinement',
    modeTitle: 'Mode',
    modes: {
      off: 'Off',
      manual: 'Approval-First',
      auto: 'Automatic'
    },
    modeHints: {
      off: 'Prompts are sent exactly as you type them.',
      manual:
        'Enter refines your prompt and shows it for review. Nothing is sent until you approve it, edit it, or choose Send original. Slash commands, queued and voice messages, and very short prompts are sent as typed.',
      auto: 'Enter refines your prompt and sends it, no approval needed. It stops and shows you the review if refinement fails, would drop something from your prompt, or asks for something your prompt did not (such as deleting, installing, contacting someone, spending or delegating). Slash commands, queued and voice messages, and very short prompts are sent as typed.'
    },
    depthTitle: 'Depth',
    autoDepthTitle: 'Choose depth automatically',
    autoDepthHint:
      'The refiner picks Polish, Clarify or Deepen to fit each prompt, and the review shows which it chose.',
    depths: {
      auto: 'Auto',
      polish: 'Polish',
      clarify: 'Clarify',
      deepen: 'Deepen'
    },
    depthHints: {
      polish: 'Fixes wording, grammar and structure only. Nothing is added, and it stays about the same length.',
      clarify:
        'Spells out what your prompt implies: the goal, requirements, expected output and assumptions to confirm. No new methods or checks; at most about twice as long.',
      deepen:
        'Asks for the task the way a top professional would do it: their standards, primary sources, specific methods and tools, and an outside check where it fits. The longest and slowest.'
    },
    styleTitle: 'Style',
    styles: {
      auto: 'Auto',
      clarity: 'Clear and concise',
      technical: 'Technical',
      research: 'Research',
      creative: 'Creative',
      agentic: 'Agentic task',
      custom: 'Custom'
    },
    styleHints: {
      auto: 'Matches the kind of task you wrote.',
      clarity: 'Plain, direct wording with no extra length.',
      technical: 'Exact versions, files, commands and acceptance checks.',
      research: 'Primary sources, evidence and stated uncertainty.',
      creative: 'Tone, audience and form, with room to explore.',
      agentic: 'Clear steps, tools to use, stopping point and what to report back.',
      custom: 'Follows your own guidance below.'
    },
    customStyleTitle: 'Custom style',
    customStylePlaceholder:
      'For example: write for a senior backend engineer, keep it under 150 words, prefer bullet points.',
    customStyleHint: (max: number) => `Up to ${max} characters. Used only when the style is Custom.`,
    contextTitle: 'Use recent chat context',
    contextHint:
      'Includes a short excerpt of the last few messages in this chat, so “do the same for the other file” can be made specific. Off by default.',
    modelTitle: 'Refinement model',
    modelDedicated: 'Prompt refinement model',
    modelDedicatedHint:
      'Uses the “Prompt refinement” slot in Settings › Model › Auxiliary (enable the prompt-refinement Hermes plugin to choose it). Without it, Hermes picks its automatic side model.',
    modelChat: 'This chat’s model',
    modelChatHint:
      'Uses the model of the chat you’re typing in. A new chat with no session yet uses the refinement model.',
    modelSpeedHint:
      'A small, fast model keeps refinement quick and cheap. Repeating a refinement of the same draft reuses the earlier answer.',
    openModelSettings: 'Open model settings',
    minWordsTitle: 'Skip short prompts',
    minWordsHint: (n: number) => `Prompts under ${n} words are sent as typed, without refining.`,
    privacy:
      'Refinement sends your prompt text and attachment names (plus a short chat excerpt, only if you turn that on) to the chosen model, as a separate request with no tools or memory. It counts toward that provider’s usage.'
  }
}

export const PROMPT_REFINEMENT_LOCALES: PluginLocaleBundles = { en }
