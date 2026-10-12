import type { Translations } from './types'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into en.ts.
export const enSimpleMode = {
  interfaceMode: {
    title: 'Interface mode',
    hint: 'Changes what is shown, not what Hermes can do.',
    sessionNote: 'Set by Simple mode. A change here lasts for this session; switch to Advanced to make it yours.',
    switchToAdvanced: 'Switch to Advanced',
    simple: {
      label: 'Simple',
      description: 'For talking to Hermes. Sidebar and chat; no terminal, file or diff panes.'
    },
    advanced: {
      label: 'Advanced',
      description: 'For developers. Terminal, files, diffs, statusbar and layouts, as you set them.'
    }
  },
  chatStarters: {
    label: 'Ideas to get started',
    intro: 'Ask me anything — or start with one of these.',
    email: {
      title: 'Help me write an email',
      prompt: "Help me write an email. Here's who it's for and what I want to say: "
    },
    dinners: {
      title: "Plan this week's dinners",
      prompt: 'Plan five easy dinners for this week and give me one shopping list for all of them.'
    },
    summarize: {
      title: 'Summarize a document',
      prompt: "I'll attach a file — give me the key points."
    },
    lookup: {
      title: 'Look something up for me',
      prompt: 'Look this up for me and tell me what you find: '
    }
  }
} satisfies Pick<Translations, 'interfaceMode' | 'chatStarters'>
