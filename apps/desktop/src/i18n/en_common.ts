import type { Translations } from './types'

export const enCommon = {
  common: {
    apply: 'Apply',
    back: 'Back',
    save: 'Save',
    saving: 'Saving…',
    cancel: 'Cancel',
    change: 'Change',
    choose: 'Choose',
    clear: 'Clear',
    close: 'Close',
    collapse: 'Collapse',
    confirm: 'Confirm',
    connect: 'Connect',
    connecting: 'Connecting',
    continue: 'Continue',
    copied: 'Copied',
    copy: 'Copy',
    copyFailed: 'Copy failed',
    delete: 'Delete',
    docs: 'Docs',
    done: 'Done',
    error: 'Error',
    expand: 'Expand',
    failed: 'Failed',
    formatJson: 'Format JSON',
    free: 'Free',
    loading: 'Loading…',
    notSet: 'Not set',
    refresh: 'Refresh',
    remove: 'Remove',
    replace: 'Replace',
    retry: 'Retry',
    run: 'Run',
    send: 'Send',
    set: 'Set',
    skip: 'Skip',
    update: 'Update',
    tryHint: term => `Try “${term}”`,
    on: 'On',
    off: 'Off'
  },

  ui: {
    search: {
      clear: 'Clear search'
    },
    pagination: {
      label: 'pagination',
      previous: 'Prev',
      previousAria: 'Go to previous page',
      next: 'Next',
      nextAria: 'Go to next page'
    },
    sidebar: {
      title: 'Sidebar',
      description: 'Displays the mobile sidebar.',
      toggle: open => `${open ? 'Show' : 'Hide'} sidebar`
    }
  },

  billingBlock: {
    titleNous: 'Out of Nous credits',
    titleProvider: provider => `Out of credits — ${provider}`,
    fallbackMessage: 'Your account is out of credits. Add credits to keep going.',
    openBilling: 'Open billing',
    addCredits: 'Add credits',
    dismiss: 'Dismiss'
  }
} satisfies Pick<Translations, 'common' | 'ui' | 'billingBlock'>
