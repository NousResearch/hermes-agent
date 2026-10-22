import type { CronReasoningEffortOption } from '@/types/hermes'

export type CronTranslations = {
  close: string
  title: string
  count: (count: number) => string
  search: string
  loading: string
  states: Record<string, string>
  lastRunFailed: string
  editJob: string
  runAgain: string
  deliveryLabels: Record<string, string>
  scheduleLabels: Record<string, string>
  scheduleHints: Record<string, string>
  days: Record<string, string>
  dayFallback: (value: string) => string
  everyDayAt: (time: string) => string
  weekdaysAt: (time: string) => string
  everyDayOfWeekAt: (day: string, time: string) => string
  monthlyOnDayAt: (dayOfMonth: string, time: string) => string
  topOfHour: string
  everyHourAt: (minute: string) => string
  newCron: string
  emptyDescNew: string
  emptyDescSearch: string
  emptyTitleNew: string
  emptyTitleSearch: string
  last: string
  next: string
  overdueSince: string
  noRuns: string
  queuedRun: string
  manage: string
  showRuns: string
  hideRuns: string
  runHistory: string
  actionsTitle: string
  resume: string
  pause: string
  resumeTitle: string
  pauseTitle: string
  triggerNow: string
  edit: string
  deleteTitle: string
  deleteDescPrefix: string
  deleteDescSuffix: string
  deleting: string
  resumed: string
  paused: string
  triggered: string
  deleted: string
  created: string
  updated: string
  failedLoad: string
  failedUpdate: string
  failedTrigger: string
  failedDelete: string
  failedSave: string
  editTitle: string
  createTitle: string
  editDesc: string
  createDesc: string
  nameLabel: string
  namePlaceholder: string
  promptLabel: string
  scriptLabel: string
  scriptBadge: string
  promptPlaceholder: string
  frequencyLabel: string
  deliverLabel: string
  deliverNeedsHomeChannel: string
  modelLabel: string
  modelDefault: string
  reasoningLabel: string
  reasoningLabels: Record<CronReasoningEffortOption, string>
  customScheduleLabel: string
  customPlaceholder: string
  customHint: string
  optional: string
  promptRequired: string
  promptScheduleRequired: string
  scheduleRequired: string
  scriptOnlyEditHint: string
  saveChanges: string
  createAction: string
  tabs: {
    jobs: string
    blueprints: string
  }
  blueprints: {
    tab: string
    startFrom: string
    custom: string
    subtitle: string
    dialogDesc: string
    scheduleIt: string
    scheduling: string
    scheduled: string
    loading: string
    failedLoad: string
    emptyTitle: string
    emptyDesc: string
  }
}
