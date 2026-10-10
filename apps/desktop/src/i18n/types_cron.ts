/** Copy for the Scheduled jobs (cron) overlay and its job editor. */
export interface CronCopy {
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
  timeLabel: string
  dayOfWeekLabel: string
  dayOfMonthLabel: string
  minuteLabel: string
  skipsShortMonths: (dayOfMonth: string) => string
  deliverLabel: string
  deliverNeedsHomeChannel: string
  deliverConnectHint: string
  deliverConnectAction: string
  testInChat: string
  testInChatHint: string
  testDraftKept: string
  testDraftKeptDesc: string
  backToDraft: string
  runsWithSkills: (skills: string) => string
  modelLabel: string
  modelDefault: string
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
    recipesGroup: string
    copyGroup: string
    copyName: (name: string) => string
    customize: string
    customizeHint: string
    customizedFrom: (title: string) => string
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
