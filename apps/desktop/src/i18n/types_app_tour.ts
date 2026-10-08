interface TourStopCopy {
  title: string
  text: string
}

/** The app's own tour, run when `gui_tour` starts with no steps. */
export interface AppTourTranslations {
  sessions: TourStopCopy
  composer: TourStopCopy
  newSession: TourStopCopy
  model: TourStopCopy
  /** Appended to the model stop when this computer can run a local model. */
  modelLocal: string
  capabilities: TourStopCopy
  messaging: TourStopCopy
  rightPane: TourStopCopy
}

/** The tour that closes the guided first run (components/onboarding-chat/signpost.ts). */
export interface HandoffTourTranslations {
  profileTitle: string
  profileText: string
  sessionsTitle: string
  sessionsText: string
  stayTitle: string
  stayText: string
  localTitle: string
  localText: (model: string) => string
}
