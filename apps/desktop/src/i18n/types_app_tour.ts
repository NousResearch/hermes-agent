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
