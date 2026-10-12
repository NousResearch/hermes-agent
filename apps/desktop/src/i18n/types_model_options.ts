// Model-options submenu copy (the hover-revealed per-model speed/effort
// editor), a `types.ts` section sibling like `types_model_menu.ts`.

export interface ModelOptionsTranslations {
  noOptions: string
  options: string
  thinking: string
  fast: string
  ultrafast: string
  useStandardSpeed: string
  auto: string
  cold: string
  effort: string
  minimal: string
  low: string
  medium: string
  high: string
  xhigh: string
  max: string
  ultra: string
  /** The CLI's `/reasoning` clamp note, e.g. "sends Max on this route". */
  sendsOnRoute: (level: string) => string
  updateFailed: string
  speedPolicy: string
  fastFailed: string
}
