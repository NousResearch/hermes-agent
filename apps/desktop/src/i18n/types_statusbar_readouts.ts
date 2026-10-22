// Per-turn status-bar readout copy; `Translations` intersects this into
// `shell.statusbar`. Split out of types.ts, which may only shrink.
export interface StatusbarReadoutTranslations {
  cacheHitRateTitle: string
  sessionCostTitle: string
  tokensPerSecondTitle: string
  toggleCacheHitRate: string
  toggleSessionCost: string
  toggleTokensPerSecond: string
}
