/** Opening a skill install link. */
export interface SkillDeepLinkCopy {
  installTitle: (name: string) => string
  installDescription: string
  installTo: string
  thisComputer: string
  installing: string
  installComplete: (name: string) => string
  destinationChanged: string
  installed: string
  source: string
}

/** Bots: reopening a gateway group by name or restored tab key (#140805). */
export interface GatewayGroupOpenCopy {
  opening: string
  lookupFailed: string
  missing: string
}
