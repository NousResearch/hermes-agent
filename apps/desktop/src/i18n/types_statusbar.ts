export interface StatusbarTranslations {
  unknown: string
  restart: string
  update: string
  updateInProgress: string
  commitsBehind: (count: number, branch: string) => string
  releaseAvailable: (tag: string) => string
  desktopVersion: (version: string) => string
  backendVersion: (version: string) => string
  clientLabel: (version: string) => string
  connectionSsh: (host: string) => string
  connectionRemote: (host: string) => string
  connectionCloud: (host: string) => string
  connectionCloudTooltip: (host: string) => string
  connectionSshTooltip: (host: string) => string
  connectionRemoteTooltip: (host: string) => string
  backendLabel: (version: string) => string
  commit: (sha: string) => string
  branch: (branch: string) => string
  closeCommandCenter: string
  openCommandCenter: string
  showTerminal: string
  hideTerminal: string
  gateway: string
  backend: string
  messagingStopped: string
  messagingDegraded: (name: string) => string
  gatewayReady: string
  gatewayNeedsSetup: string
  gatewayUnavailable: string
  gatewayChecking: string
  gatewayConnecting: string
  gatewayOffline: string
  gatewayRestarting: string
  gatewayTitle: string
  customizeTitle: string
  hideStatusbar: string
  resetStatusbar: string
  toggleApprovalMode: string
  toggleBackendVersion: string
  toggleCacheHitRate: string
  toggleCommandCenter: string
  toggleContextUsage: string
  toggleRunningTimer: string
  toggleSessionActivity: string
  toggleSessionTimer: string
  toggleTerminal: string
  toggleTokensPerSecond: string
  toggleVersion: string
  toggleFreeTier: string
  toggleWorkspace: string
  cacheHitRateTitle: string
  sessionActivityLabel: (turns: number, steps: number) => string
  sessionActivityTitle: string
  tokensPerSecondTitle: string
  agents: string
  closeAgents: string
  openAgents: string
  subagents: (count: number) => string
  failed: (count: number) => string
  running: (count: number) => string
  cron: string
  openCron: string
  webhooks: string
  openWebhooks: string
  starmap: string
  openStarmap: string
  turnRunning: string
  contextUsage: string
  compressions: (count: number) => string
  systemResources: {
    title: string
    loading: string
    gpuUtilization: string
    gpuMemory: string
    ram: string
    unifiedNote: string
    toggle: string
  }
  contextUsagePanel: {
    categories: {
      conversation: string
      mcp: string
      memory: string
      rules: string
      skills: string
      subagent_definitions: string
      system_prompt: string
      tool_definitions: string
    }
    empty: string
    loading: string
    percentFull: (percent: number) => string
    title: string
    tokenSummary: (used: string, max: string) => string
  }
  focusedSince: string
  focusedSinceTitle: string
  yoloOn: string
  yoloOff: string
  modelNone: string
  noModel: string
  switchModel: string
  openModelPicker: string
  modelPinned: string
  modelTitle: (provider: string, model: string) => string
  providerModelTitle: (provider: string, model: string) => string
}
