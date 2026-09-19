export interface CapabilitiesTranslations {
  skills: {
    tabSkills: string
    tabToolsets: string
    configuringProfile: string
    tabMcp: string
    all: string
    searchSkills: string
    searchToolsets: string
    refresh: string
    refreshing: string
    loading: string
    noSkillsTitle: string
    noSkillsDesc: string
    noToolsetsTitle: string
    noToolsetsDesc: string
    noDescription: string
    configured: string
    needsKeys: string
    visionModelHint: string
    visionModelLink: string
    toolsetsEnabled: (enabled: number, total: number) => string
    configureToolset: (label: string) => string
    toggleToolset: (label: string, enabled: boolean) => string
    skillsLoadFailed: string
    toolsetsRefreshFailed: string
    skillEnabled: string
    skillDisabled: string
    toolsetEnabled: string
    toolsetDisabled: string
    appliesToNewSessions: (name: string) => string
    failedToUpdate: (name: string) => string
    sortMostUsed: string
    sortAlpha: string
    sortMostUsedDesc: string
    sortLeastUsedAsc: string
    enableAll: string
    disableAll: string
    disableUnused: string
    bulkUpdated: (count: number) => string
    bulkNoChange: string
    usageCount: (count: number | string) => string
    provenance: Record<'agent' | 'bundled' | 'hub', string>
    emptyNoneFound: (noun: string) => string
    emptyNothingMatches: (query: string) => string
    emptyNoneAvailable: (noun: string) => string
    changesApplyNewSessions: string
    skillUpdated: string
    edit: string
    archive: string
    skillArchivedTitle: string
    skillArchivedMessage: string
    tabPlugins: string
    plugins: {
      agentTitle: string
      agentBlurb: string
      pageBlurb: string
      halfDesktop: string
      halfDesktopHint: string
      halfAgent: string
      halfAgentIn: (profile: string) => string
      defaultProfile: string
      kindAgent: string
      kindDesktop: string
      kindBoth: string
      installAgentHere: string
      installAgentHereTip: (profile: string) => string
      installAgentHereNoOrigin: string
      desktopHalfPending: string
      desktopHalfPendingTip: string
      desktopHalfRemote: string
      desktopHalfRemoteTip: string
      emptyAll: string
      empty: string
      emptyHint: string
      loadFailed: string
      toggleFailed: (name: string) => string
      legacyBackend: string
      portableBadge: string
      catalogTitle: string
      catalogBrowse: string
      catalogHide: string
      catalogHint: string
      alreadyInstalled: (name: string) => string
      catalogProvenance: (sha: string) => string
      pinnedProvenance: (sha: string) => string
      pinnedBadge: (sha: string) => string
      tierOfficial: string
      tierCommunity: string
      updateToPin: (sha: string) => string
      updateFailed: (name: string) => string
      updated: (name: string) => string
    }
    officialCatalog: string
    officialPill: string
    hub: {
      searchPlaceholder: string
      search: string
      searching: string
      connectingHubs: string
      connectedHubs: string
      featured: string
      landingHint: string
      noResults: string
      resultCount: (count: number, ms: number | null) => string
      timedOut: (sources: string) => string
      installed: string
      install: string
      installing: string
      uninstall: string
      uninstalling: string
      updateAll: string
      updating: string
      preview: string
      scan: string
      scanning: string
      close: string
      files: string
      noReadme: string
      trust: Record<string, string>
      verdictSafe: string
      verdictCaution: string
      verdictDangerous: string
      policyAllow: string
      policyAsk: string
      policyBlock: string
      findings: (count: number) => string
      noFindings: string
      installStarted: (name: string) => string
      uninstallStarted: (name: string) => string
      updateStarted: string
      actionFailed: string
      installBlockedTitle: (name: string) => string
      installBlockedMessage: (findings: number, unverified: boolean) => string
      viewScan: string
      openLog: string
      actionLog: string
      alreadyInstalled: (name: string) => string
      pickerTitle: string
      pickerBrowse: string
      pickerHide: string
      pickerHint: string
      loadFailed: string
      previewFailed: string
      scanFailed: string
      searchFailed: string
    }
  }

  starmap: {
    title: string
    subtitle: (nodes: number, clusters: number) => string
    close: string
    refresh: string
    memory: string
    filterAll: string
    filterUsed: string
    filterLearned: string
    viewGraph: string
    loadFailed: string
    loading: string
    emptyTitle: string
    emptyDesc: string
    share: string
    shareHint: string
    shareTitle: string
    sharePlaceholder: string
    copy: string
    copied: string
    importMap: string
    importBtn: string
    importEmpty: string
    importSuccess: (nodes: number) => string
    importedBadge: string
    resetToMine: string
  }
  agents: {
    extendedTranscript: string
    transcriptTruncated: string
    transcriptUnavailable: string
    close: string
    title: string
    subtitle: string
    emptyTitle: string
    emptyDesc: string
    running: string
    failed: string
    done: string
    streaming: string
    files: string
    moreFiles: (count: number) => string
    moreAgents: (count: number) => string
    queued: string
    waitingActivity: string
    steer: string
    steerPlaceholder: string
    steerQueued: string
    stopRequested: string
    requestRejected: string
    delegation: (index: number) => string
    workers: (count: number) => string
    workersActive: (count: number) => string
    agentsCount: (count: number) => string
    activeCount: (count: number) => string
    failedCount: (count: number) => string
    toolsCount: (count: number) => string
    filesCount: (count: number) => string
    updatedAgo: (age: string) => string
    ageNow: string
    ageSeconds: (seconds: number) => string
    ageMinutes: (minutes: number) => string
    ageHours: (hours: number) => string
    ageDays: (days: number) => string
    durationSeconds: (seconds: string) => string
    durationMinutes: (minutes: number, seconds: number) => string
    tokens: (value: number | string) => string
  }
}
