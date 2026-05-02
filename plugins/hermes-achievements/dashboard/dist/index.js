(function () {
  "use strict";
  // hermes-achievements dashboard plugin
  // Originally authored by @PCinkusz — https://github.com/PCinkusz/hermes-achievements (MIT).
  // Bundled into hermes-agent. Upstream repo remains the staging ground for new
  // badges and UI iteration; the in-progress scan banner below is a small addition
  // layered on top of the original dist bundle.
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const hooks = SDK.hooks;
  const C = SDK.components;
  const cn = SDK.utils.cn;

  const LUCIDE = {"flame":"<path d=\"M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z\" />","avalanche":"<path d=\"m8 3 4 8 5-5 5 15H2L8 3z\" />\n  <path d=\"M4.14 15.08c2.62-1.57 5.24-1.43 7.86.42 2.74 1.94 5.49 2 8.23.19\" />","nodes":"<rect x=\"16\" y=\"16\" width=\"6\" height=\"6\" rx=\"1\" />\n  <rect x=\"2\" y=\"16\" width=\"6\" height=\"6\" rx=\"1\" />\n  <rect x=\"9\" y=\"2\" width=\"6\" height=\"6\" rx=\"1\" />\n  <path d=\"M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3\" />\n  <path d=\"M12 12V8\" />","rocket":"<path d=\"M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z\" />\n  <path d=\"m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z\" />\n  <path d=\"M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0\" />\n  <path d=\"M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5\" />","branch":"<line x1=\"6\" x2=\"6\" y1=\"3\" y2=\"15\" />\n  <circle cx=\"18\" cy=\"6\" r=\"3\" />\n  <circle cx=\"6\" cy=\"18\" r=\"3\" />\n  <path d=\"M18 9a9 9 0 0 1-9 9\" />","daemon":"<path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\" />\n  <path d=\"M21 3v5h-5\" />","clock":"<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <polyline points=\"12 6 12 12 16 14\" />","warning":"<path d=\"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3\" />\n  <path d=\"M12 9v4\" />\n  <path d=\"M12 17h.01\" />","wine":"<path d=\"M8 22h8\" />\n  <path d=\"M7 10h10\" />\n  <path d=\"M12 15v7\" />\n  <path d=\"M12 15a5 5 0 0 0 5-5c0-2-.5-4-2-8H9c-1.5 4-2 6-2 8a5 5 0 0 0 5 5Z\" />","scroll":"<path d=\"M15 12h-5\" />\n  <path d=\"M15 8h-5\" />\n  <path d=\"M19 17V5a2 2 0 0 0-2-2H4\" />\n  <path d=\"M8 21h12a2 2 0 0 0 2-2v-1a1 1 0 0 0-1-1H11a1 1 0 0 0-1 1v1a2 2 0 1 1-4 0V5a2 2 0 1 0-4 0v2a1 1 0 0 0 1 1h3\" />","plug":"<path d=\"m19 5 3-3\" />\n  <path d=\"m2 22 3-3\" />\n  <path d=\"M6.3 20.3a2.4 2.4 0 0 0 3.4 0L12 18l-6-6-2.3 2.3a2.4 2.4 0 0 0 0 3.4Z\" />\n  <path d=\"M7.5 13.5 10 11\" />\n  <path d=\"M10.5 16.5 13 14\" />\n  <path d=\"m12 6 6 6 2.3-2.3a2.4 2.4 0 0 0 0-3.4l-2.6-2.6a2.4 2.4 0 0 0-3.4 0Z\" />","lock":"<circle cx=\"12\" cy=\"16\" r=\"1\" />\n  <rect x=\"3\" y=\"10\" width=\"18\" height=\"12\" rx=\"2\" />\n  <path d=\"M7 10V7a5 5 0 0 1 10 0v3\" />","package_skull":"<path d=\"M21 10V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l2-1.14\" />\n  <path d=\"m7.5 4.27 9 5.15\" />\n  <polyline points=\"3.29 7 12 12 20.71 7\" />\n  <line x1=\"12\" x2=\"12\" y1=\"22\" y2=\"12\" />\n  <path d=\"m17 13 5 5m-5 0 5-5\" />","restart":"<path d=\"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8\" />\n  <path d=\"M21 3v5h-5\" />\n  <path d=\"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16\" />\n  <path d=\"M8 16H3v5\" />","key":"<path d=\"M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z\" />\n  <circle cx=\"16.5\" cy=\"7.5\" r=\".5\" fill=\"currentColor\" />","colon":"<path d=\"M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1\" />\n  <path d=\"M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1\" />","container":"<path d=\"M22 7.7c0-.6-.4-1.2-.8-1.5l-6.3-3.9a1.72 1.72 0 0 0-1.7 0l-10.3 6c-.5.2-.9.8-.9 1.4v6.6c0 .5.4 1.2.8 1.5l6.3 3.9a1.72 1.72 0 0 0 1.7 0l10.3-6c.5-.3.9-1 .9-1.5Z\" />\n  <path d=\"M10 21.9V14L2.1 9.1\" />\n  <path d=\"m10 14 11.9-6.9\" />\n  <path d=\"M14 19.8v-8.1\" />\n  <path d=\"M18 17.5V9.4\" />","melting_clock":"<line x1=\"10\" x2=\"14\" y1=\"2\" y2=\"2\" />\n  <line x1=\"12\" x2=\"15\" y1=\"14\" y2=\"11\" />\n  <circle cx=\"12\" cy=\"14\" r=\"8\" />","pencil":"<path d=\"M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z\" />\n  <path d=\"m15 5 4 4\" />","blueprint":"<path d=\"m12.99 6.74 1.93 3.44\" />\n  <path d=\"M19.136 12a10 10 0 0 1-14.271 0\" />\n  <path d=\"m21 21-2.16-3.84\" />\n  <path d=\"m3 21 8.02-14.26\" />\n  <circle cx=\"12\" cy=\"5\" r=\"2\" />","pixel":"<path d=\"M3 7V5a2 2 0 0 1 2-2h2\" />\n  <path d=\"M17 3h2a2 2 0 0 1 2 2v2\" />\n  <path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" />\n  <path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />\n  <path d=\"M7 12h10\" />","ship":"<path d=\"M12 10.189V14\" />\n  <path d=\"M12 2v3\" />\n  <path d=\"M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6\" />\n  <path d=\"M19.38 20A11.6 11.6 0 0 0 21 14l-8.188-3.639a2 2 0 0 0-1.624 0L3 14a11.6 11.6 0 0 0 2.81 7.76\" />\n  <path d=\"M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1s1.2 1 2.5 1c2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1\" />","spark_cursor":"<path d=\"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z\" />\n  <path d=\"M20 3v4\" />\n  <path d=\"M22 5h-4\" />\n  <path d=\"M4 17v2\" />\n  <path d=\"M5 18H3\" />","needle":"<path d=\"M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z\" />","hammer_scroll":"<path d=\"m15 12-8.373 8.373a1 1 0 1 1-3-3L12 9\" />\n  <path d=\"m18 15 4-4\" />\n  <path d=\"m21.5 11.5-1.914-1.914A2 2 0 0 1 19 8.172V7l-2.26-2.26a6 6 0 0 0-4.202-1.756L9 2.96l.92.82A6.18 6.18 0 0 1 12 8.4V10l2 2h1.172a2 2 0 0 1 1.414.586L18.5 14.5\" />","anvil":"<path d=\"M7 10H6a4 4 0 0 1-4-4 1 1 0 0 1 1-1h4\" />\n  <path d=\"M7 5a1 1 0 0 1 1-1h13a1 1 0 0 1 1 1 7 7 0 0 1-7 7H8a1 1 0 0 1-1-1z\" />\n  <path d=\"M9 12v5\" />\n  <path d=\"M15 12v5\" />\n  <path d=\"M5 20a3 3 0 0 1 3-3h8a3 3 0 0 1 3 3 1 1 0 0 1-1 1H6a1 1 0 0 1-1-1\" />","crystal":"<path d=\"M6 3h12l4 6-10 13L2 9Z\" />\n  <path d=\"M11 3 8 9l4 13 4-13-3-6\" />\n  <path d=\"M2 9h20\" />","palace":"<line x1=\"3\" x2=\"21\" y1=\"22\" y2=\"22\" />\n  <line x1=\"6\" x2=\"6\" y1=\"18\" y2=\"11\" />\n  <line x1=\"10\" x2=\"10\" y1=\"18\" y2=\"11\" />\n  <line x1=\"14\" x2=\"14\" y1=\"18\" y2=\"11\" />\n  <line x1=\"18\" x2=\"18\" y1=\"18\" y2=\"11\" />\n  <polygon points=\"12 2 20 7 4 7\" />","dragon":"<path d=\"M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z\" />","antenna":"<path d=\"M4.9 16.1C1 12.2 1 5.8 4.9 1.9\" />\n  <path d=\"M7.8 4.7a6.14 6.14 0 0 0-.8 7.5\" />\n  <circle cx=\"12\" cy=\"9\" r=\"2\" />\n  <path d=\"M16.2 4.8c2 2 2.26 5.11.8 7.47\" />\n  <path d=\"M19.1 1.9a9.96 9.96 0 0 1 0 14.1\" />\n  <path d=\"M9.5 18h5\" />\n  <path d=\"m8 22 4-11 4 11\" />","puzzle":"<path d=\"M15.39 4.39a1 1 0 0 0 1.68-.474 2.5 2.5 0 1 1 3.014 3.015 1 1 0 0 0-.474 1.68l1.683 1.682a2.414 2.414 0 0 1 0 3.414L19.61 15.39a1 1 0 0 1-1.68-.474 2.5 2.5 0 1 0-3.014 3.015 1 1 0 0 1 .474 1.68l-1.683 1.682a2.414 2.414 0 0 1-3.414 0L8.61 19.61a1 1 0 0 0-1.68.474 2.5 2.5 0 1 1-3.014-3.015 1 1 0 0 0 .474-1.68l-1.683-1.682a2.414 2.414 0 0 1 0-3.414L4.39 8.61a1 1 0 0 1 1.68.474 2.5 2.5 0 1 0 3.014-3.015 1 1 0 0 1-.474-1.68l1.683-1.682a2.414 2.414 0 0 1 3.414 0z\" />","rewind":"<path d=\"M9 14 4 9l5-5\" />\n  <path d=\"M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11\" />","spiral":"<path d=\"M13 16a3 3 0 0 1 2.24 5\" />\n  <path d=\"M18 12h.01\" />\n  <path d=\"M18 21h-8a4 4 0 0 1-4-4 7 7 0 0 1 7-7h.2L9.6 6.4a1 1 0 1 1 2.8-2.8L15.8 7h.2c3.3 0 6 2.7 6 6v1a2 2 0 0 1-2 2h-1a3 3 0 0 0-3 3\" />\n  <path d=\"M20 8.54V4a2 2 0 1 0-4 0v3\" />\n  <path d=\"M7.612 12.524a3 3 0 1 0-1.6 4.3\" />","quote":"<path d=\"M16 3a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2 1 1 0 0 1 1 1v1a2 2 0 0 1-2 2 1 1 0 0 0-1 1v2a1 1 0 0 0 1 1 6 6 0 0 0 6-6V5a2 2 0 0 0-2-2z\" />\n  <path d=\"M5 3a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2 1 1 0 0 1 1 1v1a2 2 0 0 1-2 2 1 1 0 0 0-1 1v2a1 1 0 0 0 1 1 6 6 0 0 0 6-6V5a2 2 0 0 0-2-2z\" />","compass":"<path d=\"m16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z\" />\n  <circle cx=\"12\" cy=\"12\" r=\"10\" />","browser":"<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <path d=\"M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20\" />\n  <path d=\"M2 12h20\" />","terminal":"<polyline points=\"4 17 10 11 4 5\" />\n  <line x1=\"12\" x2=\"20\" y1=\"19\" y2=\"19\" />","wand":"<path d=\"m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72\" />\n  <path d=\"m14 7 3 3\" />\n  <path d=\"M5 6v4\" />\n  <path d=\"M19 14v4\" />\n  <path d=\"M10 2v2\" />\n  <path d=\"M7 8H3\" />\n  <path d=\"M21 16h-4\" />\n  <path d=\"M11 3H9\" />","folder":"<path d=\"M10.7 20H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H20a2 2 0 0 1 2 2v4.1\" />\n  <path d=\"m21 21-1.9-1.9\" />\n  <circle cx=\"17\" cy=\"17\" r=\"3\" />","eye":"<path d=\"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0\" />\n  <circle cx=\"12\" cy=\"12\" r=\"3\" />","wave":"<path d=\"M2 13a2 2 0 0 0 2-2V7a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0V4a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0v-4a2 2 0 0 1 2-2\" />","swap":"<path d=\"m17 2 4 4-4 4\" />\n  <path d=\"M3 11v-1a4 4 0 0 1 4-4h14\" />\n  <path d=\"m7 22-4-4 4-4\" />\n  <path d=\"M21 13v1a4 4 0 0 1-4 4H3\" />","router":"<rect width=\"20\" height=\"8\" x=\"2\" y=\"14\" rx=\"2\" />\n  <path d=\"M6.01 18H6\" />\n  <path d=\"M10.01 18H10\" />\n  <path d=\"M15 10v4\" />\n  <path d=\"M17.84 7.17a4 4 0 0 0-5.66 0\" />\n  <path d=\"M20.66 4.34a8 8 0 0 0-11.31 0\" />","codex":"<path d=\"M10 9.5 8 12l2 2.5\" />\n  <path d=\"m14 9.5 2 2.5-2 2.5\" />\n  <rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />","prism":"<path d=\"M6 3h12l4 6-10 13L2 9Z\" />\n  <path d=\"M11 3 8 9l4 13 4-13-3-6\" />\n  <path d=\"M2 9h20\" />","marathon":"<line x1=\"10\" x2=\"14\" y1=\"2\" y2=\"2\" />\n  <line x1=\"12\" x2=\"15\" y1=\"14\" y2=\"11\" />\n  <circle cx=\"12\" cy=\"14\" r=\"8\" />","calendar":"<path d=\"M8 2v4\" />\n  <path d=\"M16 2v4\" />\n  <rect width=\"18\" height=\"18\" x=\"3\" y=\"4\" rx=\"2\" />\n  <path d=\"M3 10h18\" />\n  <path d=\"M8 14h.01\" />\n  <path d=\"M12 14h.01\" />\n  <path d=\"M16 14h.01\" />\n  <path d=\"M8 18h.01\" />\n  <path d=\"M12 18h.01\" />\n  <path d=\"M16 18h.01\" />","moon":"<path d=\"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z\" />","cache":"<ellipse cx=\"12\" cy=\"5\" rx=\"9\" ry=\"3\" />\n  <path d=\"M3 5V19A9 3 0 0 0 21 19V5\" />\n  <path d=\"M3 12A9 3 0 0 0 21 12\" />","secret":"<path d=\"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z\" />\n  <path d=\"M9.1 9a3 3 0 0 1 5.82 1c0 2-3 3-3 3\" />\n  <path d=\"M12 17h.01\" />"};

  // ---------------------------------------------------------------------------
  // i18n — Chinese translations for UI chrome
  // ---------------------------------------------------------------------------
  const useI18n = SDK.useI18n;
  const I18N = {
    en: {
      kicker: "Agentic Gamerscore",
      title: "Hermes Achievements",
      scanDesc: "Scanning Hermes session history. First scan can take 5\u201310 seconds on large histories.",
      buildingProfile: "Building achievement profile\u2026",
      readingSessions: "Reading sessions, tool calls, model metadata, and unlock state.",
      rescan: "Rescan",
      unlocked: "Unlocked",
      discovered: "Discovered",
      secrets: "Secrets",
      secret: "Secret",
      hidden: "Hidden",
      complete: "Complete",
      objective: "Objective",
      target: "Target",
      earnedBadges: "earned badges",
      knownNotEarned: "known, not earned yet",
      hiddenUntilSignal: "hidden until first signal",
      highestTier: "Highest tier",
      latest: "Latest",
      noneYet: "None yet",
      runHermesMore: "run Hermes more",
      tiers: "Tiers",
      secretAchievements: "Secret achievements",
      secretDesc: "Secrets hide their exact trigger. Once Hermes sees a related signal, the card becomes Discovered and shows its requirement.",
      recentUnlocks: "Recent unlocks",
      noSecretsLeft: "No hidden secrets left in this scan.",
      secretClue: "Clue: secrets usually start from unusual failure or power-user patterns \u2014 port conflicts, permission walls, missing env vars, YAML mistakes, Docker collisions, rollback/checkpoint use, cache hits, or tiny fixes after lots of red text.",
      evidence: "Evidence",
      noEvidence: "No evidence yet",
      whatCounts: "What counts",
      howToReveal: "How to reveal",
      scanStatus: "Scan status",
      scanStatusDesc: "Hermes is scanning local history once, then cards will appear automatically. Nothing is stuck if this takes a few seconds.",
      whatIsScanned: "What is scanned",
      whatIsScannedDesc: "Sessions, tool calls, model metadata, errors, achievements, and local unlock state.",
      startingScan: "Starting achievement scan\u2026",
      all: "All",
      allFilter: "all",
      unlockedFilter: "unlocked",
      discoveredFilter: "discovered",
      secretFilter: "secret",
    },
    zh: {
      kicker: "\u6e38\u620f\u5316\u6210\u5c31\u79ef\u5206",
      title: "Hermes \u6210\u5c31\u7cfb\u7edf",
      scanDesc: "\u6b63\u5728\u626b\u63cf Hermes \u4f1a\u8bdd\u5386\u53f2\u8bb0\u5f55\uff0c\u9996\u6b21\u626b\u63cf\u53ef\u80fd\u9700\u8981 5\u201310 \u79d2\u3002",
      buildingProfile: "\u6b63\u5728\u6784\u5efa\u6210\u5c31\u6863\u6848\u2026",
      readingSessions: "\u6b63\u5728\u8bfb\u53d6\u4f1a\u8bdd\u3001\u5de5\u5177\u8c03\u7528\u3001\u6a21\u578b\u5143\u6570\u636e\u548c\u89e3\u9501\u72b6\u6001\u3002",
      rescan: "\u91cd\u65b0\u626b\u63cf",
      unlocked: "\u5df2\u89e3\u9501",
      discovered: "\u5df2\u53d1\u73b0",
      secrets: "\u9690\u85cf\u6210\u5c31",
      secret: "\u9690\u85cf",
      hidden: "\u9690\u85cf",
      complete: "\u5df2\u5b8c\u6210",
      objective: "\u76ee\u6807",
      target: "\u76ee\u6807",
      earnedBadges: "\u5df2\u83b7\u5f97\u5fbd\u7ae0",
      knownNotEarned: "\u5df2\u77e5\u4f46\u672a\u83b7\u5f97",
      hiddenUntilSignal: "\u9690\u85cf\uff0c\u76f4\u5230\u89e6\u53d1\u6761\u4ef6",
      highestTier: "\u6700\u9ad8\u7b49\u7ea7",
      latest: "\u6700\u8fd1\u83b7\u5f97",
      noneYet: "\u6682\u65e0",
      runHermesMore: "\u591a\u4f7f\u7528 Hermes",
      tiers: "\u7b49\u7ea7",
      secretAchievements: "\u9690\u85cf\u6210\u5c31",
      secretDesc: "\u9690\u85cf\u6210\u5c31\u4f1a\u9690\u85cf\u5176\u5177\u4f53\u89e6\u53d1\u6761\u4ef6\u3002\u5f53 Hermes \u68c0\u6d4b\u5230\u76f8\u5173\u4fe1\u53f7\u65f6\uff0c\u5361\u7247\u5c06\u53d8\u4e3a\u5df2\u53d1\u73b0\u72b6\u6001\u5e76\u663e\u793a\u5176\u8981\u6c42\u3002",
      recentUnlocks: "\u6700\u8fd1\u89e3\u9501",
      noSecretsLeft: "\u672c\u6b21\u626b\u63cf\u4e2d\u6ca1\u6709\u5269\u4f59\u7684\u9690\u85cf\u6210\u5c31\u3002",
      secretClue: "\u63d0\u793a\uff1a\u9690\u85cf\u6210\u5c31\u901a\u5e38\u6765\u6e90\u4e8e\u5f02\u5e38\u5931\u8d25\u6216\u9ad8\u7ea7\u7528\u6237\u64cd\u4f5c\u6a21\u5f0f\u2014\u2014\u7aef\u53e3\u51b2\u7a81\u3001\u6743\u9650\u95ee\u9898\u3001\u73af\u5883\u53d8\u91cf\u7f3a\u5931\u3001YAML \u8bed\u6cd5\u9519\u8bef\u3001Docker \u547d\u540d\u51b2\u7a81\u3001\u56de\u6eda/\u68c0\u67e5\u70b9\u4f7f\u7528\u3001\u7f13\u5b58\u547d\u4e2d\uff0c\u6216\u5927\u91cf\u7ea2\u8272\u9519\u8bef\u6587\u672c\u540e\u7684\u5c0f\u4fee\u590d\u3002",
      evidence: "\u8bc1\u636e",
      noEvidence: "\u6682\u65e0\u8bc1\u636e",
      whatCounts: "\u8fbe\u6210\u6761\u4ef6",
      howToReveal: "\u5982\u4f55\u89e6\u53d1",
      scanStatus: "\u626b\u63cf\u72b6\u6001",
      scanStatusDesc: "Hermes \u6b63\u5728\u626b\u63cf\u672c\u5730\u5386\u53f2\u8bb0\u5f55\uff0c\u626b\u63cf\u5b8c\u6210\u540e\u5361\u7247\u5c06\u81ea\u52a8\u51fa\u73b0\u3002\u5982\u679c\u8017\u65f6\u8f83\u957f\uff0c\u8bf7\u8010\u5fc3\u7b49\u5f85\u3002",
      whatIsScanned: "\u626b\u63cf\u5185\u5bb9",
      whatIsScannedDesc: "\u4f1a\u8bdd\u3001\u5de5\u5177\u8c03\u7528\u3001\u6a21\u578b\u5143\u6570\u636e\u3001\u9519\u8bef\u4fe1\u606f\u3001\u6210\u5c31\u6570\u636e\u548c\u672c\u5730\u89e3\u9501\u72b6\u6001\u3002",
      startingScan: "\u6b63\u5728\u542f\u52a8\u6210\u5c31\u626b\u63cf\u2026",
      all: "\u5168\u90e8",
      allFilter: "\u5168\u90e8",
      unlockedFilter: "\u5df2\u89e3\u9501",
      discoveredFilter: "\u5df2\u53d1\u73b0",
      secretFilter: "\u9690\u85cf",
    },
  };

  function tierClass (tier) {
    return tier ? "ha-tier-" + tier.toLowerCase() : "ha-tier-pending";
  };

  async function api(path, options) {
    const url = "/api/plugins/hermes-achievements" + path;
    const res = await fetch(url, options || {});
    if (!res.ok) {
      const text = await res.text().catch(function () { return res.statusText; });
      throw new Error(res.status + ": " + text);
    }
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch (_) {
      return null;
    }
  }

  function AchievementIcon({ icon }) {
    const svg = LUCIDE[icon] || LUCIDE.secret;
    const ref = React.useRef(null);
    React.useEffect(function () {
      if (!ref.current) return;
      const el = ref.current;
      while (el.firstChild) el.removeChild(el.firstChild);
      try {
        const doc = new DOMParser().parseFromString(
          "<svg xmlns=\"http://www.w3.org/2000/svg\">" + svg + "</svg>",
          "image/svg+xml"
        );
        if (!doc.querySelector("parsererror")) {
          Array.from(doc.documentElement.childNodes).forEach(function (n) {
            el.appendChild(document.importNode(n, true));
          });
        }
      } catch (_) {}
    }, [svg]);
    return React.createElement("svg", {
      ref: ref,
      className: "ha-lucide",
      viewBox: "0 0 24 24",
      fill: "none",
      stroke: "currentColor",
      strokeWidth: 2,
      strokeLinecap: "round",
      strokeLinejoin: "round",
      "aria-hidden": "true",
    });
  }

  function StatCard(props) {
    return React.createElement(C.Card, { className: "ha-stat" },
      React.createElement(C.CardContent, { className: "ha-stat-content" },
        React.createElement("div", { className: "ha-stat-label" }, props.label),
        React.createElement("div", { className: "ha-stat-value" }, props.value),
        props.hint && React.createElement("div", { className: "ha-stat-hint" }, props.hint)
      )
    );
  }

  function TierLegend() {
    const { locale } = useI18n();
    const L = I18N[locale] || I18N.en;
    const TIER_I18N = {Copper: locale === "zh" ? "\u94dc" : "Copper", Silver: locale === "zh" ? "\u94f6" : "Silver", Gold: locale === "zh" ? "\u91d1" : "Gold", Diamond: locale === "zh" ? "\u94bb\u77f3" : "Diamond", Olympian: locale === "zh" ? "\u5965\u6797\u5339\u65af" : "Olympian"};
    return React.createElement("div", { className: "ha-tier-legend" },
      React.createElement("strong", null, L.tiers),
      ["Copper", "Silver", "Gold", "Diamond", "Olympian"].map(function (tier, index, arr) {
        return React.createElement(React.Fragment, { key: tier },
          React.createElement("span", { className: "ha-tier-step ha-tier-" + tier.toLowerCase() },
            React.createElement("i", null),
            TIER_I18N[tier] || tier
          ),
          index < arr.length - 1 && React.createElement("span", { className: "ha-tier-arrow" }, "→")
        );
      })
    );
  }


  function LoadingSkeletonCard(props) {
    return React.createElement(C.Card, { className: "ha-card ha-skeleton-card ha-tier-pending" },
      React.createElement(C.CardContent, { className: "ha-card-content" },
        React.createElement("div", { className: "ha-card-head" },
          React.createElement("div", { className: "ha-skeleton ha-skeleton-icon" }),
          React.createElement("div", { className: "ha-skeleton-stack" },
            React.createElement("div", { className: "ha-skeleton ha-skeleton-title" }),
            React.createElement("div", { className: "ha-skeleton ha-skeleton-meta" })
          ),
          React.createElement("div", { className: "ha-badges" },
            React.createElement("div", { className: "ha-skeleton ha-skeleton-badge" }),
            React.createElement("div", { className: "ha-skeleton ha-skeleton-badge ha-skeleton-badge-short" })
          )
        ),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-line" }),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-line ha-skeleton-line-short" }),
        React.createElement("div", { className: "ha-skeleton ha-skeleton-criteria" }),
        React.createElement("div", { className: "ha-evidence-slot" }, React.createElement("div", { className: "ha-skeleton ha-skeleton-evidence" })),
        React.createElement("div", { className: "ha-progress-row" },
          React.createElement("div", { className: "ha-skeleton ha-skeleton-progress" }),
          React.createElement("div", { className: "ha-skeleton ha-skeleton-progress-text" })
        )
      )
    );
  }

  function LoadingPage() {
    const { locale } = useI18n();
    const L = I18N[locale] || I18N.en;
    return React.createElement("div", { className: "ha-page ha-page-loading" },
      React.createElement("section", { className: "ha-hero ha-loading-hero" },
        React.createElement("div", null,
          React.createElement("div", { className: "ha-kicker" }, L.kicker),
          React.createElement("h1", null, L.title),
          React.createElement("p", null, L.scanDesc)
        ),
        React.createElement("div", { className: "ha-scan-status", role: "status", "aria-live": "polite" },
          React.createElement("span", { className: "ha-scan-pulse", "aria-hidden": "true" }),
          React.createElement("div", null,
            React.createElement("strong", null, L.buildingProfile),
            React.createElement("p", null, L.readingSessions)
          )
        )
      ),
      React.createElement("div", { className: "ha-stats" },
        [L.unlocked, L.discovered, L.secrets, L.highestTier, L.latest].map(function (label) {
          return React.createElement(C.Card, { key: label, className: "ha-stat ha-skeleton-stat" },
            React.createElement(C.CardContent, { className: "ha-stat-content" },
              React.createElement("div", { className: "ha-stat-label" }, label),
              React.createElement("div", { className: "ha-skeleton ha-skeleton-stat-value" }),
              React.createElement("div", { className: "ha-skeleton ha-skeleton-stat-hint" })
            )
          );
        })
      ),
      React.createElement("section", { className: "ha-guide ha-loading-guide" },
        React.createElement("div", null,
          React.createElement("strong", null, L.scanStatus),
          React.createElement("p", null, L.scanStatusDesc)
        ),
        React.createElement("div", null,
          React.createElement("strong", null, L.whatIsScanned),
          React.createElement("p", null, L.whatIsScannedDesc)
        )
      ),
      React.createElement("section", { className: "ha-grid" }, [0, 1, 2, 3, 4, 5].map(function (i) {
        return React.createElement(LoadingSkeletonCard, { key: i });
      }))
    );
  }


  function AchievementCard({ achievement }) {
    const { locale } = useI18n();
    const L = I18N[locale] || I18N.en;
    const unlocked = achievement.unlocked;
    const progress = achievement.progress || 0;
    const pct = achievement.progress_pct || (unlocked ? 100 : 0);
    const state = achievement.state || (unlocked ? "unlocked" : "discovered");
    const stateLabel = state === "unlocked" ? L.unlocked : (state === "secret" ? L.secret : L.discovered);
    const TIER_ZH = {Copper: "\u94dc", Silver: "\u94f6", Gold: "\u91d1", Diamond: "\u94bb\u77f3", Olympian: "\u5965\u6797\u5339\u65af"};
    function tierName(t) { return locale === "zh" ? (TIER_ZH[t] || t) : t; }
    const targetTier = achievement.next_tier || achievement.tier;
    const tierLabel = achievement.tier ? tierName(achievement.tier) : (targetTier ? L.target + " " + tierName(targetTier) : (state === "secret" ? L.hidden : (unlocked ? L.complete : L.objective)));
    const progressText = state === "secret" ? L.hidden : (progress + (achievement.next_threshold ? " / " + achievement.next_threshold : ""));
    const displayName = locale === "zh" ? (achievement.name_zh || achievement.name) : achievement.name;
    const displayDesc = locale === "zh" ? (achievement.description_zh || achievement.description) : achievement.description;
    const displayCategory = locale === "zh" ? (achievement.category_zh || achievement.category) : achievement.category;
    return React.createElement(C.Card, { className: cn("ha-card", "ha-state-" + state, tierClass(achievement.tier || achievement.next_tier)) },
      React.createElement(C.CardContent, { className: "ha-card-content" },
        React.createElement("div", { className: "ha-card-head" },
          React.createElement("div", { className: "ha-icon" }, React.createElement(AchievementIcon, { icon: achievement.icon || "secret" })),
          React.createElement("div", { className: "ha-card-title-wrap" },
            React.createElement("div", { className: "ha-card-title" }, displayName),
            React.createElement("div", { className: "ha-card-category" }, displayCategory)
          ),
          React.createElement("div", { className: "ha-badges" },
            React.createElement("span", { className: "ha-state-badge" }, stateLabel),
            React.createElement("span", { className: "ha-tier-badge" }, tierLabel)
          )
        ),
        React.createElement("p", { className: "ha-description" }, displayDesc),
        achievement.criteria && React.createElement("details", { className: "ha-criteria" },
          React.createElement("summary", null, state === "secret" ? L.howToReveal : L.whatCounts),
          React.createElement("p", null, locale === "zh" ? (achievement.criteria_zh || achievement.criteria) : achievement.criteria)
        ),
        React.createElement("div", { className: "ha-evidence-slot" },
          achievement.evidence ? React.createElement("div", { className: "ha-evidence" },
            React.createElement("span", { className: "ha-evidence-label" }, L.evidence),
            React.createElement("span", { className: "ha-evidence-title" }, achievement.evidence.title || achievement.evidence.session_id || "session")
          ) : React.createElement("div", { className: "ha-evidence ha-evidence-empty", "aria-hidden": "true" }, L.noEvidence)
        ),
        React.createElement("div", { className: "ha-progress-row" },
          React.createElement("div", { className: "ha-progress-track" },
            React.createElement("div", { className: "ha-progress-fill", style: { width: Math.max(state === "secret" ? 0 : 3, Math.min(100, pct)) + "%" } })
          ),
          React.createElement("span", { className: "ha-progress-text" }, progressText)
        )
      )
    );
  }

  function AchievementsPage() {
    const { locale } = useI18n();
    const L = I18N[locale] || I18N.en;
    const [data, setData] = hooks.useState(null);
    const [loading, setLoading] = hooks.useState(true);
    const [error, setError] = hooks.useState(null);
    const [category, setCategory] = hooks.useState(L.all);
    const [visibility, setVisibility] = hooks.useState("all");

    function load() {
      setLoading(true);
      api("/achievements")
        .then(function (payload) { setData(payload); setError((payload && payload.error) || null); })
        .catch(function (err) { setError(String(err)); })
        .finally(function () { setLoading(false); });
    }
    // refresh() re-fetches without flipping the loading state — used by the
    // auto-poller during an in-progress background scan so the page updates
    // with growing unlock counts instead of flashing the loading skeleton.
    function refresh() {
      api("/achievements")
        .then(function (payload) { setData(payload); setError((payload && payload.error) || null); })
        .catch(function (err) { setError(String(err)); });
    }
    hooks.useEffect(load, []);

    // Auto-poll while the backend is still scanning. scan_meta.mode is
    // "pending" on the very first request (no cache yet) and "in_progress"
    // while the background thread is publishing partial snapshots. Once it
    // flips to "full" or "incremental" the scan is done and we stop polling.
    const scanMode = (data && data.scan_meta && data.scan_meta.mode) || null;
    const scanInFlight = scanMode === "pending" || scanMode === "in_progress";
    hooks.useEffect(function () {
      if (!scanInFlight) return undefined;
      const id = setInterval(refresh, 4000);
      return function () { clearInterval(id); };
    }, [scanInFlight]);

    const achievements = (data && data.achievements) || [];
    const catField = locale === "zh" ? "category_zh" : "category";
    const categories = [L.all].concat(Array.from(new Set(achievements.map(function (a) { return a[catField] || a.category; }))));
    const visible = achievements.filter(function (a) {
      if (category !== L.all && (a[catField] || a.category) !== category) return false;
      if (visibility === "unlocked" && a.state !== "unlocked") return false;
      if (visibility === "discovered" && a.state !== "discovered") return false;
      if (visibility === "secret" && a.state !== "secret") return false;
      return true;
    });
    const unlocked = achievements.filter(function (a) { return a.state === "unlocked"; });
    const discovered = achievements.filter(function (a) { return a.state === "discovered"; });
    const secret = achievements.filter(function (a) { return a.state === "secret"; });
    const latest = unlocked.slice().sort(function (a, b) { return (b.unlocked_at || 0) - (a.unlocked_at || 0); }).slice(0, 5);
    const TIER_ZH2 = {Copper: "\u94dc", Silver: "\u94f6", Gold: "\u91d1", Diamond: "\u94bb\u77f3", Olympian: "\u5965\u6797\u5339\u65af"};
    const highestRaw = ["Olympian", "Diamond", "Gold", "Silver", "Copper"].find(function (tier) { return unlocked.some(function (a) { return a.tier === tier; }); });
    const highest = highestRaw ? (locale === "zh" ? (TIER_ZH2[highestRaw] || highestRaw) : highestRaw) : L.noneYet;

    // Build the in-progress scan banner once so the JSX below stays readable.
    // Shows nothing when the scan is idle. When a scan is running it renders
    // a pulsing status row with "X / Y sessions · Z%" and a filling bar, so
    // the user gets continuous visual feedback during long cold scans on
    // large session databases (can take several minutes on 8000+ sessions).
    let scanBanner = null;
    if (scanInFlight) {
      const meta = (data && data.scan_meta) || {};
      const scanned = Number(meta.sessions_scanned_so_far || meta.sessions_total || 0);
      const total = Number(meta.sessions_expected_total || 0);
      const pct = total > 0 ? Math.max(0, Math.min(100, Math.floor((scanned / total) * 100))) : 0;
      const headline = scanMode === "pending"
        ? L.startingScan
        : L.buildingProfile;
      const detail = total > 0
        ? ("\u5DF2\u626B\u63CF " + scanned.toLocaleString() + " / " + total.toLocaleString() + " \u4E2A\u4F1A\u8BDD \u00B7 " + pct + "%\u3002\u968F\u7740\u5386\u53F2\u8BB0\u5F55\u7684\u52A0\u8F7D\uFF0C\u5FBD\u7AE0\u5C06\u9010\u6B65\u89E3\u9501\u3002")
        : L.readingSessions + (locale === "zh" ? "\u5FBD\u7AE0\u89E3\u9501\u540E\u5C06\u5728\u6B64\u663E\u793A\u3002" : " Badges appear here as they unlock.");
      scanBanner = React.createElement("section", { className: "ha-scan-banner", role: "status", "aria-live": "polite" },
        React.createElement("div", { className: "ha-scan-banner-head" },
          React.createElement("span", { className: "ha-scan-pulse", "aria-hidden": "true" }),
          React.createElement("div", { className: "ha-scan-banner-text" },
            React.createElement("strong", null, headline),
            React.createElement("p", null, detail)
          )
        ),
        total > 0 && React.createElement("div", { className: "ha-scan-progress-track", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": pct },
          React.createElement("div", { className: "ha-scan-progress-fill", style: { width: pct + "%" } })
        )
      );
    }

    if (loading) {
      return React.createElement(LoadingPage, null);
    }

    return React.createElement("div", { className: "ha-page" },
      React.createElement("section", { className: "ha-hero" },
        React.createElement("div", null,
          React.createElement("div", { className: "ha-kicker" }, L.kicker),
          React.createElement("h1", null, L.title),
          React.createElement("p", null, locale === "zh" ? "\u6765\u81EA\u771F\u5B9E\u4F1A\u8BDD\u5386\u53F2\u7684\u53EF\u6536\u96C6 Hermes \u5FBD\u7AE0\u3002\u5DF2\u77E5\u7684\u672A\u5B8C\u6210\u6210\u5C31\u663E\u793A\u4E3A\u5DF2\u53D1\u73B0\uFF1B\u9690\u85CF\u6210\u5C31\u5728\u7B2C\u4E00\u6B21\u5339\u914D\u884C\u4E3A\u51FA\u73B0\u524D\u4FDD\u6301\u9690\u85CF\u3002" : "Collectible Hermes badges earned from real session history. Known unfinished achievements are shown as Discovered; Secret achievements stay hidden until the first matching behavior appears.")
        ),
        React.createElement(C.Button, { onClick: load, className: "ha-refresh" }, L.rescan)
      ),
      scanBanner,
      error && React.createElement(C.Card, { className: "ha-error" }, React.createElement(C.CardContent, null, String(error))),
      React.createElement("div", { className: "ha-stats" },
        React.createElement(StatCard, { label: L.unlocked, value: (data ? data.unlocked_count : 0) + " / " + (data ? data.total_count : 0), hint: L.earnedBadges }),
        React.createElement(StatCard, { label: L.discovered, value: discovered.length, hint: L.knownNotEarned }),
        React.createElement(StatCard, { label: L.secrets, value: secret.length, hint: L.hiddenUntilSignal }),
        React.createElement(StatCard, { label: L.highestTier, value: highest, hint: (locale === "zh" ? "\u94DC \u2192 \u94F6 \u2192 \u91D1 \u2192 \u94BB\u77F3 \u2192 \u5965\u6797\u5339\u65AF" : "Copper \u2192 Silver \u2192 Gold \u2192 Diamond \u2192 Olympian") }),
        React.createElement(StatCard, { label: L.latest, value: latest[0] ? (locale === "zh" ? (latest[0].name_zh || latest[0].name) : latest[0].name) : L.noneYet, hint: latest[0] ? (locale === "zh" ? (latest[0].category_zh || latest[0].category) : latest[0].category) : L.runHermesMore })
      ),
      React.createElement("section", { className: "ha-guide" },
        React.createElement("div", null,
          React.createElement("strong", null, L.tiers),
          React.createElement(TierLegend, null)
        ),
        React.createElement("div", null,
          React.createElement("strong", null, L.secretAchievements),
          React.createElement("p", null, L.secretDesc)
        )
      ),
      React.createElement("div", { className: "ha-toolbar" },
        React.createElement("div", { className: "ha-pills" }, categories.map(function (cat) {
          return React.createElement("button", { key: cat, onClick: function () { setCategory(cat); }, className: cat === category ? "active" : "" }, cat);
        })),
        React.createElement("div", { className: "ha-pills" }, ["all", "unlocked", "discovered", "secret"].map(function (v) {
          const label = v === "all" ? L.allFilter : v === "unlocked" ? L.unlockedFilter : v === "discovered" ? L.discoveredFilter : L.secretFilter;
          return React.createElement("button", { key: v, onClick: function () { setVisibility(v); }, className: v === visibility ? "active" : "" }, label);
        }))
      ),
      latest.length > 0 && React.createElement("section", { className: "ha-latest" },
        React.createElement("h2", null, L.recentUnlocks),
        React.createElement("div", { className: "ha-latest-row" }, latest.map(function (a) {
          return React.createElement("div", { key: a.id, className: cn("ha-chip", tierClass(a.tier)) },
            React.createElement("span", { className: "ha-chip-icon" }, React.createElement(AchievementIcon, { icon: a.icon || "secret" })),
            locale === "zh" ? (a.name_zh || a.name) : a.name
          );
        }))
      ),
      visibility === "secret" && visible.length === 0 && React.createElement(C.Card, { className: "ha-secret-empty" },
        React.createElement(C.CardContent, { className: "ha-secret-empty-content" },
          React.createElement("strong", null, L.noSecretsLeft),
          React.createElement("p", null, L.secretClue)
        )
      ),
      React.createElement("section", { className: "ha-grid" }, visible.map(function (a) {
        return React.createElement(AchievementCard, { key: a.id, achievement: a });
      }))
    );
  }

  window.__HERMES_PLUGINS__.register("hermes-achievements", AchievementsPage);
})();
