/* ============================================================
   ARO — cron automations + messaging channels (seed data)
   ============================================================ */

/* ---------------- cron automations ---------------- */
export type ToneName = "neutral" | "iris" | "cyan" | "mint" | "amber" | "rose" | "sky" | "plum";

export type CronJob = {
  id: string;
  name: string;
  schedule: string; /* human, e.g. "every weekday · 9:00" */
  cron: string; /* expression, e.g. "0 9 * * 1-5" */
  agentId: string;
  channel: string | null; /* CHANNELS id, null = dashboard only */
  enabled: boolean;
  lastRun: { ok: boolean; at: string; summary: string };
  nextRun: string;
  runsThisMonth: number;
  monthlyCost: number;
  costCap: number;
  tone: ToneName;
};

export const CRON_JOBS: CronJob[] = [
  { id: "j1", name: "release-notes-digest", schedule: "every Friday · 16:00", cron: "0 16 * * 5", agentId: "aro", channel: "telegram", enabled: true, lastRun: { ok: true, at: "2h ago", summary: "14 commits → digest sent" }, nextRun: "in 6d 22h", runsThisMonth: 4, monthlyCost: 3.4, costCap: 5, tone: "mint" },
  { id: "j2", name: "dep-drift report", schedule: "Mondays · 08:00", cron: "0 8 * * 1", agentId: "glm-code", channel: "discord", enabled: true, lastRun: { ok: false, at: "4d ago", summary: "renovate API 502 · 2 retries" }, nextRun: "in 2d 14h", runsThisMonth: 4, monthlyCost: 0.62, costCap: 1, tone: "rose" },
  { id: "j3", name: "inbox-zero sweep", schedule: "every 4 hours", cron: "0 */4 * * *", agentId: "aro", channel: "email", enabled: true, lastRun: { ok: true, at: "41m ago", summary: "3 drafts queued · 1 unsub" }, nextRun: "in 3h 19m", runsThisMonth: 172, monthlyCost: 2.1, costCap: 5, tone: "iris" },
  { id: "j4", name: "standup-prep", schedule: "weekdays · 09:00", cron: "0 9 * * 1-5", agentId: "claude-code", channel: "telegram", enabled: true, lastRun: { ok: true, at: "9h ago", summary: "3 blockers surfaced" }, nextRun: "in 2d 15h", runsThisMonth: 21, monthlyCost: 1.85, costCap: 5, tone: "cyan" },
  { id: "j5", name: "weekly changelog → #product", schedule: "Fridays · 17:00", cron: "0 17 * * 5", agentId: "glm-code", channel: "slack", enabled: true, lastRun: { ok: true, at: "1h ago", summary: "22 PRs → #product" }, nextRun: "in 6d 23h", runsThisMonth: 4, monthlyCost: 0.28, costCap: 1, tone: "sky" },
  { id: "j6", name: "flaky-test triage", schedule: "nightly · 02:30", cron: "30 2 * * *", agentId: "codex", channel: "discord", enabled: true, lastRun: { ok: true, at: "16h ago", summary: "2 flakes quarantined" }, nextRun: "in 8h 30m", runsThisMonth: 30, monthlyCost: 1.9, costCap: 5, tone: "amber" },
  { id: "j7", name: "security advisory scan", schedule: "daily · 06:15", cron: "15 6 * * *", agentId: "darwin", channel: "email", enabled: true, lastRun: { ok: true, at: "12h ago", summary: "0 new CVEs · 12 pkgs" }, nextRun: "in 12h 15m", runsThisMonth: 30, monthlyCost: 0.54, costCap: 1, tone: "plum" },
  { id: "j8", name: "blog drafts review", schedule: "Sundays · 10:00", cron: "0 10 * * 0", agentId: "aro", channel: null, enabled: false, lastRun: { ok: true, at: "5d ago", summary: "3 drafts annotated" }, nextRun: "in 2d 16h", runsThisMonth: 0, monthlyCost: 0, costCap: 1, tone: "neutral" },
];

/* ---------------- messaging channels (~30-platform gateway) ---------------- */
export type Channel = {
  id: string;
  platform: string;
  identity: string; /* handle, number or room the agent speaks as */
  status: "connected" | "auth-required" | "off";
  msgs24h: number;
  spark: number[]; /* last ~12h activity */
  approvalRule: "auto <$1" | "ask always" | "readonly";
  note: string;
};

export const CHANNELS: Channel[] = [
  { id: "telegram", platform: "Telegram", identity: "@aro_agent", status: "connected", msgs24h: 86, spark: [4, 6, 5, 8, 7, 9, 12, 10, 11, 14, 12, 15], approvalRule: "auto <$1", note: "Primary thread. Voice memos transcribed inline, replies in seconds." },
  { id: "whatsapp", platform: "WhatsApp", identity: "+1 ••• ••4821", status: "connected", msgs24h: 21, spark: [1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 1, 2], approvalRule: "auto <$1", note: "Replies only when mentioned — the family group stays out of the loop." },
  { id: "discord", platform: "Discord", identity: "#agent-ops", status: "connected", msgs24h: 143, spark: [9, 12, 11, 14, 13, 16, 15, 18, 14, 17, 19, 21], approvalRule: "ask always", note: "Run notifications land here; approvals are reactions on the message." },
  { id: "slack", platform: "Slack", identity: "#product", status: "connected", msgs24h: 57, spark: [4, 3, 5, 6, 4, 5, 7, 6, 5, 8, 6, 7], approvalRule: "ask always", note: "Workspace app scoped to #product. Changelog posts, nothing else." },
  { id: "signal", platform: "Signal", identity: "+44 ••• ••3390", status: "connected", msgs24h: 9, spark: [0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 0, 1], approvalRule: "readonly", note: "Private pings only. No channels, no bots, no logs kept." },
  { id: "matrix", platform: "Matrix", identity: "@aro:matrix.org", status: "connected", msgs24h: 17, spark: [1, 2, 1, 3, 2, 2, 1, 3, 2, 4, 2, 3], approvalRule: "auto <$1", note: "Bridged to 3 rooms. Verified device, E2EE everywhere." },
  { id: "imessage", platform: "iMessage", identity: "dev@icloud.com", status: "auth-required", msgs24h: 0, spark: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], approvalRule: "ask always", note: "BlueBubbles bridge needs re-pairing after the macOS update." },
  { id: "email", platform: "Email", identity: "aro@acme.dev", status: "connected", msgs24h: 214, spark: [14, 18, 16, 21, 19, 24, 22, 27, 25, 23, 28, 26], approvalRule: "auto <$1", note: "SMTP + IMAP. The inbox-zero sweep runs on this identity." },
  { id: "sms", platform: "SMS", identity: "+1 ••• ••7704", status: "off", msgs24h: 0, spark: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], approvalRule: "ask always", note: "Paused — per-message billing. Flip on for trips only." },
  { id: "teams", platform: "Teams", identity: "Aro · Acme org", status: "off", msgs24h: 0, spark: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], approvalRule: "readonly", note: "Blocked on IT approval for custom bots in the tenant." },
  { id: "line", platform: "LINE", identity: "@aro_jp", status: "auth-required", msgs24h: 0, spark: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], approvalRule: "ask always", note: "Channel token expired — refresh in settings to resume." },
  { id: "irc", platform: "IRC", identity: "aro @ libera/#acme", status: "connected", msgs24h: 34, spark: [3, 2, 4, 3, 5, 4, 2, 3, 5, 4, 6, 5], approvalRule: "readonly", note: "Plain relay. Logs archived nightly to the workspace." },
  { id: "ntfy", platform: "ntfy", identity: "dev-vale/phone", status: "connected", msgs24h: 61, spark: [2, 5, 1, 0, 3, 4, 2, 6, 1, 3, 2, 4], approvalRule: "auto <$1", note: "Push alerts for approvals and failures — nothing else." },
  { id: "webhook", platform: "Webhook", identity: "hooks.acme.dev/aro", status: "connected", msgs24h: 12, spark: [1, 0, 2, 1, 3, 0, 2, 1, 1, 2, 0, 1], approvalRule: "auto <$1", note: "Generic JSON relay into n8n and home-assistant." },
];

/* ---------------- recent cron runs ---------------- */
export const CRON_HISTORY: { id: string; jobId: string; at: string; ok: boolean; duration: string; cost: number; note: string }[] = [
  { id: "rh1", jobId: "j3", at: "41m ago", ok: true, duration: "3.2s", cost: 0.02, note: "3 drafts queued · 1 newsletter unsubscribed" },
  { id: "rh2", jobId: "j1", at: "2h ago", ok: true, duration: "1m 08s", cost: 0.42, note: "14 commits → digest sent to @aro_agent" },
  { id: "rh3", jobId: "j7", at: "12h ago", ok: true, duration: "22.4s", cost: 0.03, note: "0 new CVEs across 12 dependencies" },
  { id: "rh4", jobId: "j6", at: "16h ago", ok: true, duration: "4m 51s", cost: 0.61, note: "2 flakes quarantined · GH-1203 GH-1187 opened" },
  { id: "rh5", jobId: "j4", at: "9h ago", ok: true, duration: "9.7s", cost: 0.09, note: "3 blockers surfaced before standup" },
  { id: "rh6", jobId: "j3", at: "5h ago", ok: true, duration: "2.8s", cost: 0.02, note: "2 drafts queued · nothing to escalate" },
  { id: "rh7", jobId: "j2", at: "4d ago", ok: false, duration: "12.1s", cost: 0, note: "renovate API 502 · failed after 2 retries" },
  { id: "rh8", jobId: "j5", at: "1h ago", ok: true, duration: "48.3s", cost: 0.28, note: "22 PRs grouped by area → #product" },
  { id: "rh9", jobId: "j6", at: "1d ago", ok: true, duration: "3m 12s", cost: 0.44, note: "clean run · 0 flakes in 50 tries" },
  { id: "rh10", jobId: "j7", at: "2d ago", ok: false, duration: "8.0s", cost: 0, note: "OSV feed timeout · fell back to cached advisories" },
];
