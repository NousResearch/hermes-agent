export type HookEvent = {
  type?: string;
  action?: string;
  messages?: string[];
};

export type Advisory = {
  id?: string;
  severity?: string;
  type?: string;
  application?: string | string[];
  title?: string;
  description?: string;
  action?: string;
  published?: string;
  updated?: string;
  affected?: string[];
};

export type FeedPayload = {
  version: string;
  updated?: string;
  advisories: Advisory[];
};

export type InstalledSkill = {
  name: string;
  dirName: string;
  version: string | null;
};

export type AdvisoryMatch = {
  advisory: Advisory;
  skill: InstalledSkill;
  matchedAffected: string[];
};

export type AdvisoryState = {
  schema_version: string;
  known_advisories: string[];
  last_feed_check: string | null;
  last_feed_updated: string | null;
  last_hook_scan: string | null;
  notified_matches: Record<string, string>;
};
