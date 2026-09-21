import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { uniqueStrings, resolveConfiguredPath } from "./lib/utils.mjs";
import { defaultChecksumsUrl, loadLocalFeed, loadRemoteFeed } from "./lib/feed.mjs";
import type { HookEvent, FeedPayload, AdvisoryMatch } from "./lib/types.ts";
import { loadState, persistState } from "./lib/state.ts";
import { discoverInstalledSkills, findMatches, matchKey, buildAlertMessage } from "./lib/matching.ts";
import { loadAdvisorySuppression, isAdvisorySuppressed } from "./lib/suppression.mjs";

const DEFAULT_FEED_URL =
  "https://clawsec.prompt.security/advisories/feed.json";
const DEFAULT_SCAN_INTERVAL_SECONDS = 300;
let unsignedModeWarningShown = false;

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function toEventName(event: HookEvent): string {
  const eventType = String(event.type ?? "").trim();
  const action = String(event.action ?? "").trim();
  if (!eventType || !action) return "";
  return `${eventType}:${action}`;
}

function shouldHandleEvent(event: HookEvent): boolean {
  const eventName = toEventName(event);
  return eventName === "agent:bootstrap" || eventName === "command:new";
}

function epochMs(isoTimestamp: string | null): number {
  if (!isoTimestamp) return 0;
  const parsed = Date.parse(isoTimestamp);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function scannedRecently(lastScan: string | null, minIntervalSeconds: number): boolean {
  const sinceMs = Date.now() - epochMs(lastScan);
  return sinceMs >= 0 && sinceMs < minIntervalSeconds * 1000;
}

function configuredPath(
  explicit: string | undefined,
  fallback: string,
  label: string,
): string {
  return resolveConfiguredPath(explicit, fallback, {
    label,
    onInvalid: (error, rawValue) => {
      console.warn(
        `[clawsec-advisory-guardian] invalid ${label} path "${rawValue}", using default "${fallback}": ${String(error)}`,
      );
    },
  });
}

async function loadFeed(options: {
  feedUrl: string;
  feedSignatureUrl: string;
  feedChecksumsUrl: string;
  feedChecksumsSignatureUrl: string;
  localFeedPath: string;
  localFeedSignaturePath: string;
  localFeedChecksumsPath: string;
  localFeedChecksumsSignaturePath: string;
  feedPublicKeyPath: string;
  allowUnsigned: boolean;
  verifyChecksumManifest: boolean;
}): Promise<FeedPayload> {
  const publicKeyPem = options.allowUnsigned ? "" : await fs.readFile(options.feedPublicKeyPath, "utf8");

  const remoteFeed = await loadRemoteFeed(options.feedUrl, {
    signatureUrl: options.feedSignatureUrl,
    checksumsUrl: options.feedChecksumsUrl,
    checksumsSignatureUrl: options.feedChecksumsSignatureUrl,
    publicKeyPem,
    checksumsPublicKeyPem: publicKeyPem,
    allowUnsigned: options.allowUnsigned,
    verifyChecksumManifest: options.verifyChecksumManifest,
  });
  if (remoteFeed) return remoteFeed;

  return await loadLocalFeed(options.localFeedPath, {
    signaturePath: options.localFeedSignaturePath,
    checksumsPath: options.localFeedChecksumsPath,
    checksumsSignaturePath: options.localFeedChecksumsSignaturePath,
    publicKeyPem,
    checksumsPublicKeyPem: publicKeyPem,
    allowUnsigned: options.allowUnsigned,
    verifyChecksumManifest: options.verifyChecksumManifest,
    checksumPublicKeyEntry: path.basename(options.feedPublicKeyPath),
  });
}

const handler = async (event: HookEvent): Promise<void> => {
  if (!shouldHandleEvent(event)) return;

  const installRoot = configuredPath(
    process.env.CLAWSEC_INSTALL_ROOT || process.env.INSTALL_ROOT,
    path.join(os.homedir(), ".openclaw", "skills"),
    "CLAWSEC_INSTALL_ROOT",
  );
  const suiteDir = configuredPath(
    process.env.CLAWSEC_SUITE_DIR,
    path.join(installRoot, "clawsec-suite"),
    "CLAWSEC_SUITE_DIR",
  );
  const localFeedPath = configuredPath(
    process.env.CLAWSEC_LOCAL_FEED,
    path.join(suiteDir, "advisories", "feed.json"),
    "CLAWSEC_LOCAL_FEED",
  );
  const localFeedSignaturePath = configuredPath(
    process.env.CLAWSEC_LOCAL_FEED_SIG,
    `${localFeedPath}.sig`,
    "CLAWSEC_LOCAL_FEED_SIG",
  );
  const localFeedChecksumsPath = configuredPath(
    process.env.CLAWSEC_LOCAL_FEED_CHECKSUMS,
    path.join(path.dirname(localFeedPath), "checksums.json"),
    "CLAWSEC_LOCAL_FEED_CHECKSUMS",
  );
  const localFeedChecksumsSignaturePath = configuredPath(
    process.env.CLAWSEC_LOCAL_FEED_CHECKSUMS_SIG,
    `${localFeedChecksumsPath}.sig`,
    "CLAWSEC_LOCAL_FEED_CHECKSUMS_SIG",
  );
  const feedPublicKeyPath = configuredPath(
    process.env.CLAWSEC_FEED_PUBLIC_KEY,
    path.join(suiteDir, "advisories", "feed-signing-public.pem"),
    "CLAWSEC_FEED_PUBLIC_KEY",
  );
  const stateFile = configuredPath(
    process.env.CLAWSEC_SUITE_STATE_FILE,
    path.join(os.homedir(), ".openclaw", "clawsec-suite-feed-state.json"),
    "CLAWSEC_SUITE_STATE_FILE",
  );
  const feedUrl = process.env.CLAWSEC_FEED_URL || DEFAULT_FEED_URL;
  const feedSignatureUrl = process.env.CLAWSEC_FEED_SIG_URL || `${feedUrl}.sig`;
  const feedChecksumsUrl = process.env.CLAWSEC_FEED_CHECKSUMS_URL || defaultChecksumsUrl(feedUrl);
  const feedChecksumsSignatureUrl =
    process.env.CLAWSEC_FEED_CHECKSUMS_SIG_URL || `${feedChecksumsUrl}.sig`;
  const allowUnsigned = process.env.CLAWSEC_ALLOW_UNSIGNED_FEED === "1";
  const verifyChecksumManifest = process.env.CLAWSEC_VERIFY_CHECKSUM_MANIFEST !== "0";
  const scanIntervalSeconds = parsePositiveInteger(
    process.env.CLAWSEC_HOOK_INTERVAL_SECONDS,
    DEFAULT_SCAN_INTERVAL_SECONDS,
  );

  if (allowUnsigned && !unsignedModeWarningShown) {
    unsignedModeWarningShown = true;
    console.warn(
      "[clawsec-advisory-guardian] CLAWSEC_ALLOW_UNSIGNED_FEED=1 is enabled. " +
        "This bypass is temporary migration compatibility and should be removed as soon as signed feed artifacts are available.",
    );
  }

  const forceScan = toEventName(event) === "command:new";
  const state = await loadState(stateFile);
  if (!forceScan && scannedRecently(state.last_hook_scan, scanIntervalSeconds)) {
    return;
  }

  let feed: FeedPayload;
  try {
    feed = await loadFeed({
      feedUrl,
      feedSignatureUrl,
      feedChecksumsUrl,
      feedChecksumsSignatureUrl,
      localFeedPath,
      localFeedSignaturePath,
      localFeedChecksumsPath,
      localFeedChecksumsSignaturePath,
      feedPublicKeyPath,
      allowUnsigned,
      verifyChecksumManifest,
    });
  } catch (error) {
    console.warn(`[clawsec-advisory-guardian] failed to load advisory feed: ${String(error)}`);
    return;
  }

  const nowIso = new Date().toISOString();
  state.last_hook_scan = nowIso;
  state.last_feed_check = nowIso;

  if (typeof feed.updated === "string" && feed.updated.trim()) {
    state.last_feed_updated = feed.updated;
  }

  const advisoryIds = feed.advisories
    .map((advisory) => advisory.id)
    .filter((id): id is string => typeof id === "string" && id.trim() !== "");
  state.known_advisories = uniqueStrings([...state.known_advisories, ...advisoryIds]);

  const installedSkills = await discoverInstalledSkills(installRoot);
  const allMatches = findMatches(feed, installedSkills);

  if (allMatches.length === 0) {
    await persistState(stateFile, state);
    return;
  }

  // Load advisory suppression config (sentinel-gated: requires enabledFor: ["advisory"])
  let suppressionConfig;
  try {
    suppressionConfig = await loadAdvisorySuppression();
  } catch (err) {
    console.warn(`[clawsec-advisory-guardian] failed to load suppression config: ${String(err)}`);
    suppressionConfig = { suppressions: [], enabledFor: [], source: "none" };
  }

  // Partition matches into active and suppressed
  const matches: AdvisoryMatch[] = [];
  const suppressedMatches: AdvisoryMatch[] = [];
  for (const match of allMatches) {
    if (isAdvisorySuppressed(match, suppressionConfig.suppressions)) {
      suppressedMatches.push(match);
    } else {
      matches.push(match);
    }
  }

  const unseenMatches: AdvisoryMatch[] = [];
  for (const match of matches) {
    const key = matchKey(match);
    if (state.notified_matches[key]) {
      continue;
    }
    unseenMatches.push(match);
    state.notified_matches[key] = nowIso;
  }

  if (unseenMatches.length > 0 && Array.isArray(event.messages)) {
    event.messages.push(buildAlertMessage(unseenMatches, installRoot));
  }

  if (suppressedMatches.length > 0 && Array.isArray(event.messages)) {
    event.messages.push(
      `[clawsec-advisory-guardian] ${suppressedMatches.length} advisory match(es) suppressed by allowlist config.`,
    );
  }

  await persistState(stateFile, state);
};

export default handler;
