#!/usr/bin/env node

import { spawnSync as runProcessSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { normalizeSkillName, uniqueStrings, resolveUserPath } from "../hooks/clawsec-advisory-guardian/lib/utils.mjs";
import { versionMatches } from "../hooks/clawsec-advisory-guardian/lib/version.mjs";
import {
  defaultChecksumsUrl,
  parseAffectedSpecifier,
  loadLocalFeed,
  loadRemoteFeed,
} from "../hooks/clawsec-advisory-guardian/lib/feed.mjs";

const DEFAULT_FEED_URL =
  "https://clawsec.prompt.security/advisories/feed.json";
const DEFAULT_SUITE_DIR = path.join(os.homedir(), ".openclaw", "skills", "clawsec-suite");
const DEFAULT_LOCAL_FEED = path.join(DEFAULT_SUITE_DIR, "advisories", "feed.json");
const DEFAULT_LOCAL_FEED_SIG = `${DEFAULT_LOCAL_FEED}.sig`;
const DEFAULT_LOCAL_FEED_CHECKSUMS = path.join(DEFAULT_SUITE_DIR, "advisories", "checksums.json");
const DEFAULT_LOCAL_FEED_CHECKSUMS_SIG = `${DEFAULT_LOCAL_FEED_CHECKSUMS}.sig`;
const DEFAULT_FEED_PUBLIC_KEY = path.join(DEFAULT_SUITE_DIR, "advisories", "feed-signing-public.pem");
const EXIT_CONFIRM_REQUIRED = 42;

function envPathOrDefault(name, fallback, label) {
  const envValue = process.env[name];
  const candidate = typeof envValue === "string" && envValue.trim() ? envValue.trim() : fallback;
  return resolveUserPath(candidate, { label });
}

function printUsage() {
  process.stderr.write(
    [
      "Usage:",
      "  node scripts/guarded_skill_install.mjs --skill <skill-name> [--version <version>] [--confirm-advisory] [--dry-run]",
      "",
      "Examples:",
      "  node scripts/guarded_skill_install.mjs --skill helper-plus --version 1.0.1",
      "  node scripts/guarded_skill_install.mjs --skill helper-plus --version 1.0.1 --confirm-advisory",
      "",
      "Exit codes:",
      "  0  success / no advisory block",
      "  42 advisory matched and second confirmation is required",
      "  1  error",
      "",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const parsed = {
    skill: "",
    version: "",
    confirmAdvisory: false,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];

    if (token === "--skill") {
      parsed.skill = String(argv[i + 1] ?? "").trim();
      i += 1;
      continue;
    }
    if (token === "--version") {
      parsed.version = String(argv[i + 1] ?? "").trim();
      i += 1;
      continue;
    }
    if (token === "--confirm-advisory") {
      parsed.confirmAdvisory = true;
      continue;
    }
    if (token === "--dry-run") {
      parsed.dryRun = true;
      continue;
    }
    if (token === "--help" || token === "-h") {
      printUsage();
      process.exit(0);
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!parsed.skill) {
    throw new Error("Missing required argument: --skill");
  }
  if (!/^[a-z0-9-]+$/.test(parsed.skill)) {
    throw new Error("Invalid --skill value. Use lowercase letters, digits, and hyphens only.");
  }

  return parsed;
}

function affectedSpecifierMatches(specifier, skillName, version) {
  const parsed = parseAffectedSpecifier(specifier);
  if (!parsed) return false;
  if (normalizeSkillName(parsed.name) !== normalizeSkillName(skillName)) return false;
  return versionMatches(version, parsed.versionSpec);
}

function affectedSpecifierMatchesWithoutVersion(specifier, skillName) {
  const parsed = parseAffectedSpecifier(specifier);
  if (!parsed) return false;
  return normalizeSkillName(parsed.name) === normalizeSkillName(skillName);
}

function advisoryLooksHighRisk(advisory) {
  const type = String(advisory.type ?? "").toLowerCase();
  const severity = String(advisory.severity ?? "").toLowerCase();
  const combined = `${advisory.title ?? ""} ${advisory.description ?? ""} ${advisory.action ?? ""}`.toLowerCase();
  if (type === "malicious_skill" || type === "malicious_plugin") return true;
  if (/\b(malicious|exfiltrate|exfiltration|backdoor|trojan|stealer|credential theft)\b/.test(combined)) return true;
  if (/\b(remove|uninstall|disable|do not use|quarantine)\b/.test(combined)) return true;
  if (severity === "critical") return true;
  return false;
}

async function loadFeed() {
  const feedUrl = process.env.CLAWSEC_FEED_URL || DEFAULT_FEED_URL;
  const feedSignatureUrl = process.env.CLAWSEC_FEED_SIG_URL || `${feedUrl}.sig`;
  const feedChecksumsUrl = process.env.CLAWSEC_FEED_CHECKSUMS_URL || defaultChecksumsUrl(feedUrl);
  const feedChecksumsSignatureUrl = process.env.CLAWSEC_FEED_CHECKSUMS_SIG_URL || `${feedChecksumsUrl}.sig`;
  const localFeedPath = envPathOrDefault("CLAWSEC_LOCAL_FEED", DEFAULT_LOCAL_FEED, "CLAWSEC_LOCAL_FEED");
  const localFeedSigPath = envPathOrDefault("CLAWSEC_LOCAL_FEED_SIG", DEFAULT_LOCAL_FEED_SIG, "CLAWSEC_LOCAL_FEED_SIG");
  const localFeedChecksumsPath = envPathOrDefault(
    "CLAWSEC_LOCAL_FEED_CHECKSUMS",
    DEFAULT_LOCAL_FEED_CHECKSUMS,
    "CLAWSEC_LOCAL_FEED_CHECKSUMS",
  );
  const localFeedChecksumsSigPath = envPathOrDefault(
    "CLAWSEC_LOCAL_FEED_CHECKSUMS_SIG",
    DEFAULT_LOCAL_FEED_CHECKSUMS_SIG,
    "CLAWSEC_LOCAL_FEED_CHECKSUMS_SIG",
  );
  const feedPublicKeyPath = envPathOrDefault("CLAWSEC_FEED_PUBLIC_KEY", DEFAULT_FEED_PUBLIC_KEY, "CLAWSEC_FEED_PUBLIC_KEY");
  const allowUnsigned = process.env.CLAWSEC_ALLOW_UNSIGNED_FEED === "1";
  const verifyChecksumManifest = process.env.CLAWSEC_VERIFY_CHECKSUM_MANIFEST !== "0";

  if (allowUnsigned) {
    process.stderr.write(
      "WARNING: CLAWSEC_ALLOW_UNSIGNED_FEED=1 is enabled. This temporary migration compatibility bypass should be removed once signed feed artifacts are available.\n",
    );
  }

  const publicKeyPem = allowUnsigned ? "" : await fs.readFile(feedPublicKeyPath, "utf8");

  const remoteFeed = await loadRemoteFeed(feedUrl, {
    signatureUrl: feedSignatureUrl,
    checksumsUrl: feedChecksumsUrl,
    checksumsSignatureUrl: feedChecksumsSignatureUrl,
    publicKeyPem,
    checksumsPublicKeyPem: publicKeyPem,
    allowUnsigned,
    verifyChecksumManifest,
  });
  if (remoteFeed) return { feed: remoteFeed, source: `remote:${feedUrl}` };

  const localFeed = await loadLocalFeed(localFeedPath, {
    signaturePath: localFeedSigPath,
    checksumsPath: localFeedChecksumsPath,
    checksumsSignaturePath: localFeedChecksumsSigPath,
    publicKeyPem,
    checksumsPublicKeyPem: publicKeyPem,
    allowUnsigned,
    verifyChecksumManifest,
    checksumPublicKeyEntry: path.basename(feedPublicKeyPath),
  });
  return { feed: localFeed, source: `local:${localFeedPath}` };
}

function findMatches(feed, skillName, version) {
  const advisories = Array.isArray(feed.advisories) ? feed.advisories : [];
  const matches = [];

  for (const advisory of advisories) {
    const affected = Array.isArray(advisory.affected) ? advisory.affected : [];
    if (affected.length === 0) continue;

    const matchedAffected = uniqueStrings(
      affected.filter((specifier) =>
        version
          ? affectedSpecifierMatches(specifier, skillName, version)
          : affectedSpecifierMatchesWithoutVersion(specifier, skillName),
      ),
    );

    if (matchedAffected.length > 0) {
      matches.push({ advisory, matchedAffected });
    }
  }

  return matches;
}

function printMatches(matches, skillName, version) {
  process.stdout.write("Advisory matches detected for requested install target.\n");
  process.stdout.write(`Target: ${skillName}${version ? `@${version}` : ""}\n`);

  for (const entry of matches) {
    const advisory = entry.advisory;
    const severity = String(advisory.severity ?? "unknown").toUpperCase();
    const advisoryId = advisory.id ?? "unknown-id";
    const title = advisory.title ?? "Untitled advisory";
    process.stdout.write(`- [${severity}] ${advisoryId}: ${title}\n`);
    process.stdout.write(`  matched: ${entry.matchedAffected.join(", ")}\n`);
    if (advisory.action) {
      process.stdout.write(`  action: ${advisory.action}\n`);
    }
  }
}

function runInstall(skillName, version) {
  const target = version ? `${skillName}@${version}` : skillName;
  process.stdout.write(`Install target: ${target}\n`);

  const result = runProcessSync("npx", ["clawhub@latest", "install", target], {
    stdio: "inherit",
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { feed, source } = await loadFeed();
  const matches = findMatches(feed, args.skill, args.version);
  const highRisk = matches.some((entry) => advisoryLooksHighRisk(entry.advisory));

  process.stdout.write(`Advisory source: ${source}\n`);

  if (!args.version) {
    process.stdout.write(
      "No --version provided. Conservatively matching any advisory for the requested skill name.\n",
    );
  }

  if (matches.length > 0) {
    printMatches(matches, args.skill, args.version);

    process.stdout.write("\n");
    process.stdout.write("Install request recognized as first confirmation.\n");
    process.stdout.write("Additional explicit confirmation is required with advisory context.\n");

    if (!args.confirmAdvisory) {
      process.stdout.write(
        "Re-run with --confirm-advisory to proceed after the user explicitly confirms.\n",
      );
      process.exit(EXIT_CONFIRM_REQUIRED);
    }
    process.stdout.write("Second confirmation provided via --confirm-advisory.\n");
  }

  if (args.dryRun) {
    process.stdout.write("Dry run only; install command was not executed.\n");
    return;
  }

  if (highRisk) {
    process.stdout.write(
      "High-risk advisory context acknowledged. Proceeding only because --confirm-advisory was provided.\n",
    );
  }

  runInstall(args.skill, args.version);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
