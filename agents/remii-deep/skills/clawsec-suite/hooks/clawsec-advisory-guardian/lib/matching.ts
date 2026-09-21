import fs from "node:fs/promises";
import path from "node:path";
import { isObject, normalizeSkillName, uniqueStrings } from "./utils.mjs";
import { advisoryAppliesToOpenclaw } from "./advisory_scope.mjs";
import { versionMatches } from "./version.mjs";
import { parseAffectedSpecifier } from "./feed.mjs";
import type { Advisory, FeedPayload, InstalledSkill, AdvisoryMatch } from "./types.ts";

export async function discoverInstalledSkills(installRoot: string): Promise<InstalledSkill[]> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(installRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  const skills: InstalledSkill[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;

    const fallbackName = entry.name;
    const skillDir = path.join(installRoot, entry.name);
    const skillJsonPath = path.join(skillDir, "skill.json");

    let skillName = fallbackName;
    let version: string | null = "unknown";

    try {
      const rawSkillJson = await fs.readFile(skillJsonPath, "utf8");
      const parsedSkillJson = JSON.parse(rawSkillJson);
      if (isObject(parsedSkillJson) && typeof parsedSkillJson.name === "string" && parsedSkillJson.name.trim()) {
        skillName = parsedSkillJson.name.trim();
      }
      if (
        isObject(parsedSkillJson) &&
        typeof parsedSkillJson.version === "string" &&
        parsedSkillJson.version.trim()
      ) {
        version = parsedSkillJson.version.trim();
      }
    } catch {
      // best-effort scan: keep fallback directory name when skill.json is missing or invalid
    }

    skills.push({ name: skillName, dirName: entry.name, version });
  }

  return skills;
}

export function affectedSpecifierMatchesSkill(rawSpecifier: string, skill: InstalledSkill): boolean {
  const parsed = parseAffectedSpecifier(rawSpecifier);
  if (!parsed) return false;

  const specName = normalizeSkillName(parsed.name);
  const skillName = normalizeSkillName(skill.name);
  if (specName !== skillName) return false;

  return versionMatches(skill.version, parsed.versionSpec);
}

export function advisoryMatchesSkill(advisory: Advisory, skill: InstalledSkill): string[] {
  const affected = Array.isArray(advisory.affected) ? advisory.affected : [];
  const matches = affected.filter((specifier) => affectedSpecifierMatchesSkill(specifier, skill));
  return uniqueStrings(matches);
}

export function findMatches(feed: FeedPayload, installedSkills: InstalledSkill[]): AdvisoryMatch[] {
  const matches: AdvisoryMatch[] = [];

  for (const advisory of feed.advisories) {
    if (!advisoryAppliesToOpenclaw(advisory)) continue;

    const affected = Array.isArray(advisory.affected) ? advisory.affected : [];
    if (affected.length === 0) continue;

    for (const skill of installedSkills) {
      const matchedAffected = advisoryMatchesSkill(advisory, skill);
      if (matchedAffected.length === 0) continue;
      matches.push({ advisory, skill, matchedAffected });
    }
  }

  return matches;
}

export function matchKey(match: AdvisoryMatch): string {
  const normalizedSkillName = normalizeSkillName(match.skill.name);
  const version = match.skill.version ?? "unknown";
  const advisoryId =
    match.advisory.id ??
    `${match.advisory.title ?? "untitled"}::${match.advisory.published ?? match.advisory.updated ?? "unknown-ts"}`;
  return `${advisoryId}::${normalizedSkillName}@${version}`;
}

export function looksMalicious(advisory: Advisory): boolean {
  const type = String(advisory.type ?? "").toLowerCase();
  const combined = `${advisory.title ?? ""} ${advisory.description ?? ""} ${advisory.action ?? ""}`.toLowerCase();

  if (type === "malicious_skill" || type === "malicious_plugin") return true;
  if (/\b(malicious|exfiltrat(e|ion)|backdoor|trojan|credential theft|stealer)\b/.test(combined)) return true;
  return false;
}

export function looksRemovalRecommended(advisory: Advisory): boolean {
  const combined = `${advisory.action ?? ""} ${advisory.title ?? ""} ${advisory.description ?? ""}`.toLowerCase();
  return /\b(remove|uninstall|delete|disable|do not use|quarantine)\b/.test(combined);
}

export function buildAlertMessage(matches: AdvisoryMatch[], installRoot: string): string {
  const lines: string[] = [];
  lines.push("CLAWSEC ALERT: advisory feed matches installed skill(s).");
  lines.push("Affected skill advisories:");

  const MAX_LISTED = 8;
  for (const match of matches.slice(0, MAX_LISTED)) {
    const severity = String(match.advisory.severity ?? "unknown").toUpperCase();
    const advisoryId = match.advisory.id ?? "unknown-id";
    const version = match.skill.version ?? "unknown";
    const matched = match.matchedAffected.join(", ");
    lines.push(
      `- [${severity}] ${advisoryId} -> ${match.skill.name}@${version}` +
        (matched ? ` (matched: ${matched})` : ""),
    );
    if (match.advisory.action) {
      lines.push(`  Action: ${match.advisory.action}`);
    }
  }

  if (matches.length > MAX_LISTED) {
    lines.push(`- ... ${matches.length - MAX_LISTED} additional match(es) not shown`);
  }

  const removalMatches = matches.filter((entry) => looksMalicious(entry.advisory) || looksRemovalRecommended(entry.advisory));
  if (removalMatches.length > 0) {
    const impactedSkills = uniqueStrings(removalMatches.map((entry) => entry.skill.name));
    const impactedDirs = uniqueStrings(removalMatches.map((entry) => entry.skill.dirName));
    lines.push("");
    lines.push("Recommendation: one or more matches indicate potentially malicious or unsafe skills.");
    lines.push("Best practice: remove or disable affected skills only after explicit user approval.");
    lines.push(
      "Double-confirmation policy: treat the install request as first intent and require an additional explicit confirmation with this advisory context.",
    );
    lines.push(`Approval needed: ask the user to approve removal of: ${impactedSkills.join(", ")}.`);
    lines.push("Candidate removal paths:");
    for (const dir of impactedDirs) {
      lines.push(`- ${path.join(installRoot, dir)}`);
    }
  } else {
    lines.push("");
    lines.push("Recommendation: review advisories and update/remove affected skills as directed.");
  }

  return lines.join("\n");
}
