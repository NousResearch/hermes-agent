#!/usr/bin/env node
/**
 * Preflight. Run this BEFORE the interview, not after the first failure.
 *
 *   node scripts/doctor.mjs            check everything
 *   node scripts/doctor.mjs --probe    also spend one API call to read the balance
 *
 * Every check that can fail deep inside a build with a misleading message is
 * checked here with an honest one. The three that actually bite:
 *
 *   - a STRIPPED ffmpeg on PATH. It carries ~50 filters and silently lacks
 *     scale, fps, psnr and the webp muxer, then fails with "No option name
 *     near ..." or "Unable to choose an output format", both of which read as
 *     a mistake in your command rather than a missing feature.
 *   - no KIE_AI_API_KEY, which only matters if you are generating assets.
 *   - playwright-core resolving from the wrong directory. It is required from
 *     the BUILD folder, not from the skill.
 */

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { paths } from "./workspace.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const rows = [];
const add = (sev, name, ok, detail, fix) => rows.push({ sev, name, ok, detail, fix });

const run = (cmd, args) => {
  try {
    return execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
  } catch {
    return null;
  }
};

// ---------------------------------------------------------------- node ----
const major = Number(process.versions.node.split(".")[0]);
add("required", "node", major >= 18, `v${process.versions.node}`,
  "Install Node 18 or newer.");

// -------------------------------------------------------------- ffmpeg ----
function globWinGet() {
  const home = process.env.USERPROFILE || process.env.HOME || "";
  const base = path.join(home, "AppData/Local/Microsoft/WinGet/Packages");
  if (!fs.existsSync(base)) return [];
  const out = [];
  for (const d of fs.readdirSync(base)) {
    if (!/^Gyan\.FFmpeg/i.test(d)) continue;
    const inner = path.join(base, d);
    for (const e of fs.readdirSync(inner)) {
      const p = path.join(inner, e, "bin/ffmpeg.exe");
      if (fs.existsSync(p)) out.push(p);
    }
  }
  return out;
}

const candidates = [
  process.env.SCROLLCRAFT_FFMPEG,
  "ffmpeg",
  ...globWinGet(),
  "/usr/local/bin/ffmpeg",
  "/opt/homebrew/bin/ffmpeg",
  "/usr/bin/ffmpeg",
  "/snap/bin/ffmpeg",
].filter(Boolean);

let ffmpeg = null, filterCount = 0;
for (const c of candidates) {
  const out = run(c, ["-hide_banner", "-filters"]);
  if (!out) continue;
  const n = out.split("\n").length;
  if (n > filterCount) { filterCount = n; ffmpeg = c; }
  if (n > 200) break;
}
add("required", "ffmpeg (full build)", filterCount > 200,
  ffmpeg ? `${ffmpeg}  (${filterCount} filters)` : "not found",
  "A stripped ffmpeg lacks scale/fps/psnr and the webp muxer. Install a full build (Windows: winget install Gyan.FFmpeg) or set SCROLLCRAFT_FFMPEG to one.");

if (ffmpeg && filterCount > 200) {
  const enc = run(ffmpeg, ["-hide_banner", "-encoders"]) || "";
  add("optional", "  └ libwebp encoder", /libwebp/.test(enc),
    /libwebp/.test(enc) ? "present" : "missing",
    "Posters fall back to JPEG. Not fatal, just heavier.");
}

// ---------------------------------------------------------- playwright ----
let pw = false, pwWhere = "";
try {
  createRequire(path.join(process.cwd(), "package.json"))("playwright-core");
  pw = true; pwWhere = "resolves from cwd";
} catch {
  try {
    createRequire(path.join(HERE, "package.json"))("playwright-core");
    pw = true; pwWhere = "resolves from the skill, but NOT from cwd";
  } catch { pwWhere = "not installed"; }
}
add("verify", "playwright-core", pw, pwWhere,
  "Run `npm i playwright-core` inside the build folder. Only needed for the verification pass.");

const chrome = [
  process.env.SCROLLCRAFT_CHROME,
  // Windows
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  // macOS
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  // Linux
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
].find((p) => p && fs.existsSync(p));
add("verify", "Chrome", Boolean(chrome), chrome || "not found",
  "Install Chrome, or set SCROLLCRAFT_CHROME to an executable.");

// ------------------------------------------------------------- kie key ----
function findKey() {
  if (process.env.KIE_AI_API_KEY) return "env";
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    const p = path.join(dir, ".env");
    if (fs.existsSync(p) && /^\s*KIE_AI_API_KEY\s*=\s*\S+/m.test(fs.readFileSync(p, "utf8"))) return p;
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}
const keyWhere = findKey();
add("optional", "KIE_AI_API_KEY", Boolean(keyWhere), keyWhere || "not set",
  "Only needed to GENERATE imagery. Building from your own photos and footage needs no key and no spend. Copy .env.example to .env to set one.");

// ----------------------------------------------------------- workspace ----
let ws = null;
try {
  ws = paths();
  add("required", "workspace", true, `${ws.workspace}\n      via ${ws.via}`, "");
  add("optional", "  └ registry", fs.existsSync(ws.fingerprints),
    fs.existsSync(ws.fingerprints) ? "present" : "not created yet",
    "Run `node scripts/workspace.mjs --ensure` to create it.");
} catch (e) {
  add("required", "workspace", false, e.message, "Fix or delete the offending .scrollcraft.json.");
}

// -------------------------------------------------------------- report ----
const mark = (r) => (r.ok ? "\u001b[32m ok \u001b[0m" : r.sev === "required" ? "\u001b[31mFAIL\u001b[0m" : "\u001b[33mwarn\u001b[0m");
console.log("\nscrollcraft preflight\n");
for (const r of rows) {
  console.log(` [${mark(r)}] ${r.name.padEnd(22)} ${r.detail}`);
  if (!r.ok && r.fix) console.log(`        ${"\u001b[2m"}${r.fix}${"\u001b[0m"}`);
}

const hardFails = rows.filter((r) => !r.ok && r.sev === "required");
const softFails = rows.filter((r) => !r.ok && r.sev !== "required");
console.log("");
if (hardFails.length) {
  console.log(`\u001b[31m${hardFails.length} required check(s) failed. Fix these before building.\u001b[0m\n`);
  process.exit(1);
}
console.log(softFails.length
  ? `\u001b[33mReady, with ${softFails.length} optional item(s) missing (see above).\u001b[0m\n`
  : "\u001b[32mReady.\u001b[0m\n");

if (process.argv.includes("--probe") && keyWhere) {
  const { execFileSync: x } = await import("node:child_process");
  try {
    console.log("credit: " + x(process.execPath, [path.join(HERE, "kie.mjs"), "probe"], { encoding: "utf8" }).trim().replace(/^credit:\s*/, ""));
  } catch (e) { console.log("balance probe failed: " + e.message); }
}
