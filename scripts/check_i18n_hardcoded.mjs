#!/usr/bin/env node
/**
 * Hardcoded user-facing English strings → build failure.
 *
 * The fa-IR localization sweep (PR #112035) wired every user-facing string
 * on the web dashboard through the i18n catalogs (web/src/i18n/en.ts +
 * fa.ts, deep-merged per locale). This scanner keeps it that way, and now
 * gates the desktop renderer (apps/desktop/src, catalogs under
 * apps/desktop/src/i18n/) with the same idiom classes plus the desktop's
 * own call shapes (object-property copy in confirm()/notify() payloads).
 *
 * Per target:
 *   web      web/src              scripts/i18n-hardcoded-allowlist.json
 *   desktop  apps/desktop/src     scripts/i18n-hardcoded-allowlist-desktop.json
 *
 * Scope: renderer sources only. The electron main process
 * (apps/desktop/electron) has no i18n runtime today — native menus,
 * dialogs and tray copy there need their own mechanism before they can
 * gate; do not point this scanner at it until then.
 *
 * Usage:
 *   node scripts/check_i18n_hardcoded.mjs                    # gate all targets
 *   node scripts/check_i18n_hardcoded.mjs --target=desktop   # gate one target
 *   node scripts/check_i18n_hardcoded.mjs --update-allowlist # re-baseline
 *   node scripts/check_i18n_hardcoded.mjs --update-allowlist --target=desktop
 *
 * The allowlists record file:line-insensitive findings as
 * `file:::pattern-tag:::text` so unrelated edits above a line do not
 * silently re-arm old findings. Regenerate ONLY when a newly added string
 * is genuinely not user-facing (a proper noun, a technical token, a brand
 * name); prefer fixing by wiring the string through useI18n().
 */
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const update = process.argv.includes("--update-allowlist");
const targetArg = process.argv.find((a) => a.startsWith("--target="));
const onlyTarget = targetArg ? targetArg.slice("--target=".length) : null;

/** Lines that can never carry user-facing copy. */
const LINE_NOISE =
  /^\s*(\/\/|\/\*|\*|import\b|export\b.*\bfrom\b|console\.|className=|data-|key=|ref=|href=|src=|type\s+\w|interface\b|declare\b)/;

/**
 * Remove /* block comment *\/ portions of a line, carrying open/close state
 * across lines. Continuation lines inside a comment (which do not start
 * with "*" and so evade LINE_NOISE) hold prose but never UI copy.
 */
function makeCommentStripper() {
  let inBlock = false;
  return function strip(line) {
    let out = "";
    let rest = line;
    while (rest.length > 0) {
      if (inBlock) {
        const end = rest.indexOf("*/");
        if (end === -1) return out;
        inBlock = false;
        rest = rest.slice(end + 2);
      } else {
        const start = rest.indexOf("/*");
        if (start === -1) return out + rest;
        out += rest.slice(0, start);
        rest = rest.slice(start + 2);
        inBlock = true;
      }
    }
    return out;
  };
}

/** Literals that are identifiers/paths/URLs/commands rather than prose. */
const VALUE_NOISE = new RegExp(
  [
    "^https?://",
    "^/",
    "^\\.{0,2}/",
    "^-", // CLI flags: "-y", "--config"
    "^[a-z]+(-[a-z]+)*$",
    "^[A-Za-z][\\w.-]*\\.(tsx?|png|svg|css|md|json|woff2?)$",
    "^[a-z][\\w.-]*\\.(ts|js|py|sh)$",
    "^#[0-9a-fA-F]{3,8}$",
    "^[a-z-]+:",
    "^@", // npm scopes / decorators
    "^\\$\\{", // pure interpolation
  ].join("|"),
);

/**
 * Static residue of a template literal: the text outside ${...} jumps.
 * A template whose residue is only punctuation/whitespace around t.*
 * interpolations is translated; one with English residue is not.
 */
function templateResidue(literal) {
  return literal.replace(/^`|`$/g, "").replace(/\$\{[^}]*\}/g, "");
}

/**
 * Pattern classes, each with a tag. `userFacing` heuristics: the literal
 * must look like English prose (contains a space-separated word with a
 * lowercase letter, at least 2 words OR a known UI word) — this keeps
 * brand names ("Telegram"), technical tokens ("owner/repo"), and enum
 * values ("pre_tool_call") out of the report.
 */
const BASE_PATTERNS = [
  {
    tag: "toast",
    re: /showToast\((?!"\s*t\.)(`[^`]*`|"[^"]+")/,
    lit: (m) => m[1],
  },
  {
    tag: "confirm",
    re: /(?:window\.)?confirm\((`[^`]*`|"[^"]+")\)/,
    lit: (m) => m[1],
  },
  {
    tag: "attr",
    re:
      /\b(label|placeholder|title|tooltip|aria-label|emptyLabel|confirmLabel|cancelLabel|description|hint)="([^"]{3,})"/,
    lit: (m) => `"${m[2]}"`,
  },
  {
    tag: "attr-tpl",
    re: /\b(aria-label|title|label)=\{(`[^`]*\$\{[^`]*`)\}/,
    lit: (m) => m[2],
  },
  {
    tag: "jsx-text",
    // JSX text node: capitalized start, ≥9 more chars, allows acronyms
    // ("Add Server", "No MCP servers yet", "Browse catalog") — the 2026-09-27
    // MCP-page escape proved Title-Case/acronym-only text evades any pattern
    // that demands a lowercase second word. Real-word filter still applies in
    // looksUserFacing (some lowercase somewhere, ≥2 words).
    re: /(^|>)\s*([A-Z][A-Za-z0-9'’,.!?;:()\-\s]{9,})(<|$)/,
    lit: (m) => m[2],
  },
  {
    tag: "label-map",
    re: /\b(\w+):\s*"([A-Z][A-Za-z]*(?:\s+[a-z][A-Za-z]*)+)"/,
    lit: (m) => `"${m[2]}"`,
  },
];

/** Desktop-only: object-property copy inside confirm()/notify()/dialog payloads. */
const DESKTOP_PATTERNS = [
  {
    tag: "obj-copy",
    re:
      /\b(title|message|description|label|placeholder|hint|body|detail|confirmLabel|cancelLabel|tooltip):\s*("[^"]{3,}"|'[^']{3,}')/,
    lit: (m) => m[2],
  },
  {
    tag: "notify-arg",
    // notifyError(err, "Failed to save") / notify(target, "…") second-arg literal.
    re: /\bnotify(?:Error)?\([^,()]+,\s*(["'][^"']{3,}["'])/,
    lit: (m) => m[1],
  },
];

const TARGETS = [
  {
    name: "web",
    dir: join(root, "web", "src"),
    allowlistPath: join(root, "scripts", "i18n-hardcoded-allowlist.json"),
    patterns: BASE_PATTERNS,
  },
  {
    name: "desktop",
    dir: join(root, "apps", "desktop", "src"),
    allowlistPath: join(root, "scripts", "i18n-hardcoded-allowlist-desktop.json"),
    patterns: [...BASE_PATTERNS, ...DESKTOP_PATTERNS],
  },
];

function looksUserFacing(literal) {
  // Templates are judged on their static residue: `${t.a}: ${err(e)}` has
  // none (translated), `${n} pastes${redacted}` has English residue.
  const s = (
    literal.startsWith("`") ? templateResidue(literal) : literal
  ).replace(/^[`"']|[`"']$/g, "");
  if (VALUE_NOISE.test(s)) return false;
  // Code, not prose: JS expressions ("Math.max(0, prev - x)"), type
  // signatures ("Icon: React.ComponentType"), comma-separated identifier
  // lists (multi-import JSX re-exports, barrel re-exports with "as",
  // table-column generics like "Cell: PrimaryCell").
  if (/^\s*(Math\.|JSON\.|window\.|document\.)/.test(s)) return false;
  if (/\bReact\.\w/.test(s)) return false;
  // Ternaries / type annotations ("x ? a : b", "Icon?: ComponentType").
  if (/\w\?\s*:/.test(s)) return false;
  // Expressions that end in a call: String(x).padStart(2, '0'),
  // Object.assign(a, b) — prose never ends with a function invocation.
  if (/\w+\(.*\)\s*$/.test(s)) return false;
  const tokens = s.split(/,\s*/).filter(Boolean);
  const identifierish = (tok) =>
    /^[A-Za-z_$][\w$]*$/.test(tok.trim()) ||
    /^[A-Za-z_$][\w$]*(\s+as\s+[A-Za-z_$][\w$]*)+$/.test(tok.trim()) ||
    /^[A-Za-z_$][\w$]*:\s*[A-Za-z_$][\w$.]*$/.test(tok.trim());
  if (
    tokens.length > 0 &&
    tokens.every(
      (tok) =>
        identifierish(tok) ||
        /^[A-Z][A-Za-z0-9]*$/.test(tok.trim()) ||
        /^[A-Za-z_$][\w$]*\s*\(/.test(tok.trim()),
    )
  ) {
    return false;
  }
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length < 2) return false;
  return words.some((w) => /[a-z]/.test(w)) && words.some((w) => /[A-Za-z]/.test(w));
}

/** Walk a target's source tree recursively (catalogs & tests excluded). */
function walk(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name),
  )) {
    if (entry.isDirectory()) {
      // "i18n" = catalogs; "test"/"__tests__" = test helpers & suites.
      if (entry.name === "i18n" || entry.name === "test" || entry.name === "__tests__")
        continue;
      walk(join(dir, entry.name), out);
    } else if (/\.(tsx?|mjs)$/.test(entry.name) && !/\.test\./.test(entry.name)) {
      out.push(join(dir, entry.name));
    }
  }
  return out;
}

function scanTarget(target) {
  const findings = [];
  for (const file of walk(target.dir)) {
    const rel = file.slice(root.length + 1).replace(/\\/g, "/");
    const lines = readFileSync(file, "utf8").split(/\r?\n/);
    const stripComments = makeCommentStripper();
    for (let i = 0; i < lines.length; i++) {
      const line = stripComments(lines[i]);
      if (!line || LINE_NOISE.test(line)) continue;
      for (const p of target.patterns) {
        const m = line.match(p.re);
        if (!m) continue;
        const literal = p.lit(m);
        if (!literal || !looksUserFacing(literal)) continue;
        findings.push({ file: rel, tag: p.tag, line: i + 1, text: literal });
      }
    }
  }
  return findings;
}

function loadAllowlist(path) {
  if (!existsSync(path)) return new Set();
  return new Set(JSON.parse(readFileSync(path, "utf8")));
}

const selected = onlyTarget
  ? TARGETS.filter((t) => t.name === onlyTarget)
  : TARGETS;
if (selected.length === 0) {
  console.error(`unknown target: ${onlyTarget} (known: ${TARGETS.map((t) => t.name).join(", ")})`);
  process.exit(2);
}

let novelTotal = 0;
const perTarget = [];
for (const target of selected) {
  const findings = scanTarget(target);
  const allowlist = loadAllowlist(target.allowlistPath);
  const keyOf = (f) => `${f.file}:::${f.tag}:::${f.text}`;
  const novel = findings.filter((f) => !allowlist.has(keyOf(f)));
  perTarget.push({ target, findings, allowlist, novel, keyOf });
  novelTotal += novel.length;
}

if (update) {
  for (const { target, findings } of perTarget) {
    const keep = [...new Set(findings.map((f) => `${f.file}:::${f.tag}:::${f.text}`))].sort();
    writeFileSync(target.allowlistPath, JSON.stringify(keep, null, 2) + "\n");
    console.log(
      `allowlist updated [${target.name}]: ${keep.length} entr(y|ies) -> ${target.allowlistPath}`,
    );
  }
  process.exit(0);
}

if (novelTotal === 0) {
  const detail = perTarget
    .map(
      ({ target, findings, allowlist }) =>
        `${target.name}: ${findings.length} allowlisted, ${allowlist.size} allowlist entries`,
    )
    .join("; ");
  console.log(`ok: no hardcoded user-facing strings (${detail})`);
  process.exit(0);
}

console.error(
  `FAIL: ${novelTotal} hardcoded user-facing string(s).\n` +
    `Wire them through the i18n catalogs of the failing target\n` +
    `(web: web/src/i18n/, desktop: apps/desktop/src/i18n/).\n` +
    `If a finding is genuinely not user-facing, re-baseline with:\n` +
    `  node scripts/check_i18n_hardcoded.mjs --update-allowlist [--target=web|desktop]\n`,
);
for (const { target, novel } of perTarget) {
  const prefix = target.name === "web" ? "" : `[${target.name}] `;
  for (const f of novel) {
    console.error(`  ${prefix}${f.file}:${f.line} [${f.tag}] ${f.text}`);
  }
}
process.exit(1);
