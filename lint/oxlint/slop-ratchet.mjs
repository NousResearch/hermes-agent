#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const MAX_BUFFER = 256 * 1024 * 1024;
const SCRIPT_NAME = "lint:slop";

function lexicalCompare(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function fail(message, details = "") {
  console.error(message);
  if (details.trim()) {
    console.error(details.trim());
  }
  process.exit(2);
}

function stderrTail(result) {
  const stderr = result.stderr?.toString() ?? "";
  return stderr.slice(-4000);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    encoding: options.encoding ?? "utf8",
    env: process.env,
    maxBuffer: MAX_BUFFER,
  });

  if (result.error) {
    fail(`${SCRIPT_NAME}: could not run ${command}: ${result.error.message}`, stderrTail(result));
  }

  return result;
}

function repoRoot() {
  const result = run("git", ["rev-parse", "--show-toplevel"]);
  if (result.status !== 0) {
    fail(`${SCRIPT_NAME}: could not find the repository root`, stderrTail(result));
  }
  return result.stdout.trim();
}

const root = repoRoot();
const BASELINE_REPO_PATH = "lint/oxlint/slop-baseline.json";
const baselinePath = path.join(root, BASELINE_REPO_PATH);
const configPath = "lint/oxlint/oxlint.config.ts";
const localOxlint = path.join(root, "lint/oxlint/node_modules/.bin/oxlint");
const oxlint = existsSync(localOxlint) ? localOxlint : "oxlint";

function normalizedFilename(filename) {
  const relative = path.isAbsolute(filename) ? path.relative(root, filename) : filename;
  return relative.split(path.sep).join("/").replace(/^\.\//, "");
}

function lint(files = []) {
  const result = run(oxlint, ["-c", configPath, "--ignore-path", ".slopignore", "--format", "json", ...files], { cwd: root });
  let report;

  try {
    report = JSON.parse(result.stdout);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    fail(`${SCRIPT_NAME}: oxlint returned unparsable JSON: ${reason}`, stderrTail(result));
  }

  if (!Array.isArray(report.diagnostics)) {
    fail(`${SCRIPT_NAME}: oxlint JSON did not contain a diagnostics array`, stderrTail(result));
  }

  return report.diagnostics.map((diagnostic) => ({
    ...diagnostic,
    filename: normalizedFilename(diagnostic.filename),
  }));
}

function countsFrom(diagnostics) {
  const counts = new Map();

  for (const diagnostic of diagnostics) {
    let rules = counts.get(diagnostic.filename);
    if (!rules) {
      rules = new Map();
      counts.set(diagnostic.filename, rules);
    }
    rules.set(diagnostic.code, (rules.get(diagnostic.code) ?? 0) + 1);
  }

  return counts;
}

function sortedObject(counts) {
  return Object.fromEntries(
    [...counts.entries()]
      .sort(([left], [right]) => lexicalCompare(left, right))
      .map(([filename, rules]) => [
        filename,
        Object.fromEntries([...rules.entries()].sort(([left], [right]) => lexicalCompare(left, right))),
      ]),
  );
}

function parseBaseline(text, sourceLabel) {
  let parsed;

  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    fail(`${SCRIPT_NAME}: could not parse ${sourceLabel}: ${reason}`);
  }

  if (parsed === null || Array.isArray(parsed) || Object.getPrototypeOf(parsed) !== Object.prototype) {
    fail(`${SCRIPT_NAME}: ${sourceLabel} must contain an object`);
  }

  return parsed;
}

/** The working-tree baseline: what `baseline` rewrites and compares against. */
function readBaseline() {
  let text;
  try {
    text = readFileSync(baselinePath, "utf8");
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    fail(`${SCRIPT_NAME}: could not read lint/oxlint/slop-baseline.json: ${reason}`);
  }
  return parseBaseline(text, "lint/oxlint/slop-baseline.json");
}

/**
 * The baseline as committed at `ref` (the diff's merge base), NOT the one in
 * the checkout: a PR that could edit the baseline it is judged against could
 * raise its own allowance in the same change. Returns null when the file does
 * not exist at that ref (a branch cut before the ratchet landed).
 */
function readBaselineAt(ref) {
  const result = run("git", ["show", `${ref}:${BASELINE_REPO_PATH}`], { cwd: root });
  if (result.status !== 0) {
    return null;
  }
  return parseBaseline(result.stdout, `${BASELINE_REPO_PATH} at ${ref.slice(0, 12)}`);
}

function countAt(baseline, filename, rule) {
  const count = baseline[filename]?.[rule];
  return Number.isInteger(count) && count >= 0 ? count : 0;
}

function changesFrom(previous, current) {
  const changes = [];
  const filenames = new Set([...Object.keys(previous), ...Object.keys(current)]);

  for (const filename of [...filenames].sort(lexicalCompare)) {
    const previousRules = previous[filename] ?? {};
    const currentRules = current[filename] ?? {};
    const rules = new Set([...Object.keys(previousRules), ...Object.keys(currentRules)]);

    for (const rule of [...rules].sort(lexicalCompare)) {
      const before = countAt(previous, filename, rule);
      const after = countAt(current, filename, rule);
      if (before !== after) {
        changes.push({ filename, rule, before, after });
      }
    }
  }

  return changes;
}

function printChanges(label, changes) {
  if (changes.length === 0) return;
  console.log(`${label}:`);
  for (const { filename, rule, before, after } of changes) {
    const delta = after - before;
    console.log(`  ${filename} ${rule}: ${before} -> ${after} (${delta > 0 ? "+" : ""}${delta})`);
  }
}

function baselineCommand(args) {
  if (args.some((arg) => arg !== "--allow-increase")) {
    fail("usage: node lint/oxlint/slop-ratchet.mjs baseline [--allow-increase]");
  }

  const allowIncrease = args.includes("--allow-increase");
  const diagnostics = lint();
  const current = sortedObject(countsFrom(diagnostics));

  if (existsSync(baselinePath)) {
    const previous = readBaseline();
    const changes = changesFrom(previous, current);
    const increases = changes.filter(({ after, before }) => after > before);
    const lowered = changes.filter(({ after, before }) => after < before);

    if (increases.length > 0 && !allowIncrease) {
      console.error(`${SCRIPT_NAME}:baseline: refusing to increase the baseline`);
      for (const { filename, rule, before, after } of increases) {
        console.error(`  ${filename} ${rule}: ${before} -> ${after} (+${after - before})`);
      }
      console.error("rerun with --allow-increase only when the increase is intentional");
      process.exit(2);
    }

    printChanges("lowered", lowered);
    if (allowIncrease) {
      printChanges("increased", increases);
    }
  }

  writeFileSync(baselinePath, `${JSON.stringify(current, null, 2)}\n`);
  console.log(
    `${SCRIPT_NAME}:baseline: wrote lint/oxlint/slop-baseline.json (${diagnostics.length} findings in ${Object.keys(current).length} files)`,
  );
}

function changedFiles(base) {
  const mergeBaseResult = run("git", ["merge-base", base, "HEAD"]);
  if (mergeBaseResult.status !== 0) {
    fail(`${SCRIPT_NAME}:diff: could not compute merge-base with ${base}`, stderrTail(mergeBaseResult));
  }
  const mergeBase = mergeBaseResult.stdout.trim();

  // Deliberately omit rename detection, matching the old runner. A rename is
  // treated as an added file; a pure `git mv` should regenerate the baseline.
  const diffResult = run(
    "git",
    [
      "diff",
      "-z",
      "--name-only",
      "--diff-filter=d",
      mergeBase,
      "--",
      "*.ts",
      "*.tsx",
      "*.js",
      "*.jsx",
      "*.mjs",
      "*.cjs",
    ],
    { encoding: "buffer" },
  );
  if (diffResult.status !== 0) {
    fail(`${SCRIPT_NAME}:diff: git diff failed`, stderrTail(diffResult));
  }

  const changed = diffResult.stdout.toString().split("\0").filter(Boolean);
  // oxlint lints explicitly named files even when an ignore rule covers them,
  // so apply .slopignore here too: a PR that touches a vendored or generated
  // file must not be judged on it. `git check-ignore` speaks the same
  // gitignore dialect .slopignore is written in.
  const files = dropIgnored(changed).map((filename) => `./${filename}`);
  return { files, mergeBase };
}

function dropIgnored(files) {
  if (files.length === 0) return files;
  // check-ignore reads the repo's .gitignore chain; point core.excludesFile at
  // the lane's list so .slopignore is consulted too. Exit 1 = nothing matched.
  const result = spawnSync(
    "git",
    ["-c", `core.excludesFile=${path.join(root, ".slopignore")}`, "check-ignore", "--no-index", "-z", "--stdin"],
    { cwd: root, input: files.join("\0"), encoding: "utf8", maxBuffer: MAX_BUFFER },
  );
  if (result.status !== 0 && result.status !== 1) {
    fail(`${SCRIPT_NAME}:diff: git check-ignore failed`, stderrTail(result));
  }
  const ignored = new Set(result.stdout.split("\0").filter(Boolean));
  return files.filter((filename) => !ignored.has(filename));
}

function displayRule(rule) {
  const match = /^anti-slop\((.+)\)$/.exec(rule);
  return match?.[1] ?? rule;
}

function locationOf(diagnostic) {
  const span = diagnostic.labels?.[0]?.span;
  return `${span?.line ?? 0}:${span?.column ?? 0}`;
}

function diffCommand(args) {
  if (args.length > 0) {
    fail("usage: node lint/oxlint/slop-ratchet.mjs diff");
  }

  const base = process.env.SLOP_BASE ?? "origin/main";
  const { files, mergeBase } = changedFiles(base);
  if (files.length === 0) {
    console.log(`${SCRIPT_NAME}:diff: no lintable files changed since ${mergeBase}`);
    return;
  }

  // Judge against the merge base's baseline so the PR cannot authorise its
  // own regressions by editing the file. No baseline there (branch predates
  // the ratchet) means nothing is allowed yet: everything reports as new,
  // which is the pre-ratchet behaviour for those files.
  const trustedBaseline = readBaselineAt(mergeBase);
  if (trustedBaseline === null) {
    console.log(
      `${SCRIPT_NAME}:diff: no ${BASELINE_REPO_PATH} at merge base ${mergeBase.slice(0, 12)}; treating every finding as net-new (rebase onto main once the ratchet has landed)`,
    );
  }
  const baseline = trustedBaseline ?? {};
  const diagnostics = lint(files);
  const counts = countsFrom(diagnostics);
  const findingsByPair = new Map();

  for (const diagnostic of diagnostics) {
    const key = `${diagnostic.filename}\0${diagnostic.code}`;
    const findings = findingsByPair.get(key) ?? [];
    findings.push(diagnostic);
    findingsByPair.set(key, findings);
  }

  const netNew = [];
  let baselinedFindings = 0;
  const baselinedFiles = new Set();
  const loweredFiles = new Set();
  const touchedFiles = files.map((filename) => normalizedFilename(filename));

  for (const filename of touchedFiles) {
    const currentRules = counts.get(filename) ?? new Map();
    const rules = new Set([...Object.keys(baseline[filename] ?? {}), ...currentRules.keys()]);

    for (const rule of [...rules].sort(lexicalCompare)) {
      const current = currentRules.get(rule) ?? 0;
      const allowed = countAt(baseline, filename, rule);
      if (current > allowed) {
        netNew.push({ filename, rule, current, allowed });
      } else {
        baselinedFindings += current;
        if (current > 0) baselinedFiles.add(filename);
        if (current < allowed) loweredFiles.add(filename);
      }
    }
  }

  if (netNew.length > 0) {
    console.log(
      `${SCRIPT_NAME}:diff: lines listed are all hits of a rule whose count grew; the new ones are among them`,
    );
    let previousFile;
    for (const finding of netNew) {
      if (finding.filename !== previousFile) {
        if (previousFile !== undefined) console.log();
        console.log(finding.filename);
        previousFile = finding.filename;
      }
      console.log(
        `${displayRule(finding.rule)}: ${finding.current} (baseline ${finding.allowed}, +${finding.current - finding.allowed})`,
      );
      const pairFindings = findingsByPair.get(`${finding.filename}\0${finding.rule}`) ?? [];
      pairFindings.sort((left, right) => locationOf(left).localeCompare(locationOf(right), undefined, { numeric: true }));
      for (const diagnostic of pairFindings) {
        console.log(`  ${locationOf(diagnostic)}  ${diagnostic.message.replaceAll("\n", " ")}`);
      }
    }
  }

  if (baselinedFindings > 0) {
    console.log(
      `baselined: ${baselinedFindings} findings in ${baselinedFiles.size} touched files, unchanged or lower`,
    );
  }
  for (const filename of [...loweredFiles].sort(lexicalCompare)) {
    console.log(`tip: ${filename} is under its baseline; run npm run lint:slop:baseline to lock it in`);
  }

  if (netNew.length > 0) {
    process.exit(1);
  }

  console.log(`${SCRIPT_NAME}:diff: no net-new findings`);
}

const [command, ...args] = process.argv.slice(2);

if (command === "baseline") {
  baselineCommand(args);
} else if (command === "diff") {
  diffCommand(args);
} else {
  fail("usage: node lint/oxlint/slop-ratchet.mjs <baseline|diff>");
}
