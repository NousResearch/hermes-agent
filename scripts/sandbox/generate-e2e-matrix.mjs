#!/usr/bin/env node
// @ts-check
/**
 * Expand the install/update support matrix into concrete E2E combinations.
 *
 * One source of truth for every {os, install-method, update-method} pair a
 * user could be on. This file only DECLARES and EXPANDS: it knows nothing
 * about which combinations CI can drive. Every combination is dispatched to
 * its OS's run workflow, and THAT workflow natively skips the method pairs
 * its driver cannot run yet -- capability knowledge lives next to each
 * driver (install-e2e-run.yml for linux, install-e2e-macos-run.yml,
 * and install-e2e-windows-run.yml). Correctness here is enforced by the type
 * unions below (checked via `tsc --checkJs`), not by runtime validation --
 * anything the types can't catch is self-evident on the next CI run.
 *
 * Used by .github/workflows/install-e2e.yml, which runs it with the picked
 * release tags (annotated at pick time with what each tag's tree ships):
 *
 *   node scripts/sandbox/generate-e2e-matrix.mjs \
 *     --tags '[{"ref":"v2026.8.3","desktop":true}]'
 *
 * Prints JSON: { linux: {include:[...]}, windows: {include:[...]},
 * macos: {include:[...]} } -- every entry is {name, install_method,
 * update_method, install_ref, update_ref, tag_has_desktop} (the tag
 * annotation lets a run workflow natively skip desktop-surface legs from
 * releases that predate the desktop app).
 */

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';

/**
 * The closed method/version vocabulary. Workflows key off these exact
 * strings, so they are types, not conventions.
 *
 * @typedef {'latest'} InstallerVersion
 *   The artifact published on the website right now -- Hermes-Setup.exe has
 *   no versioned archive yet. Widen this union when one exists.
 * @typedef {'installer-script' | 'installer-script+desktop' | 'desktop-installer' | 'packaged-app'} InstallMethod
 *   installer-script is the platform's one-liner (curl | bash on
 *   linux/macos, irm | iex on windows); installer-script+desktop is the
 *   same one-liner with its desktop stage opted in (--include-desktop /
 *   -IncludeDesktop), which also builds the desktop app -- on windows it
 *   registers Start Menu / Desktop shortcuts too, on linux/macos it
 *   builds into the checkout without registering an OS entry point;
 *   packaged-app installs a signed bundle. Its native in-app update pair
 *   is declared separately, not crossed with source-checkout update methods.
 * @typedef {InstallMethod | 'hermes-update' | 'open-app-update' | 'hermes-desktop-app-update'} UpdateMethod
 *   Every install method doubles as an update method (re-run it over the
 *   existing install), plus the updater CLI and the two app-update
 *   variants. The variants differ by launch surface: open-app-update
 *   starts the app from the OS entry point the install registered
 *   (Start Menu / Desktop shortcuts) -- today only the windows desktop
 *   installer's stage registers one (install.sh --include-desktop
 *   builds the app but registers nothing), so these legs pair with a
 *   desktop-installer install; hermes-desktop-app-update starts the app
 *   via `hermes desktop`, which every install method provides on every
 *   OS that ships the desktop app. Both then update through the app's
 *   own Update button.
 * @typedef {'linux' | 'windows' | 'macos'} Os
 *
 * @typedef {{method: InstallMethod, versions?: InstallerVersion[]}} InstallEntry
 * @typedef {{method: UpdateMethod, versions?: InstallerVersion[]}} UpdateEntry
 *   `versions` expands the entry into one combination per version
 *   ("desktop-installer@latest").
 * @typedef {{install: InstallEntry[], update: UpdateEntry[], secondUpdate?: never[]}} OsSpec
 *   secondUpdate (install -> update -> update again) is a real future axis
 *   -- the updater that RESULTS from an update must itself update -- typed
 *   `never[]` so declaring one is a type error until a leg implements it.
 *
 * @typedef {{ref: string, desktop: boolean}} TagAnnotation
 *   A picked release tag plus what its own tree ships (annotated by
 *   pick-releases in install-e2e.yml).
 *
 * @typedef {'HEAD' | 'NEXT'} UpdateTarget
 *   HEAD is the commit under test. NEXT is not a git ref: the drivers mint
 *   it as a synthetic child of the install ref, so a leg starting AT HEAD
 *   still has an update to take.
 *
 * @typedef {{from: string, to: UpdateTarget, desktop: boolean}} Start
 *   One column of the plan: install `from`, update to `to`.
 *
 * @typedef {{name: string, leg_id: string, install_method: string, update_method: string,
 *            install_ref: string, update_ref: UpdateTarget, tag_has_desktop?: boolean}} MatrixEntry
 */

/**
 * Artifact-safe key for one leg: the matrix name with every character
 * outside [A-Za-z0-9._-] collapsed to '-'. Reconstructed by the results
 * renderer from the parsed job name (same formula, same bytes), and used
 * by every run workflow to name its logs/player artifacts, so the report
 * job can map a concluded leg back to its artifacts without GitHub
 * linking jobs to artifacts.
 * @param {string} name
 * @returns {string}
 */
export function legId(name) {
  return name.replace(/[^A-Za-z0-9._-]+/g, '-');
}

/** One pinned package transition; source-release sampling does not apply.
 * @param {'windows' | 'macos'} os
 * @param {string} oldTag
 * @returns {{include: MatrixEntry[]}}
 */
export function bundledMatrix(os, oldTag) {
  const name = `${os}: packaged-app -> open-app-update (${oldTag} -> HEAD)`;
  return { include: [{ name, leg_id: legId(name), install_method: 'packaged-app',
    update_method: 'open-app-update', install_ref: oldTag, update_ref: 'HEAD', tag_has_desktop: true }] };
}

/**
 * The starting points every combination runs from: each sampled release
 * tag updating to HEAD, then HEAD itself updating to NEXT. The HEAD start
 * is the only leg whose installer runs against an empty machine with the
 * CURRENT script, and the only one whose updater is the one we ship today
 * (every tag leg exercises the OLD build's updater handing off to HEAD).
 * @param {TagAnnotation[]} tags
 * @returns {Start[]}
 */
export function startsFor(tags) {
  return [
    ...tags.map((t) => /** @type {Start} */ ({ from: t.ref, to: 'HEAD', desktop: t.desktop })),
    { from: 'HEAD', to: 'NEXT', desktop: true },
  ];
}

/**
 * Column label for a start. Tag columns keep their bare tag (every one
 * targets HEAD); the HEAD start spells out its synthetic target.
 * @param {string} from @param {string} to
 * @returns {string}
 */
function startLabel(from, to) {
  return to === 'HEAD' ? from : `${from} -> ${to}`;
}

/** @type {Record<Os, OsSpec>} */
export const SPEC = {
  windows: {
    install: [
      // irm https://hermes.nousresearch.com/install.ps1 | iex
      { method: 'installer-script' },
      // The same one-liner with -IncludeDesktop: builds Hermes.exe AND
      // registers Start Menu / Desktop shortcuts, so it is a second real
      // path to a hand-launchable app.
      { method: 'installer-script+desktop' },
      // Website Hermes-Setup.exe, clicked through the GUI.
      { method: 'desktop-installer', versions: ['latest'] },
    ],
    update: [
      { method: 'installer-script' },
      { method: 'installer-script+desktop' },
      // Run the bootstrap exe again over an existing install (--update flow).
      { method: 'desktop-installer', versions: ['latest'] },
      { method: 'hermes-update' },
      // Settings -> About -> "Update now", app launched from the installed
      // exe (the entry point the desktop installer created).
      { method: 'open-app-update' },
      // Same button, app launched via `hermes desktop`.
      { method: 'hermes-desktop-app-update' },
    ],
  },
  macos: {
    install: [
      { method: 'installer-script' },
      { method: 'installer-script+desktop' },
      // The published Hermes-Setup.dmg from the website, mounted and run.
      { method: 'desktop-installer', versions: ['latest'] },
    ],
    update: [
      { method: 'installer-script' },
      { method: 'installer-script+desktop' },
      { method: 'hermes-update' },
      // install.sh --include-desktop builds the .app inside the checkout
      // but registers no OS entry point, so open-app-update legs pair
      // with a desktop-installer install (the published dmg).
      { method: 'open-app-update' },
      { method: 'hermes-desktop-app-update' },
    ],
  },
  linux: {
    install: [
      { method: 'installer-script' },
      { method: 'installer-script+desktop' },
    ],
    update: [
      { method: 'installer-script' },
      { method: 'installer-script+desktop' },
      { method: 'hermes-update' },
      // No desktop installer and no packaged desktop artifact exist for
      // linux, so there is no open-app-update; `hermes desktop` is always
      // the source-mode path (build apps/desktop from the checkout, launch
      // electron) and is the one app surface a linux install has.
      { method: 'hermes-desktop-app-update' },
    ],
  },
};

/**
 * Expand one method entry into concrete ids ("desktop-installer@latest").
 * @param {InstallEntry | UpdateEntry} entry
 * @returns {string[]}
 */
export function expandMethod(entry) {
  if (!entry.versions) return [entry.method];
  return entry.versions.map((v) => `${entry.method}@${v}`);
}

/**
 * Every {os, install, update} combination in SPEC.
 * @param {Record<Os, OsSpec>} spec
 * @returns {{os: Os, install: string, update: string}[]}
 */
export function generateEnvironments(spec) {
  /** @type {{os: Os, install: string, update: string}[]} */
  const envs = [];
  for (const [os, osSpec] of /** @type {[Os, OsSpec][]} */ (Object.entries(spec))) {
    for (const install of osSpec.install.flatMap(expandMethod)) {
      for (const update of osSpec.update.flatMap(expandMethod)) {
        envs.push({ os, install, update });
      }
    }
  }
  return envs;
}

/**
 * Split the combinations into one matrix per OS.
 *
 * The starts (startsFor: each sampled tag -> HEAD, plus HEAD -> NEXT) are
 * the OUTER axis: for each start, for each combination, one dispatch that
 * installs `from` and updates to `to`. No capability filtering happens here -- every
 * declared combination is dispatched, and the OS's run workflow natively
 * skips what its driver cannot run yet. Entry names carry everything (os,
 * method pair, tag transition) because slash-joined leg names are all the
 * graph renders.
 *
 * @param {{os: Os, install: string, update: string}[]} envs
 * @param {TagAnnotation[]} tags
 * @returns {Record<Os, {include: MatrixEntry[]}>}
 */
export function buildMatrices(envs, tags) {
  /** @type {Record<Os, {include: MatrixEntry[]}>} */
  const byOs = { linux: { include: [] }, windows: { include: [] }, macos: { include: [] } };
  for (const env of envs) {
    for (const start of startsFor(tags)) {
      const name = `${env.os}: ${env.install} -> ${env.update} (${start.from} -> ${start.to})`;
      /** @type {MatrixEntry} */
      const entry = {
        name,
        leg_id: legId(name),
        install_method: env.install,
        update_method: env.update,
        install_ref: start.from,
        update_ref: start.to,
        // Every OS declares desktop-surface methods (both app-update
        // variants at minimum), so every leg carries the annotation and
        // its run workflow can natively skip pre-desktop tags.
        tag_has_desktop: start.desktop,
      };
      byOs[env.os].include.push(entry);
    }
  }
  return byOs;
}

/** Source-leg presets: the OS matrices each named route runs. */
const ROUTE_OSES = /** @type {Record<string, Os[]>} */ ({
  all: ['linux', 'windows', 'macos'],
  both: ['linux'],
  update: ['linux'],
  installer: ['linux'],
  'windows-desktop': ['windows'],
  'macos-desktop': ['macos'],
  // Bundled legs come from bundle-plan.mjs, not this matrix.
  bundled: [],
  'windows-bundled': [],
  'macos-bundled': [],
});

/**
 * Narrow the matrices to a route: a preset keeps whole OS matrices; anything else selects
 * legs by name, so a leg name, a fragment of one, or a pasted job name ("<leg> / e2e") runs
 * just those legs. A route that selects nothing throws rather than yielding a green empty run.
 *
 * @param {Record<Os, {include: MatrixEntry[]}>} matrices
 * @param {string} route
 * @returns {Record<Os, {include: MatrixEntry[]}>}
 */
export function selectRoute(matrices, route) {
  const oses = ROUTE_OSES[route];
  /** @type {(os: Os, entry: MatrixEntry) => boolean} */
  const keep = oses
    ? (os) => oses.includes(os)
    : (_os, entry) => entry.name.includes(route) || route.startsWith(`${entry.name} /`);
  /** @type {Record<Os, {include: MatrixEntry[]}>} */
  const picked = { linux: { include: [] }, windows: { include: [] }, macos: { include: [] } };
  for (const os of /** @type {Os[]} */ (Object.keys(picked))) {
    picked[os].include = matrices[os].include.filter((entry) => keep(os, entry));
  }
  if (!oses && Object.values(picked).every((m) => m.include.length === 0)) {
    throw new Error(`route ${JSON.stringify(route)} is not a preset and matches no leg name`);
  }
  return picked;
}

/**
 * Does this method id need the starting tag to ship the desktop app?
 * Shared by the plan chart (pre-desktop cells) and the results chart
 * (labeling WHY a skipped leg skipped).
 * @param {string} m
 * @returns {boolean}
 */
export function methodNeedsDesktop(m) {
  return m.startsWith('desktop-installer') || m === 'installer-script+desktop' ||
    m === 'open-app-update' || m === 'hermes-desktop-app-update';
}

/**
 * Render the plan as a markdown cross-table for $GITHUB_STEP_SUMMARY:
 * one row per {os, install -> update} combination, one column per
 * starting tag. Every cell is dispatched; whether it RUNS or greys out
 * is the run workflow's call (capability lives there, not here), so the
 * chart only distinguishes the one thing the plan itself knows:
 * desktop-surface legs from tags that predate the desktop app.
 *
 * @param {{os: Os, install: string, update: string}[]} envs
 * @param {TagAnnotation[]} tags
 * @returns {string}
 */
export function renderMarkdownPlan(envs, tags) {
  const starts = startsFor(tags);
  const lines = [
    '### Install & Update E2E plan',
    '',
    `${envs.length} combinations x ${starts.length} starting points = ${envs.length * starts.length} legs`,
    '',
    `| combination | ${starts.map((s) => startLabel(s.from, s.to)).join(' | ')} |`,
    `|---|${starts.map(() => '---').join('|')}|`,
  ];
  for (const env of envs) {
    const cells = starts.map((start) => {
      if (
        !start.desktop &&
        (methodNeedsDesktop(env.install) || methodNeedsDesktop(env.update))
      ) {
        return 'pre-desktop';
      }
      return '⏳ 	';
    });
    lines.push(`| \`${env.os}: ${env.install} -> ${env.update}\` | ${cells.join(' | ')} |`);
  }
  return lines.join('\n');
}

/**
 * Render the run's OUTCOME as the same cross-table, from the run's own job
 * list (GitHub Actions API): one row per combination, one column per tag,
 * each cell the leg's conclusion. Input is NDJSON {name, conclusion} lines
 * -- what `gh api --paginate --jq '.jobs[] | {name, conclusion}'` emits --
 * and legs are recognized by the exact name shape buildMatrices mints
 * ("os: install -> update (from -> HEAD|NEXT) / ..."), so unrelated jobs
 * (pick-releases, the report job itself) fall out naturally.
 *
 * @param {{name: string, conclusion: string | null}[]} jobs
 * @param {TagAnnotation[]} [tagAnnotations] When given (the same --tags the
 *   plan got), skipped cells carry their REASON: `pre-desktop` when a
 *   desktop-surface method meets a tag that predates apps/desktop, `TODO`
 *   when the pair is declared but no driver arm runs it yet.
 * @param {Map<string, number>} [artifactById] Artifact name -> id for this
 *   run (the report job's --artifacts). Legs that RAN (success or failure)
 *   get TWO links: 📼 to the run's single player artifact (playback.html,
 *   archive:false names artifacts after the FILE) with the leg's logs zip
 *   as a #zip= HASH param — the artifact URL 307s server-side and strips
 *   the query, the hash survives — and ⬇️ to the raw logs zip. Logs
 *   artifacts are `install-e2e-logs-<leg_id>` (windows) or
 *   `install-e2e-logs-<leg_id>-<sha>` (posix arms append the sha at
 *   upload) — matched by prefix. leg_id is rebuilt from the parsed job
 *   name with the same formula the generator mints (legId).
 * @returns {string}
 */
export function renderMarkdownResults(jobs, tagAnnotations = [], artifactById = new Map()) {
  const knownRules = JSON.parse(fs.readFileSync(new URL('../../tests/install/e2e-assets/known-failures.json', import.meta.url), 'utf8'));
  /** @type {Map<string, {number: number, rule: any}>} */
  const footnotes = new Map();
  const LEG = /^(linux|windows|macos): (\S+) -> (\S+) \(([^)]+) -> (HEAD|NEXT)\) \//;
  /** @type {Map<string, boolean>} */
  const desktopByTag = new Map(tagAnnotations.map((t) => [t.ref, t.desktop]));
  /**
   * @param {string} install @param {string} update @param {string} tag
   * @returns {string}
   */
  const skipLabel = (install, update, tag) => {
    if (desktopByTag.size === 0) return 'skip';
    if (
      desktopByTag.get(tag) === false &&
      (methodNeedsDesktop(install) || methodNeedsDesktop(update))
    ) {
      return 'pre-desktop';
    }
    return 'TODO';
  };
  // A combination can surface as SEVERAL jobs with the same leg name (a
  // run workflow may have one inner job per driver arm; exactly one runs
  // and the others natively skip), so cells merge by significance: a real
  // outcome always beats a skip, and a bad outcome beats a good one.
  const RANK = ['skip', 'TODO', 'pre-desktop', '&#x2705;', 'known', 'running', 'cancelled', '&#x274C;'];
  const SKIPS = ['skip', 'TODO', 'pre-desktop'];
  // Rendered success/failure cells carry artifact links after the glyph;
  // rank by the leading token or every such cell would rank as unknown (-1)
  // and lose to any skip already in the map.
  /** @param {string} cell */
  const rankOf = (cell) => RANK.findIndex((t) => cell === t || cell.startsWith(`${t} `));
  /** @type {Map<string, Map<string, string>>} */
  const rows = new Map();
  /** @type {string[]} */
  const tags = [];
  for (const job of jobs) {
    const m = job.name.match(LEG);
    if (!m) continue;
    const combo = `${m[1]}: ${m[2]} -> ${m[3]}`;
    const tag = startLabel(m[4], m[5]);
    if (!tags.includes(tag)) tags.push(tag);
    if (!rows.has(combo)) rows.set(combo, new Map());
    const cell = (() => {
    const legId2 = legId(`${m[1]}: ${m[2]} -> ${m[3]} (${m[4]} -> ${m[5]})`);
    // Logs artifacts: `install-e2e-logs-<leg_id>` (windows) or with a
    // trailing `-<sha>` (posix arms append it at upload). Match by prefix.
    const logsName = [...artifactById.keys()].find((n) => n.startsWith(`install-e2e-logs-${legId2}`));
    // Player artifacts: the single per-run leg-player upload; archive:
    // false names it after the FILE (playback.html), ignoring `name:`.
    const playerName = [...artifactById.keys()].find((n) => n === 'playback.html');
    const playerId = playerName !== undefined ? artifactById.get(playerName) : undefined;
    const logsId = logsName !== undefined ? artifactById.get(logsName) : undefined;
    // Two links per ran leg: the player with the zip as a HASH param
    // (GitHub's artifact URL 307s to /suites/... and strips the query —
    // the hash survives client-side), and the raw zip download.
    const runBase = `https://github.com/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`;
    const reel = (playerId !== undefined && logsId !== undefined)
      ? ` [📼](${runBase}/artifacts/${playerId}#zip=${encodeURIComponent(`${runBase}/artifacts/${logsId}`)}) [⬇️](${runBase}/artifacts/${logsId})`
      : '';
    switch (job.conclusion) {
      case 'success': {
        // Only the classifier's uploaded receipt turns a successful job into
        // a known-failure cell. Tag membership alone never suppresses a red.
        const rule = knownRules.find((/** @type {any} */ r) => artifactById.has(`install-e2e-known-${r.id}--${legId2}`));
        if (!rule) return `&#x2705;${reel}`;
        if (!footnotes.has(rule.id)) footnotes.set(rule.id, { number: footnotes.size + 1, rule });
        return `known [^${footnotes.get(rule.id)?.number}]${reel}`;
      }
      case 'failure': return `&#x274C;${reel}`;
      case 'skipped': return skipLabel(m[2], m[3], m[4]);
      case 'cancelled': return 'cancelled';
      default: return 'running';
    }
  })();
    const byTag = /** @type {Map<string, string>} */ (rows.get(combo));
    const prev = byTag.get(tag);
    if (prev === undefined || rankOf(cell) > rankOf(prev)) {
      byTag.set(tag, cell);
    }
  }
  if (rows.size === 0) return '### Install & Update E2E results\n\n(no legs found in this run)\n';
  const cells = [...rows.values()].flatMap((r) => [...r.values()]);
  const passed = cells.filter((c) => c.startsWith('&#x2705;')).length;
  const failed = cells.filter((c) => c.startsWith('&#x274C;')).length;
  const skipped = cells.filter((c) => SKIPS.includes(c)).length;
  const known = cells.filter((c) => c.startsWith('known ')).length;
  const lines = [
    '### Install & Update E2E results',
    '',
    `${passed} passed, ${failed} failed, ${known} known failures, ${skipped} skipped (TODO = declared, no driver arm yet; pre-desktop = the starting release predates apps/desktop), ${cells.length} legs total`,
    '',
    `| combination | ${tags.join(' | ')} |`,
    `|---|${tags.map(() => '---').join('|')}|`,
  ];
  for (const [combo, byTag] of rows) {
    lines.push(`| \`${combo}\` | ${tags.map((t) => byTag.get(t) || '-').join(' | ')} |`);
  }
  lines.push('');
  for (const { number, rule } of footnotes.values()) {
    lines.push(`[^${number}]: **${rule.title}.** ${rule.explanation} [Evidence](${rule.evidence}).`);
  }
  return lines.join('\n');
}

/** @returns {Promise<string>} all of stdin */
function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.on('data', (c) => { data += c; });
    process.stdin.on('end', () => resolve(data));
  });
}

async function main() {
  const { values } = parseArgs({
    options: {
      tags: { type: 'string', default: '[]' },
      route: { type: 'string', default: 'all' },
      format: { type: 'string', default: 'json' },
      artifacts: { type: 'string' },
    },
  });
  if (values.format === 'results') {
    const jobs = (await readStdin()).split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
    const annotations = /** @type {TagAnnotation[]} */ (JSON.parse(values.tags));
    /** @type {Map<string, number>} */
    const artifactById = new Map();
    if (values.artifacts) {
      for (const line of (await fs.promises.readFile(values.artifacts, 'utf8')).split('\n')) {
        if (!line.trim()) continue;
        const a = JSON.parse(line);
        if (typeof a.name === 'string' && typeof a.id === 'number') artifactById.set(a.name, a.id);
      }
    }
    process.stdout.write(renderMarkdownResults(jobs, annotations, artifactById));
    return;
  }
  const tags = /** @type {TagAnnotation[]} */ (JSON.parse(values.tags));
  const envs = generateEnvironments(SPEC);
  if (values.format === 'markdown') {
    process.stdout.write(renderMarkdownPlan(envs, tags));
    return;
  }
  const matrices = selectRoute(buildMatrices(envs, tags), values.route || 'all');
  process.stdout.write(`${JSON.stringify(matrices, null, 2)}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  await main();
}
