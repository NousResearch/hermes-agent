#!/usr/bin/env node
// Check that every internal link in the exported HTML resolves: pages and
// artifacts must exist in the export, and page anchors (id attributes plus
// legacy <a name>) must exist on the page they target. Read-only: no network
// requests, no writes, no remote-host or mailto/custom-protocol checks.
//
// validate-output.mjs runs this as one of its checks, passing ctx.routes from
// the prerender manifest. Standalone use derives the same routes from the
// export tree itself, so a previously exported tree can be inspected without
// rebuilding it.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { parse } from 'parse5';
import { DOCS_BASE_PATH, SITE_URL, publicHref } from '../src/lib/urls.ts';
import { htmlPath } from './postbuild.mjs';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE_ORIGIN = new URL(SITE_URL).origin;
// Full-URL forms: any scheme (mailto:, custom app protocols, https) plus
// protocol-relative //host. Same-origin ones are validated like hand-written
// paths below; only same-origin links leaving /docs — deliberate product
// pages on this origin — stay out of the checker's scope.
const OUTSIDE_PATH = /^(?:[a-z][a-z\d+.-]*:|\/\/)/i;

/** Read one exported HTML file once: its link hrefs and its own anchors. */
function readPage(cache, file) {
  let page = cache.get(file);
  if (!page) {
    const hrefs = [];
    const anchors = new Set();
    const walk = (node) => {
      for (const child of node.childNodes ?? []) {
        if (child.tagName) {
          const attr = (name) => child.attrs?.find((attribute) => attribute.name === name)?.value;
          if (child.tagName === 'a' && attr('href') !== undefined) hrefs.push(attr('href'));
          const id = attr('id');
          if (id !== undefined) anchors.add(id);
          if (child.tagName === 'a') {
            const name = attr('name'); // Docusaurus-era deep links; still honored.
            if (name !== undefined) anchors.add(name);
          }
        }
        walk(child);
      }
    };
    walk(parse(readFileSync(file, 'utf8')));
    page = { hrefs, anchors };
    cache.set(file, page);
  }
  return page;
}

/** Resolve an internal /docs path against the export: a directory's
 * index.html page or an exact artifact file. A bare directory is not a
 * destination, encoded segments must decode safely, and the decoded path
 * must stay inside the export (an encoded slash can still spell a
 * traversal that path.join would normalize). */
function resolveTarget(outDir, pathname) {
  let relative;
  try {
    relative = decodeURIComponent(pathname.slice(DOCS_BASE_PATH.length));
  } catch {
    return { error: `invalid percent-encoding` };
  }
  if (relative.includes('\0')) return { error: `a null byte` };
  const exact = path.join(outDir, relative);
  const inside = path.relative(outDir, exact);
  if (inside === '..' || inside.startsWith(`..${path.sep}`) || path.isAbsolute(inside)) {
    return { escape: true };
  }
  if (statSync(exact, { throwIfNoEntry: false })?.isFile()) {
    return { file: exact, html: exact.endsWith('.html') };
  }
  const index = path.join(exact, 'index.html');
  if (statSync(index, { throwIfNoEntry: false })?.isFile()) return { file: index, html: true };
  return null;
}

/** Page anchors are decoded in the DOM; link fragments arrive percent-encoded
 * (and may be malformed), so compare after one safe decode. */
function decodeFragment(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return null;
  }
}

/** Mapping data is an authoring contract, not a broken-link exemption. */
function checkFragmentTranslations(appDir, cache, fail) {
  const file = path.join(appDir, 'src/lib/fragment-translations.json');
  if (!statSync(file, { throwIfNoEntry: false })?.isFile()) return;
  const mappings = JSON.parse(readFileSync(file, 'utf8'));
  for (const [route, pairs] of Object.entries(mappings)) {
    for (const [english, chinese] of Object.entries(pairs)) {
      const targets = [[route, english]];
      if (chinese !== null) targets.push([`/zh-Hans${route === '/' ? '' : route}`, chinese]);
      for (const [targetRoute, fragment] of targets) {
        const target = resolveTarget(path.join(appDir, 'out'), publicHref(targetRoute));
        if (typeof fragment !== 'string' || !fragment || !target?.html
          || !readPage(cache, target.file).anchors.has(fragment)) {
          fail(`fragment translation ${route}#${english} targets a missing section ${targetRoute}#${fragment}`);
        }
      }
    }
  }
}

/**
 * Check every internal href on the exported pages. Diagnostics go through
 * `fail(message)`; `ctx.routes` is the exported route list the page checks
 * already validated, so this check never throws for a broken link.
 */
export function checkExportLinks(appDir, fail, ctx) {
  const routes = ctx?.routes;
  if (!routes) return;
  const outDir = path.join(appDir, 'out');
  const cache = new Map();
  const findings = new Map();
  const report = (route, message) => {
    const finding = findings.get(message) ?? { routes: [], seen: new Set(), more: 0 };
    if (finding.seen.has(route)) return; // a page repeating a link adds no example
    finding.seen.add(route);
    if (finding.routes.length < 3) finding.routes.push(route);
    else finding.more += 1;
    findings.set(message, finding);
  };
  for (const route of routes) {
    let file;
    try {
      file = htmlPath(outDir, route);
    } catch {
      continue; // invalid route key: the page checks own that failure
    }
    if (!statSync(file, { throwIfNoEntry: false })?.isFile()) continue; // missing page: owned too
    const page = readPage(cache, file);
    const pageBase = publicHref(route === '/' ? '/' : `${route}/`);
    for (const href of page.hrefs) {
      if (!href) continue;
      if (href.startsWith('#')) {
        const anchor = decodeFragment(href.slice(1));
        if (anchor === null) {
          report(route, `link "${href}" has invalid percent-encoding in its fragment`);
        } else if (anchor && !page.anchors.has(anchor)) {
          report(route, `link "${href}" targets an anchor missing from its own page`);
        }
        continue;
      }
      let url;
      try {
        url = new URL(href, `${SITE_ORIGIN}${pageBase}`);
      } catch (error) {
        report(route, `link "${href}" is not a valid URL: ${error.message}`);
        continue;
      }
      if (url.origin !== SITE_ORIGIN) continue; // remote hosts, mailto:, custom protocols
      if (url.pathname !== DOCS_BASE_PATH && !url.pathname.startsWith(`${DOCS_BASE_PATH}/`)) {
        // Full-URL forms deliberately point at other products on this origin;
        // hand-written paths leaving /docs are authoring escapes.
        if (OUTSIDE_PATH.test(href)) continue;
        report(route, `link "${href}" escapes the ${DOCS_BASE_PATH} base path (resolves to ${url.pathname})`);
        continue;
      }
      const target = resolveTarget(outDir, url.pathname);
      if (target?.error) {
        report(route, `link "${href}" has ${target.error} in its path`);
        continue;
      }
      if (target?.escape) {
        report(route, `link "${href}" resolves outside the export directory`);
        continue;
      }
      if (!target) {
        report(route, `link "${href}" resolves to ${url.pathname}, which is missing from the export`);
        continue;
      }
      if (url.hash && target.html) {
        const anchor = decodeFragment(url.hash.slice(1));
        if (anchor === null) {
          report(route, `link "${href}" has invalid percent-encoding in its fragment`);
        } else if (anchor && !readPage(cache, target.file).anchors.has(anchor)) {
          report(route, `link "${href}" targets a missing anchor on ${url.pathname}`);
        }
      }
    }
  }
  for (const [message, { routes: pages, more }] of findings) {
    const where = pages.length === 1
      ? ` (page ${pages[0]})`
      : ` (pages ${pages.join(', ')}${more ? `, +${more} more` : ''})`;
    fail(`${message}${where}`);
  }
  checkFragmentTranslations(appDir, cache, fail);
}

/** Standalone mode has no prerender manifest: derive routes from out/ itself. */
function derivedRoutes(outDir) {
  const routes = [];
  const walk = (directory) => {
    const seenIndex = statSync(path.join(directory, 'index.html'), { throwIfNoEntry: false })?.isFile();
    const relative = path.relative(outDir, directory).split(path.sep).join('/');
    if (seenIndex) routes.push(relative ? `/${relative}` : '/');
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (entry.isDirectory()) walk(path.join(directory, entry.name));
    }
  };
  walk(outDir);
  return routes.sort();
}

if (import.meta.main) {
  const { values } = parseArgs({ options: { 'app-dir': { type: 'string' } } });
  const appDir = values['app-dir'] ? path.resolve(values['app-dir']) : appRoot;
  const routes = derivedRoutes(path.join(appDir, 'out'));
  const errors = [];
  checkExportLinks(appDir, errors.push.bind(errors), { routes });
  for (const error of errors) console.error(`[docs check-export-links] ${error}`);
  console.log(JSON.stringify({ routes: routes.length, errors: errors.length }));
  if (errors.length) process.exitCode = 1;
}
