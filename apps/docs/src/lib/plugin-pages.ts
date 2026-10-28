// Build-time data layer for the per-plugin pages (/plugins/<name>) and author pages
// (/plugins/by/<slug>).
//
//   /plugins/<name>        one page per entry in public/api/plugins.json
//   /plugins/by/<slug>     one page per maintainer (slug from extract-plugins.py)
//
// Both are generated from the same plugins.json the catalog grid fetches, so there is no
// second data source: a merged catalog PR is the only way a page appears, changes or
// disappears. Every entry's README is fetched from the pinned commit and rendered through
// the allowlist in ../../scripts/lib/plugin-readme.mjs unless the entry sets `readme: false`.
//
// This is the Next/Fumadocs replacement for the Docusaurus plugin
// (formerly website/plugins/plugin-catalog-pages/index.js), whose loadContent/contentLoaded
// + addRoute pair becomes generateStaticParams in the route adapters plus the pure
// builders below. The site degrades, never fails: a missing plugins.json (extract-plugins.py
// did not run) yields zero pages with a warning, and a README that cannot be fetched or
// rendered leaves that section off the page.
//
// Server/build-time only (reads the filesystem and fetches over the network). Route
// adapters under src/app/(en)/(catalog) and src/app/zh-Hans/(catalog) consume it; nothing
// under a 'use client' boundary may import this module at runtime.

import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import type { CatalogMeta, CatalogPlugin } from "../components/PluginCatalog/catalog";

export interface CatalogSnapshot {
  entries: CatalogPlugin[];
  meta: CatalogMeta;
}

/** Compact author record shared by the plugin and author pages. */
export interface CatalogAuthor {
  slug: string;
  name: string;
  count: number;
}

export interface PluginPageData {
  plugin: CatalogPlugin;
  /** Allowlisted HTML rendered at build time from the README at the pinned commit; null when not opted in. */
  readmeHtml: string | null;
  /** Raw URL of the README file that was rendered (subdir or repo root at the pinned sha). */
  readmeSourceUrl: string | null;
  author: CatalogAuthor | null;
  moreByAuthor: CatalogPlugin[];
  generatedAt: string | null;
  starsFetchedAt: string | null;
}

export interface AuthorPageData {
  author: CatalogAuthor;
  plugins: CatalogPlugin[];
  generatedAt: string | null;
}

// Routes the static API JSON is served from. `public/` is the Next static dir and the
// app's global basePath is `/docs`, so the public URL of `public/api/plugins.json` is
// `/docs/api/plugins.json` — the same file the catalog grid fetches at runtime. Paths are
// cwd-relative because Next runs with the docs app as the working directory.
const PLUGINS_JSON = path.join("public", "api", "plugins.json");
const META_JSON = path.join("public", "api", "plugins-meta.json");
// Exact (sha-keyed) README cache, outside git (ignored via apps/docs/.gitignore).
const README_CACHE_DIR = path.join(".cache", "plugin-readmes");

function log(msg: string) {
  console.warn(`[plugin-catalog-pages] ${msg}`);
}

async function readJsonFile<T>(file: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await readFile(file, "utf8")) as T;
  } catch {
    return fallback;
  }
}

// ── Pure builders (no fs/network; unit-tested in apps/docs/tests) ─────────────

/** Group entries by maintainerSlug, preserving the catalog's (stars-first) order inside a group. */
export function groupAuthors(entries: CatalogPlugin[]): { slug: string; name: string; plugins: CatalogPlugin[] }[] {
  const authors = new Map<string, { slug: string; name: string; plugins: CatalogPlugin[] }>();
  for (const entry of entries) {
    const slug = entry.maintainerSlug || "unknown";
    const author = authors.get(slug) || { slug, name: entry.maintainer || slug, plugins: [] };
    author.plugins.push(entry);
    authors.set(slug, author);
  }
  return [...authors.values()];
}

/** The page payload for one entry, or null when the name is not in the catalog.
 *  `readme` is the rendered README ({html, url}) when the entry opted in and it resolved. */
export function buildPluginPageData(
  snapshot: CatalogSnapshot,
  name: string,
  readme?: { html: string; url: string } | null,
): PluginPageData | null {
  const entry = snapshot.entries.find((e) => e.name === name);
  if (!entry) return null;
  const author = groupAuthors(snapshot.entries).find((a) => a.slug === (entry.maintainerSlug || "unknown")) || null;
  // Compact cards for shelves ("More by this author") — the full entry minus nothing heavy;
  // keeping the shape identical to plugins.json keeps one card type.
  const siblings = author ? author.plugins.filter((p) => p.name !== entry.name) : [];
  return {
    plugin: entry,
    readmeHtml: readme?.html || null,
    readmeSourceUrl: readme?.url || null,
    author: author ? { slug: author.slug, name: author.name, count: author.plugins.length } : null,
    moreByAuthor: siblings,
    generatedAt: snapshot.meta.generatedAt || null,
    starsFetchedAt: snapshot.meta.starsFetchedAt || null,
  };
}

/** The page payload for one maintainer, or null when the slug is not in the catalog. */
export function buildAuthorPageData(snapshot: CatalogSnapshot, slug: string): AuthorPageData | null {
  const author = groupAuthors(snapshot.entries).find((a) => a.slug === slug);
  if (!author) return null;
  return {
    author: { slug: author.slug, name: author.name, count: author.plugins.length },
    plugins: author.plugins,
    generatedAt: snapshot.meta.generatedAt || null,
  };
}

/** Reverse the encoding generateStaticParams' params went through. Catalog names/slugs are
 *  URL-safe, so this is a no-op for every real entry; the guard keeps a literal '%' in a
 *  future name from 500ing the lookup. */
export function decodeCatalogParam(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

// ── Build-time IO (fs + network via the renderer) ──────────────────────────────

// Memoize the snapshot per process, keyed by the (mtime, size) of plugins.json so a dev
// server picks up regenerated catalogs without restarting. Missing file → empty snapshot.
let snapshotCache: { key: string; promise: Promise<CatalogSnapshot> } | null = null;

async function snapshotKey(): Promise<string> {
  try {
    const s = await stat(PLUGINS_JSON);
    return `${s.mtimeMs}:${s.size}`;
  } catch {
    return "missing";
  }
}

export async function loadCatalogSnapshot(): Promise<CatalogSnapshot> {
  const key = await snapshotKey();
  if (snapshotCache && snapshotCache.key === key) return snapshotCache.promise;
  const promise = (async () => {
    const entries = await readJsonFile<unknown>(PLUGINS_JSON, null);
    if (!Array.isArray(entries)) {
      log(`${PLUGINS_JSON} missing or invalid; generating no plugin pages`);
      return { entries: [] as CatalogPlugin[], meta: {} as CatalogMeta };
    }
    const meta = await readJsonFile<CatalogMeta>(META_JSON, {});
    return { entries: entries as CatalogPlugin[], meta };
  })();
  snapshotCache = { key, promise };
  return promise;
}

/** Every plugin name that must become a static /plugins/<name> page. Empty when the
 *  catalog snapshot is missing — the Docusaurus plugin's "zero pages with a warning". */
export async function listPluginPageParams(): Promise<string[]> {
  const { entries } = await loadCatalogSnapshot();
  return entries.map((e) => e.name);
}

/** Every maintainer slug that must become a static /plugins/by/<slug> page. */
export async function listAuthorPageParams(): Promise<string[]> {
  const { entries } = await loadCatalogSnapshot();
  return groupAuthors(entries).map((a) => a.slug);
}

/** The renderer lives outside src (build-time only) so it can never be pulled into a
 *  client bundle; it loads the remark pipeline lazily. */
async function loadRenderer() {
  return import("../../scripts/lib/plugin-readme.mjs");
}

/** Data for one plugin page: the entry, its rendered README (opt-in), its author and the
 *  "more by this author" shelf. Exactly one README is fetched per page render — bounded
 *  by the page-level build concurrency and the exact-sha disk cache — instead of fetching
 *  every README on every request. Null → the adapter renders a 404. */
export async function getPluginPageData(name: string): Promise<PluginPageData | null> {
  const snapshot = await loadCatalogSnapshot();
  const entry = snapshot.entries.find((e) => e.name === name);
  if (!entry) return null;
  let readme: { html: string; url: string } | null = null;
  if (entry.readme && entry.readmeUrl) {
    try {
      const { fetchReadme, renderReadme } = await loadRenderer();
      const fetched = await fetchReadme(entry.readmeUrl, README_CACHE_DIR, log);
      if (fetched != null) {
        try {
          readme = { html: await renderReadme(fetched.markdown, fetched.url), url: fetched.url };
        } catch (err) {
          log(`README for ${entry.name} failed to render: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    } catch (err) {
      log(`README for ${entry.name} failed to load: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return buildPluginPageData(snapshot, name, readme);
}

/** Data for one author page; null → the adapter renders a 404. */
export async function getAuthorPageData(slug: string): Promise<AuthorPageData | null> {
  const snapshot = await loadCatalogSnapshot();
  return buildAuthorPageData(snapshot, slug);
}
