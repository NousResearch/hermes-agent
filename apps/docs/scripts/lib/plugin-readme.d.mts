/**
 * Build-time README renderer for the plugin catalog pages
 * (scripts/lib/plugin-readme.mjs). Markdown from the pinned commit →
 * allowlisted HTML; see the .mjs for the security contract.
 */

/** Fetched README body + the raw URL it was served from. */
export interface FetchedReadme {
  markdown: string;
  url: string;
}

/** Markdown → allowlisted HTML string. `readmeUrl` anchors relative paths at the pinned commit. */
export function renderReadme(markdown: string, readmeUrl: string): Promise<string>;

/** Resolve the README at the pinned commit; null when the host is not allowed, the fetch
 *  fails, or the body exceeds the size limit. `cacheDir` holds exact (sha-keyed) results. */
export function fetchReadme(
  readmeUrl: string,
  cacheDir: string,
  log: (msg: string) => void,
): Promise<FetchedReadme | null>;

/** README.md candidates at the pinned commit (subdir first, then repo root, casings, docs/). */
export function readmeCandidates(readmeUrl: string): string[];

/** hast node → allowlisted HTML. Only known tags/attributes survive; text of unknown
 *  elements is kept, everything else (comments, raw HTML) is dropped. */
export function serialize(node: ReadmeNode): string;

/** Attribute filter for one tag: allowlist check + URL safety + name mapping. */
export function serializeAttrs(tagName: string, props: Record<string, unknown> | undefined): string;

/** http(s)/mailto/fragment URLs pass through as-is; everything else returns null. */
export function safeUrl(value: unknown): string | null;

/** Resolve a relative URL against a base; absolute, fragment and protocol-relative values pass through. */
export function absolutize(value: unknown, base: string): string;

/** {raw, blob} URL bases for a README URL at the pinned commit. */
export function forgeBases(readmeUrl: string): { raw: string; blob: string };

/** Resolve relative img (raw base) and link (blob base) URLs at the pinned commit and
 *  prefix README heading ids with `readme-` so they cannot clash with page chrome.
 *  Mutates the tree in place. */
export function rewriteUrls(node: ReadmeNode, bases: { raw: string; blob: string }): void;

/** Minimal hast tree shape the serializer/rewriter consume. */
export interface ReadmeNode {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: ReadmeNode[];
}
