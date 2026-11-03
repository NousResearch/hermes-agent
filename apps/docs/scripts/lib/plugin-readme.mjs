// README rendering for the per-plugin pages (/docs/plugins/<name>).
//
// Every entry gets its README rendered on its page (opt out with `readme: false`). The
// build fetches it from the PINNED commit (extract-plugins.py emits `readmeUrl` as a
// raw.githubusercontent.com / gitlab.com raw URL at <sha>) — never a branch tip — so the
// page shows exactly the README the catalog reviewer read, and it only changes when the
// author re-pins through a reviewed PR.
//
// Rendering is build-time and allowlisted: markdown → mdast (remark-parse + GFM) → hast
// (remark-rehype WITHOUT allowDangerousHtml, so raw HTML in the README is dropped) → a
// small serializer that emits only the tags/attributes below and only http(s)/mailto/#
// URLs. Relative image and link paths are rewritten to the pinned commit on the forge.
// Third-party READMEs are NEVER compiled as MDX/JS — this is plain markdown → string.
//
// Direct dependencies keep this renderer independent of the documentation engine.
// Load them lazily so pure URL/serializer helpers also work before installation.

import fs from "node:fs";
import path from "node:path";

const FETCH_TIMEOUT_MS = 15_000;
const MAX_README_BYTES = 512 * 1024;
// Hosts the build is allowed to fetch README bodies from (raw file endpoints at a sha).
const README_HOSTS = new Set(["raw.githubusercontent.com", "gitlab.com"]);

const ALLOWED_TAGS = new Set([
  "a", "p", "br", "hr", "blockquote", "pre", "code", "em", "strong", "del", "s",
  "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "thead", "tbody",
  "tr", "th", "td", "img", "input", "sup", "sub", "section", "div", "span", "kbd",
]);
const ALLOWED_ATTRS = {
  "*": new Set(["id", "className", "align"]),
  a: new Set(["href", "title"]),
  img: new Set(["src", "alt", "title", "width", "height"]),
  input: new Set(["type", "checked", "disabled"]),
  ol: new Set(["start"]),
  td: new Set(["align", "colSpan", "rowSpan"]),
  th: new Set(["align", "colSpan", "rowSpan"]),
  li: new Set(["className"]),
};
const VOID_TAGS = new Set(["br", "hr", "img", "input"]);
const ATTR_NAME = { className: "class", colSpan: "colspan", rowSpan: "rowspan" };

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function safeUrl(value) {
  const url = String(value ?? "").trim();
  if (url === "" || url.startsWith("#")) return url;
  if (/^(https?:|mailto:)/i.test(url)) return url;
  return null; // javascript:, data:, vbscript:, protocol-relative … are dropped
}

/** Where relative README paths point at the pinned commit: {raw, blob} bases. */
function forgeBases(readmeUrl) {
  const u = new URL(readmeUrl);
  const dir = u.pathname.replace(/[^/]*$/, ""); // strip README.md
  if (u.hostname === "raw.githubusercontent.com") {
    // /owner/repo/<sha>/<dir>/
    const [, owner, repo, sha, ...rest] = dir.split("/");
    const sub = rest.join("/");
    return {
      raw: `https://raw.githubusercontent.com/${owner}/${repo}/${sha}/${sub}`,
      blob: `https://github.com/${owner}/${repo}/blob/${sha}/${sub}`,
    };
  }
  // gitlab: /group/proj/-/raw/<sha>/<dir>/
  const blob = dir.replace("/-/raw/", "/-/blob/");
  return { raw: `https://${u.hostname}${dir}`, blob: `https://${u.hostname}${blob}` };
}

function absolutize(value, base) {
  const v = String(value ?? "").trim();
  if (v === "" || v.startsWith("#") || /^[a-z][a-z0-9+.-]*:/i.test(v) || v.startsWith("//")) return v;
  try {
    return new URL(v.replace(/^\.\//, ""), base).toString();
  } catch {
    return v;
  }
}

function rewriteUrls(node, bases) {
  if (node.type === "element") {
    const props = node.properties || {};
    if (node.tagName === "img" && props.src != null) props.src = absolutize(props.src, bases.raw);
    if (node.tagName === "a" && props.href != null) props.href = absolutize(props.href, bases.blob);
    // Prefix ids the pipeline already produced (remark-gfm's footnote section
    // label) with our readme- namespace; applyAnchors below assigns slug ids to
    // the remaining headings, so nothing clashes with the page chrome.
    if (/^h[1-6]$/.test(node.tagName) && typeof props.id === "string") props.id = `readme-${props.id}`;
  }
  for (const child of node.children || []) rewriteUrls(child, bases);
}

/** Concatenated text of a hast subtree — what GitHub anchors slug a heading
 *  from. Raw HTML never reaches the tree (remark-rehype runs without
 *  allowDangerousHtml), so inline code/strong/em flatten to plain text; image
 *  alt text does not contribute, matching hast-util-to-string. */
function subtreeText(node) {
  if (node.type === "text") return node.value;
  let out = "";
  for (const child of node.children || []) out += subtreeText(child);
  return out;
}

/** Percent-decode a link fragment; malformed %-sequences pass through as written. */
function decodeFragment(frag) {
  try {
    return decodeURIComponent(frag);
  } catch {
    return frag;
  }
}

/** Preserve the original pinned source when a fragment cannot be hosted safely. */
function pinnedSourceUrl(readmeUrl) {
  const u = new URL(readmeUrl);
  if (u.hostname === "raw.githubusercontent.com") {
    const [, owner, repo, sha, ...rest] = u.pathname.split("/");
    return `https://github.com/${owner}/${repo}/blob/${sha}/${rest.join("/")}`;
  }
  return `https://${u.hostname}${u.pathname.replace("/-/raw/", "/-/blob/")}`;
}

/** Namespace heading/footnote anchors away from page chrome. Unknown fragments
 *  retain the pinned source destination; sanitizing a README must not require
 *  executing its raw HTML. This does not assert the upstream fragment is valid. */
function applyAnchors(tree, slugger, slug, sourceUrl) {
  const targets = new Map(); // fragment as the README writes it → id on this page
  (function namespace(node) {
    if (node.type === "element" && typeof (node.properties || {}).id === "string") {
      const props = node.properties;
      if (!props.id.startsWith("readme-")) props.id = `readme-${props.id}`;
      targets.set(props.id.slice("readme-".length), props.id);
    }
    for (const child of node.children || []) namespace(child);
  })(tree);
  (function assignHeadingIds(node) {
    if (node.type === "element" && /^h[1-6]$/.test(node.tagName)) {
      const props = node.properties || (node.properties = {});
      if (typeof props.id !== "string") {
        const text = subtreeText(node);
        // The stateful slugger dedupes across headings; the loop additionally
        // avoids ids already claimed in step 1 (slugger repeats yield -1, -2 …).
        let headingSlug = slugger.slug(text);
        while (targets.get(headingSlug) === `readme-${headingSlug}`) headingSlug = slugger.slug(text);
        if (headingSlug) props.id = `readme-${headingSlug}`;
      }
      if (typeof props.id === "string") targets.set(props.id.slice("readme-".length), props.id);
    }
    for (const child of node.children || []) assignHeadingIds(child);
  })(tree);
  (function rewriteFragments(node) {
    if (node.type === "element" && node.tagName === "a") {
      const props = node.properties || {};
      if (typeof props.href === "string" && props.href.startsWith("#") && props.href !== "#") {
        const raw = props.href.slice(1);
        // The pure `slug` normalizes the fragment like GitHub's anchor lookup;
        // the slugger instance is never used here — it would dedupe against
        // headings and shift every later id.
        const decoded = decodeFragment(raw);
        const target = targets.get(decoded) ?? targets.get(slug(decoded));
        props.href = target ? `#${target}` : `${sourceUrl}#${raw}`;
      }
    }
    for (const child of node.children || []) rewriteFragments(child);
  })(tree);
}

function serializeAttrs(tagName, props) {
  const allowed = ALLOWED_ATTRS[tagName];
  let out = "";
  for (const [key, raw] of Object.entries(props || {})) {
    if (!(ALLOWED_ATTRS["*"].has(key) || (allowed && allowed.has(key)))) continue;
    if (raw == null || raw === false) continue;
    let value = Array.isArray(raw) ? raw.join(" ") : raw;
    if (key === "href" || key === "src") {
      value = safeUrl(value);
      if (value == null) continue;
    }
    if (tagName === "input" && key === "type" && value !== "checkbox") continue;
    const name = ATTR_NAME[key] || key;
    out += value === true ? ` ${name}` : ` ${name}="${escapeHtml(value)}"`;
  }
  return out;
}

function serialize(node) {
  if (node.type === "root") return (node.children || []).map(serialize).join("");
  if (node.type === "text") return escapeHtml(node.value);
  if (node.type !== "element") return ""; // comments, raw html, doctype — dropped
  const tag = node.tagName.toLowerCase();
  const inner = (node.children || []).map(serialize).join("");
  if (!ALLOWED_TAGS.has(tag)) return inner; // unwrap unknown elements, keep their text
  const open = `<${tag}${serializeAttrs(tag, node.properties)}>`;
  return VOID_TAGS.has(tag) ? open : `${open}${inner}</${tag}>`;
}

let pipelinePromise = null;
function loadPipeline() {
  if (pipelinePromise) return pipelinePromise;
  pipelinePromise = (async () => {
    // Direct dependencies of apps/docs (no more @docusaurus/mdx-loader indirection).
    const [{ unified }, remarkParse, remarkGfm, remarkRehype, slugger] = await Promise.all([
      import("unified"), import("remark-parse"), import("remark-gfm"), import("remark-rehype"),
      import("github-slugger"),
    ]);
    return {
      processor: unified().use(remarkParse.default).use(remarkGfm.default).use(remarkRehype.default),
      // github-slugger: the default class is stateful and duplicate-aware (a
      // fresh instance per render); the named `slug` is the pure normalizer.
      GithubSlugger: slugger.default,
      slug: slugger.slug,
    };
  })();
  return pipelinePromise;
}

/** Markdown → allowlisted HTML string. `readmeUrl` anchors relative paths at the
 *  pinned commit and is the fragment fallback target for anchors the sanitizer
 *  drops. */
async function renderReadme(markdown, readmeUrl) {
  const { processor, GithubSlugger, slug } = await loadPipeline();
  const tree = await processor.run(processor.parse(markdown));
  rewriteUrls(tree, forgeBases(readmeUrl));
  applyAnchors(tree, new GithubSlugger(), slug, pinnedSourceUrl(readmeUrl));
  return serialize(tree);
}

/** README.md candidates at the pinned commit: the entry's subdir first, then the repo root
 *  (a monorepo plugin often documents itself in the root README), each in both casings. */
function readmeCandidates(readmeUrl) {
  // subdir README first, then the repo root; common casings and a docs/ README as fallbacks.
  const root = readmeUrl.replace(/(\/[0-9a-f]{40}\/).+$/, "$1README.md");
  const paths = root === readmeUrl ? [readmeUrl] : [readmeUrl, root];
  const names = ["README.md", "readme.md", "Readme.md", "README.MD", "README", "docs/README.md"];
  return paths.flatMap((p) => names.map((n) => p.replace(/README\.md$/, n)));
}

/** Resolve the README at the pinned commit → `{ markdown, url }` (url = the file that answered), or null. */
async function fetchReadme(readmeUrl, cacheDir, log) {
  const u = new URL(readmeUrl);
  if (u.protocol !== "https:" || !README_HOSTS.has(u.hostname)) {
    log(`refusing README fetch from ${u.hostname}`);
    return null;
  }
  // The URL embeds the sha, so a cache hit is exact; the cache lives outside git.
  const cacheFile = path.join(cacheDir, `${Buffer.from(readmeUrl).toString("base64url")}.json`);
  if (fs.existsSync(cacheFile)) return JSON.parse(fs.readFileSync(cacheFile, "utf8"));
  for (const candidate of readmeCandidates(readmeUrl)) {
    try {
      const res = await fetch(candidate, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), redirect: "follow" });
      if (res.status === 404) continue;
      if (!res.ok) {
        log(`README fetch ${candidate} → HTTP ${res.status}`);
        return null;
      }
      const text = await res.text();
      if (Buffer.byteLength(text, "utf8") > MAX_README_BYTES) {
        log(`README at ${candidate} exceeds ${MAX_README_BYTES} bytes; skipped`);
        return null;
      }
      const result = { markdown: text, url: candidate };
      fs.mkdirSync(cacheDir, { recursive: true });
      fs.writeFileSync(cacheFile, JSON.stringify(result));
      return result;
    } catch (err) {
      log(`README fetch ${candidate} failed: ${err && err.message ? err.message : err}`);
      return null;
    }
  }
  log(`no README.md at ${readmeUrl}`);
  return null;
}

export {
  fetchReadme,
  renderReadme,
  readmeCandidates,
  rewriteUrls,
  serialize,
  serializeAttrs,
  safeUrl,
  absolutize,
  forgeBases,
};
