import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { parse } from 'yaml';
import { documentDescription } from './document-description.mjs';

/** Frontmatter and visible headings have separate ownership. */
export function documentMetadata(text, filename) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text);
  const data = match ? (parse(match[1]) ?? {}) : {};
  const body = match ? text.slice(match[0].length) : text;
  let fence;
  let heading;
  for (const line of body.split(/\r?\n/)) {
    const marker = /^ {0,3}(`{3,}|~{3,})/.exec(line)?.[1];
    if (marker) {
      if (!fence) fence = marker;
      else if (marker[0] === fence[0] && marker.length >= fence.length) fence = undefined;
      continue;
    }
    if (fence) continue;
    const title = /^ {0,3}#\s+(.+?)\s*#*\s*$/.exec(line)?.[1];
    if (title) {
      heading = title.replace(/\s*(?:\{#[^}]+\}|\[#([^\]]+)\])\s*$/, '').replace(/`([^`]+)`/g, '$1');
      break;
    }
  }
  return {
    ...data,
    title: data.title ?? heading ?? path.basename(filename).replace(/\.mdx?$/, ''),
    hasH1: heading !== undefined,
    description: documentDescription(text),
  };
}

export function readDocumentMetadata(filename) {
  return documentMetadata(readFileSync(filename, 'utf8'), filename);
}

export function documentSlugs(filename, slug) {
  const route = typeof slug === 'string'
    ? slug
    : filename.replace(/\.mdx?$/, '').replace(/(?:^|\/)index$/, '');
  return route.split('/').filter(Boolean);
}

/** Hand-written links come in three shapes: Markdown files, extensionless
 * pages (Docusaurus-era `./kanban`, `../configuration`) and non-document
 * assets. Only pages and Markdown files are app routes; assets keep the href
 * the author wrote. */
function linkTargetKind(pathPart) {
  if (/\.(?:md|mdx)$/i.test(pathPart)) return 'document';
  if (/\.[^./]+$/.test(pathPart)) return 'asset';
  return 'page';
}

/** A logical target lives in the author's locale first, then English; the
 * viewing locale is applied later, so one resolved route serves both trees. */
function findDocument(contentDir, locale, target) {
  const forms = /\.(?:md|mdx)$/i.test(target)
    ? [target]
    : [`${target}.md`, `${target}.mdx`, `${target}/index.md`, `${target}/index.mdx`];
  for (const base of [locale, 'en']) {
    for (const form of forms) {
      const file = path.join(contentDir, base, form);
      if (statSync(file, { throwIfNoEntry: false })?.isFile()) return { file, form };
    }
  }
  return null;
}

/** Resolve at build time; the MDX Link applies the active viewing locale. */
export function resolveDocumentLink(sourceFile, href, contentDir) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) return href;
  const [, pathPart = '', suffix = ''] = /^([^?#]*)([?#].*)?$/.exec(href);
  if (!pathPart) return href;
  const kind = linkTargetKind(pathPart);
  if (kind === 'asset') return href;
  // Absolute logical routes already work; only Markdown files rewrite those.
  if (kind === 'page' && pathPart.startsWith('/')) return href;
  const relativeSource = path.relative(contentDir, sourceFile).split(path.sep).join('/');
  const [locale, ...segments] = relativeSource.split('/');
  if (!['en', 'zh-Hans'].includes(locale)) throw new Error(`Document outside content locales: ${sourceFile}`);
  let target;
  try {
    const decoded = decodeURI(pathPart);
    target = decoded.startsWith('/')
      ? decoded.replace(/^\/(?:docs\/)?/, '')
      : path.posix.join(path.posix.dirname(segments.join('/')), decoded);
  } catch {
    throw new Error(`Link ${href} in ${relativeSource} has invalid percent-encoding`);
  }
  if (target.includes('\0')) throw new Error(`Link ${href} in ${relativeSource} contains a null byte`);
  // The normalized target must stay inside the author's locale tree: findDocument
  // rejoins with path.join, so a bare '..' or an un-normalized '/docs/a/../../x.md'
  // would otherwise be re-anchored onto files outside the locale before reading.
  // `./kanban/` names the same page as `./kanban`; drop the slash before the
  // `.md`/`.mdx` and `/index` candidates are built.
  if (kind === 'page') target = target.replace(/\/+$/, '');
  const normalized = path.posix.normalize(target);
  if (path.posix.isAbsolute(normalized) || normalized === '..' || normalized.startsWith('../')) {
    throw new Error(`Link ${href} in ${relativeSource} escapes the content directory`);
  }
  const found = findDocument(contentDir, locale, target);
  if (!found) throw new Error(`Unresolved document link ${href} in ${relativeSource}`);
  const data = readDocumentMetadata(found.file);
  return '/' + documentSlugs(found.form, data.slug).join('/') + suffix;
}
