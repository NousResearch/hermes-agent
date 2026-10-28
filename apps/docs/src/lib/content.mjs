import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { parse } from 'yaml';

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

/** Resolve at build time; the MDX Link applies the active viewing locale. */
export function resolveDocumentLink(sourceFile, href, contentDir) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) return href;
  const match = /^([^?#]+\.mdx?)([?#].*)?$/i.exec(href);
  if (!match) return href;
  const relativeSource = path.relative(contentDir, sourceFile).split(path.sep).join('/');
  const [locale, ...segments] = relativeSource.split('/');
  if (!['en', 'zh-Hans'].includes(locale)) throw new Error(`Document outside content locales: ${sourceFile}`);
  const target = match[1].startsWith('/')
    ? match[1].replace(/^\/(?:docs\/)?/, '')
    : path.posix.join(path.posix.dirname(segments.join('/')), decodeURI(match[1]));
  if (target.startsWith('../')) return href;
  const candidates = [path.join(contentDir, locale, target), path.join(contentDir, 'en', target)];
  const filename = candidates.find(existsSync);
  if (!filename) throw new Error(`Unresolved Markdown link ${href} in ${relativeSource}`);
  const data = readDocumentMetadata(filename);
  return '/' + documentSlugs(target, data.slug).join('/') + (match[2] ?? '');
}
