#!/usr/bin/env node
// Keep source-level document links resolvable at build time: every relative
// Markdown or extensionless page link in content/ must map to a document,
// so unresolved pages fail before the export can ship them as 404s.

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkDirective from 'remark-directive';
import { frontmatter } from 'fumadocs-core/content/md/frontmatter';
import { resolveDocumentLink } from '../src/lib/content.mjs';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const parser = unified().use(remarkParse).use(remarkGfm).use(remarkDirective);

async function documents(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await documents(file));
    else if (/\.mdx?$/.test(entry.name)) result.push(file);
  }
  return result;
}

/**
 * Check every link and definition in an app's content tree (default: this
 * script's own apps/docs). Read-only; returns
 * `{ documents, links, errors }` and never throws for a broken link.
 */
export async function checkSourceLinks(appDir = appRoot) {
  const contentDir = path.join(appDir, 'content');
  const files = await documents(contentDir);
  const errors = [];
  let links = 0;
  for (const file of files) {
    const text = frontmatter(await readFile(file, 'utf8')).content;
    const pending = [parser.parse(text)];
    while (pending.length) {
      const node = pending.pop();
      if (node.type === 'link' || node.type === 'definition') {
        links++;
        try { resolveDocumentLink(file, node.url, contentDir); }
        catch (error) { errors.push(error.message); }
      }
      if (node.children) pending.push(...node.children);
    }
  }
  return { documents: files.length, links, errors };
}

if (import.meta.main) {
  const { values } = parseArgs({ options: { 'app-dir': { type: 'string' } } });
  const result = await checkSourceLinks(values['app-dir'] ? path.resolve(values['app-dir']) : appRoot);
  if (result.errors.length) {
    console.error(result.errors.join('\n'));
    process.exitCode = 1;
  }
  console.log(JSON.stringify({ documents: result.documents, links: result.links, errors: result.errors.length }));
}
