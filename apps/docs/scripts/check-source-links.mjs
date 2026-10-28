import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkDirective from 'remark-directive';
import { frontmatter } from 'fumadocs-core/content/md/frontmatter';
import { resolveDocumentLink } from '../src/lib/content.mjs';

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const contentDir = path.join(appDir, 'content');
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
if (errors.length) {
  console.error(errors.join('\n'));
  process.exitCode = 1;
}
console.log(JSON.stringify({ documents: files.length, links, errors: errors.length }));
