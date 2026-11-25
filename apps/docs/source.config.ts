import path from 'node:path';
import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { pageSchema } from 'fumadocs-core/source/schema';
import { remarkDirectiveAdmonition, remarkHeading, remarkMdxMermaid, remarkStructure } from 'fumadocs-core/mdx-plugins';
import remarkDirective from 'remark-directive';
import rehypeRaw from 'rehype-raw';
import type { Pluggable } from 'unified';
import { z } from 'zod';
import { readDocumentMetadata } from './src/lib/content.mjs';
import { remarkAdmonitionLabels } from './src/remark/admonition-labels.mjs';
import { remarkCodeLanguages, remarkDocumentLinks } from './src/remark/docs.mjs';
import { remarkPreserveDirectives } from './src/remark/preserve-directives.mjs';
import { rehypeFenceMeta } from './src/rehype/fence-meta.mjs';

const contentDir = path.resolve('content');
// The vendor plugin uses `name in types`; inherited Object keys are not callouts.
const admonitionTypes = Object.assign(Object.create(null) as Record<string, string>, {
  note: 'info', tip: 'success', info: 'info', warning: 'warning', caution: 'warning',
  danger: 'error', important: 'info', success: 'success',
});

/** Consume the known admonition containers, then turn every remaining
 * directive back into its source text: remark-directive otherwise eats
 * colon-joined words like `9:00`, `1:1` or `host:9090` out of the page.
 * Add default labels before consuming callouts and preserve text before
 * the heading transform computes anchors. */
const directiveHandling: Pluggable[] = [
  remarkDirective,
  remarkAdmonitionLabels,
  [remarkDirectiveAdmonition, { types: admonitionTypes }],
  remarkPreserveDirectives,
];

function isPlugin(entry: Pluggable, plugin: unknown) {
  return Array.isArray(entry) ? entry[0] === plugin : entry === plugin;
}

/** The preset derives heading anchors (remarkHeading) from heading text
 * before user plugins run, so directive text must already be restored there
 * or anchors drop what the directives swallowed (deep links like the
 * iron-proxy `...bind-host:9090...` heading break). The other custom
 * plugins keep their default position before remarkStructure. */
function remarkChain(preset: Pluggable[]): Pluggable[] {
  const body: Pluggable[] = [remarkMdxMermaid, remarkCodeLanguages, [remarkDocumentLinks, { contentDir }]];
  const headAt = preset.findIndex((entry) => isPlugin(entry, remarkHeading));
  if (headAt === -1) return [...directiveHandling, ...body, ...preset];
  const withDirectives = [...preset.slice(0, headAt), ...directiveHandling, ...preset.slice(headAt)];
  const structureAt = withDirectives.findIndex((entry) => isPlugin(entry, remarkStructure));
  return structureAt === -1
    ? [...withDirectives, ...body]
    : [...withDirectives.slice(0, structureAt), ...body, ...withDirectives.slice(structureAt)];
}

export const docs = defineDocs({
  dir: 'content',
  docs: {
    dynamic: true,
    schema: ({ path: file }) => {
      const metadata = readDocumentMetadata(file);
      return pageSchema.extend({
        title: z.string().default(metadata.title),
        seo_title: z.string().optional(),
        description: z.string().optional().transform((value) => value?.trim() || metadata.description),
        hasH1: z.boolean().default(metadata.hasH1),
        sidebar_label: z.string().optional(),
        slug: z.string().optional(),
        hide_title: z.boolean().optional(),
        hide_table_of_contents: z.boolean().optional(),
      });
    },
  },
});

export default defineConfig({
  mdxOptions: {
    remarkImageOptions: false,
    remarkRehypeOptions: { allowDangerousHtml: true },
    remarkPlugins: remarkChain,
    rehypePlugins: (plugins) => [rehypeFenceMeta, [rehypeRaw, { passThrough: ['mdxjsEsm', 'mdxFlowExpression', 'mdxTextExpression', 'mdxJsxFlowElement', 'mdxJsxTextElement'] }], ...plugins],
    rehypeCodeOptions: { themes: { light: 'github-light', dark: 'github-dark' }, fallbackLanguage: 'text' },
  },
});
