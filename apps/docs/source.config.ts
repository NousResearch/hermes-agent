import path from 'node:path';
import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { pageSchema } from 'fumadocs-core/source/schema';
import { remarkDirectiveAdmonition, remarkMdxMermaid } from 'fumadocs-core/mdx-plugins';
import remarkDirective from 'remark-directive';
import rehypeRaw from 'rehype-raw';
import { z } from 'zod';
import { readDocumentMetadata } from './src/lib/content.mjs';
import { remarkCodeLanguages, remarkDocumentLinks } from './src/remark/docs.mjs';

const contentDir = path.resolve('content');

export const docs = defineDocs({
  dir: 'content',
  docs: {
    dynamic: true,
    schema: ({ path: file }) => {
      const metadata = readDocumentMetadata(file);
      return pageSchema.extend({
        title: z.string().default(metadata.title),
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
    remarkPlugins: [
      remarkDirective,
      [remarkDirectiveAdmonition, { types: { note: 'info', tip: 'info', info: 'info', warning: 'warning', caution: 'warning', danger: 'error', important: 'warning', success: 'success' } }],
      remarkMdxMermaid,
      remarkCodeLanguages,
      [remarkDocumentLinks, { contentDir }],
    ],
    rehypePlugins: (plugins) => [[rehypeRaw, { passThrough: ['mdxjsEsm', 'mdxFlowExpression', 'mdxTextExpression', 'mdxJsxFlowElement', 'mdxJsxTextElement'] }], ...plugins],
    rehypeCodeOptions: { themes: { light: 'github-light', dark: 'github-dark' }, fallbackLanguage: 'text' },
  },
});
