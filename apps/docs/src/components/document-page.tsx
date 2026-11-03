import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { DocsPage, DocsBody, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { source } from '@/lib/source';
import type { Locale } from '@/lib/urls';
import { documentPageMetadata } from '@/lib/document-metadata';
import { getMDXComponents } from '@/mdx-components';

export function documentParams(locale: Locale) {
  return source.getPages(locale).map((page) => ({ slug: page.slugs }));
}

export function documentMetadata(slugs: string[] | undefined, locale: Locale): Metadata {
  const page = source.getPage(slugs, locale);
  if (!page) notFound();
  return documentPageMetadata(page, {
    en: source.getPage(page.slugs, 'en'),
    'zh-Hans': source.getPage(page.slugs, 'zh-Hans'),
  });
}

export async function DocumentPage({ slugs, locale }: { slugs?: string[]; locale: Locale }) {
  const page = source.getPage(slugs, locale);
  if (!page) notFound();
  const content = await page.data.load();
  const Body = content.body;
  const toc = page.data.hide_table_of_contents ? [] : content.toc.filter((item) => item.depth >= 2 && item.depth <= 3);
  const editUrl = `https://github.com/NousResearch/hermes-agent/edit/main/apps/docs/content/${page.path}`;
  return (
    <DocsPage toc={toc} full={page.data.hide_table_of_contents}>
      {!page.data.hasH1 && !page.data.hide_title && <DocsTitle>{page.data.title}</DocsTitle>}
      <DocsBody className="theme-doc-markdown markdown"><Body components={getMDXComponents()} /></DocsBody>
      <a href={editUrl} target="_blank" rel="noreferrer noopener" className="text-sm text-fd-muted-foreground hover:text-fd-foreground">Edit this page on GitHub</a>
    </DocsPage>
  );
}
