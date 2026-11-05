// English full-page search results (/search → /docs/search with the
// app-wide basePath). The heading and metadata prerender statically; the
// URL-driven results hydrate from <SearchPage>, which reads and writes ?q= so
// direct links, reloads and OpenSearch browser engines all work.
import { Suspense } from 'react';
import type { Metadata } from 'next';
import { catalogAlternates } from '@/lib/catalog-routes';
import { SearchPage } from '@/components/search-page';

export const metadata: Metadata = {
  alternates: catalogAlternates('/search', 'en'),
  title: 'Search',
  description: 'Search the Hermes Agent documentation.',
};

export default function Page() {
  return (
    <section className="mx-auto w-full max-w-3xl px-4 py-10 sm:px-6">
      <h1 className="text-3xl font-semibold text-fd-foreground">Search the documentation</h1>
      <Suspense fallback={null}>
        <SearchPage locale="en" />
      </Suspense>
    </section>
  );
}
