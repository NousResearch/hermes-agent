import type { ReactNode } from 'react';
import { defineTranslations } from 'fumadocs-core/i18n';
import { uiTranslations } from 'fumadocs-ui/i18n';
import { zhCN } from '@fumadocs/language/zh-cn';
import { SiteProvider } from './site-provider';
import { localizedHref, publicHref, type Locale } from '@/lib/urls';
import '@/styles/global.css';

const chineseUI = defineTranslations().extend(uiTranslations()).preset(zhCN()).get();

export function SiteRoot({ children, locale }: { children: ReactNode; locale: Locale }) {
  return (
    <html lang={locale} suppressHydrationWarning>
      {/* OpenSearch autodiscovery (Chrome/Firefox "add search engine"); React
          hoists this into <head> alongside the Next metadata. */}
      <link
        rel="search"
        type="application/opensearchdescription+xml"
        title="Hermes Agent"
        href={publicHref(localizedHref('/opensearch.xml', locale))}
      />
      <body className="flex min-h-screen flex-col">
        <SiteProvider locale={locale} translations={locale === 'zh-Hans' ? chineseUI : undefined}>{children}</SiteProvider>
      </body>
    </html>
  );
}
