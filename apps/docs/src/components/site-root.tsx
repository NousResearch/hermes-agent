import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import { defineTranslations } from 'fumadocs-core/i18n';
import { uiTranslations } from 'fumadocs-ui/i18n';
import { zhCN } from '@fumadocs/language/zh-cn';
import { SiteProvider } from './site-provider';
import { publicHref, SITE_URL, type Locale } from '@/lib/urls';
import '@/styles/global.css';

const chineseUI = defineTranslations().extend(uiTranslations()).preset(zhCN()).get();

export function siteMetadata(locale: Locale): Metadata {
  return {
    metadataBase: new URL(SITE_URL),
    title: { default: 'Hermes Agent Documentation', template: '%s | Hermes Agent' },
    description: 'The self-improving AI agent built by Nous Research.',
    icons: { icon: publicHref('/img/favicon.ico'), apple: publicHref('/img/apple-touch-icon.png') },
    openGraph: { siteName: 'Hermes Agent', images: [publicHref('/img/hermes-agent-banner.png')] },
    // Preserve the existing DocSearch crawler facets during the renderer cutover.
    other: {
      'docsearch:language': locale,
      'docsearch:version': 'current',
      'docsearch:docusaurus_tag': 'docs-default-current',
      docusaurus_locale: locale,
      docusaurus_version: 'current',
      docusaurus_tag: 'docs-default-current',
    },
  };
}

export function SiteRoot({ children, locale }: { children: ReactNode; locale: Locale }) {
  return (
    <html lang={locale} suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">
        <SiteProvider locale={locale} translations={locale === 'zh-Hans' ? chineseUI : undefined}>{children}</SiteProvider>
      </body>
    </html>
  );
}
