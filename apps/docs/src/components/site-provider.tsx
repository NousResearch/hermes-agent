"use client";

import type { ReactNode } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { DocSearchDialog } from './doc-search';
import { localizedHref, type Locale } from '@/lib/urls';

export function SiteProvider({ children, locale, translations }: { children: ReactNode; locale: Locale; translations?: Record<string, string> }) {
  const pathname = usePathname();
  const router = useRouter();
  return (
    <RootProvider
      theme={{ defaultTheme: 'dark', enableSystem: true }}
      search={{ SearchDialog: DocSearchDialog }}
      i18n={{
        locale,
        translations,
        defaultLanguage: 'en',
        hideLocale: 'default-locale',
        locales: [{ locale: 'en', name: 'English' }, { locale: 'zh-Hans', name: '简体中文' }],
        onLocaleChange: (next) => router.push(localizedHref(pathname + window.location.search + window.location.hash, next === 'zh-Hans' ? next : 'en')),
      }}
    >
      {children}
    </RootProvider>
  );
}
