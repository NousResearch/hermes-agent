import type { ReactNode } from 'react';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { layoutOptions } from '@/lib/layout-options';
import { getNavigationTree } from '@/lib/navigation';
import type { Locale } from '@/lib/urls';

interface ShellProps { children: ReactNode; locale: Locale }

export function DocumentShell({ children, locale }: ShellProps) {
  return <DocsLayout {...layoutOptions(locale)} tree={getNavigationTree(locale)} sidebar={{ collapsible: true, defaultOpenLevel: 0, prefetch: false }}>{children}</DocsLayout>;
}

export function CatalogShell({ children, locale }: ShellProps) {
  return <HomeLayout {...layoutOptions(locale)}>{children}</HomeLayout>;
}
