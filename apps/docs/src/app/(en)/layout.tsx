import type { ReactNode } from 'react';
import { SiteRoot } from '@/components/site-root';
import { siteMetadata } from '@/lib/metadata-defaults';

export const metadata = siteMetadata('en');
export default function Layout({ children }: { children: ReactNode }) {
  return <SiteRoot locale="en">{children}</SiteRoot>;
}
