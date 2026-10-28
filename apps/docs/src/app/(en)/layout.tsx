import type { ReactNode } from 'react';
import { SiteRoot, siteMetadata } from '@/components/site-root';

export const metadata = siteMetadata('en');
export default function Layout({ children }: { children: ReactNode }) {
  return <SiteRoot locale="en">{children}</SiteRoot>;
}
