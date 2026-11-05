import type { ReactNode } from 'react';
import { DocumentShell } from '@/components/site-shell';

export default function Layout({ children }: { children: ReactNode }) {
  return <DocumentShell locale="en">{children}</DocumentShell>;
}
