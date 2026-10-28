// Thin route-group layout for the zh-Hans catalog pages. The parent owns the
// zh-Hans root layout (<html lang="zh-Hans">, theme provider, global styles);
// this adapter only drops the pages into the shared Fumadocs shell.
import { CatalogShell } from "@/components/site-shell";

export default function CatalogLayout({ children }: { children: React.ReactNode }) {
  return <CatalogShell locale="zh-Hans">{children}</CatalogShell>;
}
