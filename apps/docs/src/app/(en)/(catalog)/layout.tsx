// Thin route-group layout for the English catalog pages. The parent owns the root
// layout (<html>/<body>, theme provider, fonts, global styles); this adapter only
// drops the pages into the shared Fumadocs shell with the right locale.
import { CatalogShell } from "@/components/site-shell";

export default function CatalogLayout({ children }: { children: React.ReactNode }) {
  return <CatalogShell locale="en">{children}</CatalogShell>;
}
