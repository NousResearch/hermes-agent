import React from "react";
import Link from "next/link";
import { type CatalogPlugin, CATEGORY_ORDER, categoryOf, formatDate, formatStars, repoUrl } from "./catalog";
import { type Locale, localizedHref } from "@/lib/urls";
import type { AuthorPageData } from "@/lib/plugin-pages";
import MiniCard from "./MiniCard";
import styles from "./pages.module.css";

/** GitHub/GitLab profile when every listed repo belongs to one owner; null when they differ. */
function profileUrl(plugins: CatalogPlugin[]): string | null {
  const owners = new Set<string>();
  for (const p of plugins) {
    const m = repoUrl(p).match(/^https:\/\/(github\.com|gitlab\.com)\/([^/]+)\//);
    if (!m) return null;
    owners.add(`https://${m[1]}/${m[2]}`);
  }
  return owners.size === 1 ? [...owners][0] : null;
}

// Rendered at build time (static export) by the route adapters under
// src/app/(en)/(catalog)/plugins/by/[author] and the zh-Hans equivalent, with data
// from src/lib/plugin-pages.ts. A server component (no hooks).
export default function AuthorPage({ data, locale }: { data: AuthorPageData; locale: Locale }) {
  const { author, plugins } = data;
  const stars = plugins.reduce((sum, p) => sum + (typeof p.stars === "number" ? p.stars : 0), 0);
  const hasStars = plugins.some((p) => typeof p.stars === "number");
  const profile = profileUrl(plugins);
  const firstListed = plugins
    .map((p) => p.addedAt)
    .filter((d): d is string => Boolean(d))
    .sort()[0];
  const categories = CATEGORY_ORDER.filter((c) => plugins.some((p) => categoryOf(p) === categoryOf({ category: c })));
  const official = plugins.filter((p) => p.tier === "official").length;

  return (
    <div className={styles.page}>
      <nav className={styles.crumbs} aria-label="Breadcrumb">
        <Link href={localizedHref("/plugins", locale)}>Plugin Catalog</Link>
        <span aria-hidden="true">/</span>
        <span>Authors</span>
        <span aria-hidden="true">/</span>
        <span className={styles.crumbCurrent}>{author.name}</span>
      </nav>

      <header className={`${styles.hero} ${styles.authorHero}`}>
        <div className={styles.heroBody}>
          <p className={styles.eyebrow}>Plugin author</p>
          <h1 className={styles.title}>{author.name}</h1>
          <p className={styles.byline}>
            {plugins.length} plugin{plugins.length === 1 ? "" : "s"} in the catalog
            {official > 0 && <> · {official} official</>}
            {hasStars && <> · {"\u2605"} {formatStars(stars)} GitHub stars across them</>}
            {firstListed && <> · first listed {formatDate(firstListed)}</>}
            {categories.length > 0 && (
              <> · {categories.map((c) => categoryOf({ category: c }).label).join(", ")}</>
            )}
          </p>
          {profile && (
            <div className={styles.links}>
              <a className={styles.link} href={profile} target="_blank" rel="noopener noreferrer">
                {profile.replace(/^https:\/\//, "")} ↗
              </a>
            </div>
          )}
        </div>
      </header>

      <main className={styles.main}>
        <div className={styles.shelf}>
          {plugins.map((p) => <MiniCard key={p.name} plugin={p} locale={locale} />)}
        </div>
        <p className={styles.footerMeta}>
          <Link href={localizedHref("/plugins", locale)}>← Back to the catalog</Link>
        </p>
      </main>
    </div>
  );
}
