import React from "react";
import Link from "next/link";
import {
  SUBMIT_PLUGIN_URL,
  authorPagePath,
  categoryOf,
  desktopInstallLink,
  formatDate,
  formatStars,
  pinUrl,
  repoUrl,
  splitDisclosure,
  tierOf,
} from "./catalog";
import { type Locale, localizedHref } from "@/lib/urls";
import type { PluginPageData } from "@/lib/plugin-pages";
import CopyButton from "./CopyButton";
import MiniCard from "./MiniCard";
import styles from "./pages.module.css";

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={styles.fact}>
      <dt className={styles.factLabel}>{label}</dt>
      <dd className={styles.factValue}>{children}</dd>
    </div>
  );
}

function ChipList({ label, items, mono = true }: { label: string; items?: string[]; mono?: boolean }) {
  if (!items?.length) return null;
  return (
    <div className={styles.capRow}>
      <h3 className={styles.capLabel}>
        {label} <span className={styles.count}>{items.length}</span>
      </h3>
      <div className={styles.chipWrap}>
        {items.map((item) =>
          mono ? (
            <code key={item} className={styles.chip}>{item}</code>
          ) : (
            <span key={item} className={styles.chip}>{item}</span>
          ),
        )}
      </div>
    </div>
  );
}

function platformLabel(p: string): string {
  return p === "macos" ? "macOS" : p === "linux" ? "Linux" : p === "windows" ? "Windows" : p;
}

// Rendered at build time (static export) by the route adapters under
// src/app/(en)/(catalog)/plugins/[name] and src/app/zh-Hans/(catalog)/plugins/[name],
// with data from src/lib/plugin-pages.ts. A server component: no hooks here; the
// copy button is the one client island on the page.
export default function PluginPage({ data, locale }: { data: PluginPageData; locale: Locale }) {
  const { plugin, readmeHtml, author, moreByAuthor } = data;
  const tier = tierOf(plugin);
  const category = categoryOf(plugin);
  const caps = plugin.capabilities || {};
  const { prose, disclosures } = splitDisclosure(plugin.description || "");
  const repo = repoUrl(plugin);
  const pinned = pinUrl(plugin);
  const readmeFileUrl = data.readmeSourceUrl
    ? data.readmeSourceUrl
        .replace(/^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([0-9a-f]{40})\//, "https://github.com/$1/$2/blob/$3/")
        .replace("/-/raw/", "/-/blob/")
    : null;
  const added = formatDate(plugin.addedAt);
  const updated = formatDate(plugin.updatedAt);

  return (
    <div className={styles.page}>
      <nav className={styles.crumbs} aria-label="Breadcrumb">
        <Link href={localizedHref("/plugins", locale)}>Plugin Catalog</Link>
          <span aria-hidden="true">/</span>
          <span>{category.label}</span>
          <span aria-hidden="true">/</span>
          <span className={styles.crumbCurrent}>{plugin.name}</span>
        </nav>

        <header className={styles.hero}>
          {plugin.image && (
            <img
              className={styles.heroImage}
              src={plugin.image}
              alt=""
              decoding="async"
              referrerPolicy="no-referrer"
            />
          )}
          <div className={styles.heroBody}>
            <div className={styles.titleRow}>
              <span className={styles.heroIcon} aria-hidden="true" title={category.label}>
                {category.icon}
              </span>
              <h1 className={styles.title}>{plugin.name}</h1>
              <span
                className={styles.tierPill}
                style={{ color: tier.color, background: tier.bg, borderColor: tier.border }}
              >
                {tier.icon} {tier.label}
              </span>
              {plugin.version && (
                <span className={styles.versionPill} title={`Version ${plugin.version} at ${plugin.sha}`}>
                  v{plugin.version.replace(/^v/i, "")}
                </span>
              )}
              {typeof plugin.stars === "number" && (
                <a
                  className={styles.starPill}
                  href={`${repo}/stargazers`}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={`${plugin.stars.toLocaleString()} GitHub stars${data.starsFetchedAt ? ` as of ${data.starsFetchedAt}` : ""}`}
                >
                  {"\u2605"} {formatStars(plugin.stars)}
                </a>
              )}
            </div>
            <p className={styles.byline}>
              {category.label}
              {plugin.maintainer && (
                <>
                  {" · by "}
                  {author ? (
                    <Link className={styles.authorLink} href={localizedHref(authorPagePath(author.slug), locale)}>
                      {plugin.maintainer}
                    </Link>
                  ) : (
                    plugin.maintainer
                  )}
                </>
              )}
              {added && <> {" · added "}<time dateTime={plugin.addedAt || undefined}>{added}</time></>}
              {updated && updated !== added && (
                <> {" · updated "}<time dateTime={plugin.updatedAt || undefined}>{updated}</time></>
              )}
            </p>
            {prose && <p className={styles.lede}>{prose}</p>}

            <div className={styles.actions}>
              <a
                className={styles.installBtn}
                href={desktopInstallLink(plugin.name)}
                title="Opens the Install Plugin dialog in Hermes Desktop at the reviewed version. No app? Use the install command."
              >
                Open in Hermes Desktop
              </a>
              <div className={styles.installCmd}>
                <code>{plugin.installCommand}</code>
                <CopyButton text={plugin.installCommand} />
              </div>
            </div>
            <div className={styles.links}>
              <a className={styles.link} href={repo} target="_blank" rel="noopener noreferrer">Repository ↗</a>
              <a className={styles.link} href={pinned} target="_blank" rel="noopener noreferrer" title={plugin.sha}>
                Reviewed source @ {plugin.shaShort} ↗
              </a>
              {plugin.docsUrl && (
                <a className={styles.link} href={plugin.docsUrl} target="_blank" rel="noopener noreferrer">Documentation ↗</a>
              )}
            </div>
          </div>
        </header>

        <main className={styles.main}>
          <div className={styles.columns}>
            <div className={styles.primary}>
              {plugin.screenshots?.length ? (
                <section className={styles.section} aria-labelledby="screenshots">
                  <h2 id="screenshots" className={styles.sectionTitle}>Screenshots</h2>
                  <div className={styles.gallery}>
                    {plugin.screenshots.map((src, i) => (
                      <a key={src} href={src} target="_blank" rel="noopener noreferrer" className={styles.shot}>
                        <img
                          src={src}
                          alt={`${plugin.name} screenshot ${i + 1}`}
                          loading="lazy"
                          decoding="async"
                          referrerPolicy="no-referrer"
                        />
                      </a>
                    ))}
                  </div>
                </section>
              ) : null}

              {(caps.providesTools?.length || caps.providesHooks?.length || caps.providesMiddleware?.length || caps.requiresEnv?.length) ? (
                <section className={styles.section} aria-labelledby="capabilities">
                  <h2 id="capabilities" className={styles.sectionTitle}>What it adds</h2>
                  <ChipList label="Tools" items={caps.providesTools} />
                  <ChipList label="Hooks" items={caps.providesHooks} />
                  <ChipList label="Middleware" items={caps.providesMiddleware} />
                  <ChipList label="Environment variables it needs" items={caps.requiresEnv} />
                </section>
              ) : null}

              {readmeHtml ? (
                <section className={styles.section} aria-labelledby="readme">
                  <div className={styles.readmeHead}>
                    <h2 id="readme" className={styles.sectionTitle}>README</h2>
                    <span className={styles.readmeNote}>
                      From the reviewed commit{" "}
                      {readmeFileUrl ? (
                        <a href={readmeFileUrl} target="_blank" rel="noopener noreferrer"><code>{plugin.shaShort}</code> ↗</a>
                      ) : (
                        <code>{plugin.shaShort}</code>
                      )}
                      ; it updates when the author re-pins.
                    </span>
                  </div>
                  <article
                    className={`${styles.readme} prose`}
                    // Build-time output of scripts/lib/plugin-readme.mjs: raw HTML dropped,
                    // tags/attributes allowlisted, URLs restricted to http(s)/mailto/#.
                    dangerouslySetInnerHTML={{ __html: readmeHtml }}
                  />
                </section>
              ) : null}
            </div>

            <aside className={styles.side}>
              <dl className={styles.facts}>
                <Fact label="Pinned commit">
                  <a href={pinned} target="_blank" rel="noopener noreferrer" title={plugin.sha}>
                    <code>{plugin.shaShort}</code> ↗
                  </a>
                </Fact>
                {plugin.version && <Fact label="Version"><code>{plugin.version}</code></Fact>}
                {plugin.subdir && <Fact label="Path in repo"><code>{plugin.subdir}</code></Fact>}
                <Fact label="Requires">
                  <code>hermes {plugin.requiresHermes || "(any version)"}</code>
                </Fact>
                <Fact label="Platforms">
                  {plugin.platforms?.length ? plugin.platforms.map(platformLabel).join(", ") : "Linux, macOS, Windows"}
                </Fact>
                <Fact label="Category">
                  <Link href={localizedHref("/plugins", locale)}>{category.label}</Link>
                </Fact>
                <Fact label="Tier">{tier.label}</Fact>
                {plugin.maintainer && (
                  <Fact label="Maintainer">
                    {author ? <Link href={localizedHref(authorPagePath(author.slug), locale)}>{plugin.maintainer}</Link> : plugin.maintainer}
                  </Fact>
                )}
                {added && <Fact label="Listed">{added}</Fact>}
                {updated && updated !== added && <Fact label="Last re-pin">{updated}</Fact>}
              </dl>
              <p className={styles.sideNote}>
                Every entry installs exactly the commit above. Updates reach users only through a
                reviewed re-pin PR — <Link href={localizedHref(SUBMIT_PLUGIN_URL, locale)}>how listing works</Link>.
              </p>
            </aside>
          </div>

          {disclosures.length > 0 && (
            <section className={styles.section} aria-labelledby="disclosure">
              <h2 id="disclosure" className={styles.sectionTitle}>Disclosure</h2>
              {disclosures.map((text) => (
                <aside key={text} className={styles.disclosure} role="note">
                  {text}
                </aside>
              ))}
            </section>
          )}

          {author && moreByAuthor.length > 0 && (
            <section className={styles.section} aria-labelledby="more-by">
              <div className={styles.shelfHead}>
                <h2 id="more-by" className={styles.sectionTitle}>More by {author.name}</h2>
                <Link className={styles.shelfAll} href={localizedHref(authorPagePath(author.slug), locale)}>
                  All {author.count} plugins by {author.name} →
                </Link>
              </div>
              <div className={styles.shelf}>
                {moreByAuthor.slice(0, 6).map((p) => <MiniCard key={p.name} plugin={p} locale={locale} />)}
              </div>
            </section>
          )}

          <p className={styles.footerMeta}>
            <Link href={localizedHref("/plugins", locale)}>← Back to the catalog</Link>
            {formatDate(data.generatedAt) && (
              // Absolute date, not "x hours ago": the page is server-rendered and a clock-relative
              // string would differ at hydration.
              <span title={data.generatedAt || undefined}> · catalog built {formatDate(data.generatedAt)}</span>
            )}
          </p>
        </main>
      </div>
  );
}
