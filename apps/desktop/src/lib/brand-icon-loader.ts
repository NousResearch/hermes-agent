import { useEffect, useState } from 'react'

import type { BrandIcon } from './brand-icon'

export type { BrandIcon } from './brand-icon'

interface BrandIconModule {
  resolveBrandIcon: (hostname: string) => BrandIcon | null
}

type BrandIconImporter = () => Promise<BrandIconModule>

export interface BrandIconLoader {
  load: (hostname: string) => Promise<BrandIcon | null>
  peek: (hostname: string) => BrandIcon | null
}

// This string-only admission table is intentionally kept separate from the
// Simple Icons components. It lets unknown links stay cheap while the actual
// SVG catalog remains behind a dynamic import.
export const DEFERRED_BRAND_ICON_HOSTNAMES =
  `github.com github.io githubusercontent.com gitlab.com gitlab.io bitbucket.org codeberg.org gitea.com gitea.io
forgejo.org sr.ht npmjs.com pypi.org crates.io huggingface.co hf.co stackoverflow.com stackexchange.com
serverfault.com superuser.com developer.mozilla.org readthedocs.io readthedocs.org wikipedia.org archive.org
arxiv.org semanticscholar.org scholar.google.com researchgate.net zotero.org overleaf.com anthropic.com
claude.ai gemini.google.com perplexity.ai openrouter.ai mistral.ai ollama.com replicate.com wandb.ai
kaggle.com x.com twitter.com t.co reddit.com redd.it news.ycombinator.com bsky.app mastodon.social
joinmastodon.org facebook.com instagram.com tiktok.com pinterest.com discord.com discord.gg t.me telegram.org
producthunt.com crunchbase.com quora.com medium.com substack.com dev.to hashnode.dev hashnode.com
wordpress.com wordpress.org ghost.org youtube.com youtu.be vimeo.com twitch.tv soundcloud.com spotify.com
netflix.com imdb.com goodreads.com unsplash.com behance.net store.steampowered.com itch.io linear.app
notion.so notion.site figma.com atlassian.net atlassian.com confluence.com asana.com trello.com obsidian.md
miro.com excalidraw.com tldraw.com zoom.us google.com goo.gl docs.google.com drive.google.com mail.google.com
maps.google.com openstreetmap.org vercel.com vercel.app netlify.com netlify.app cloudflare.com pages.dev
workers.dev railway.app render.com digitalocean.com firebase.google.com supabase.com docker.com kubernetes.io
grafana.com datadoghq.com sentry.io snyk.io codecov.io circleci.com jenkins.io python.org nodejs.org react.dev
typescriptlang.org tailwindcss.com vite.dev vitejs.dev rust-lang.org go.dev golang.org ruby-lang.org
rubyonrails.org php.net laravel.com djangoproject.com flask.palletsprojects.com fastapi.tiangolo.com swift.org
kotlinlang.org deno.com bun.sh pnpm.io turborepo.com eslint.org storybook.js.org graphql.org prisma.io
postgresql.org redis.io mongodb.com pytorch.org tensorflow.org scikit-learn.org jupyter.org threejs.org
blender.org godotengine.org unity.com electronjs.org tauri.app brew.sh archlinux.org debian.org ubuntu.com
raspberrypi.com nvidia.com codesandbox.io stackblitz.com replit.com cursor.com windsurf.com zed.dev warp.dev
raycast.com postman.com stripe.com paypal.com patreon.com ko-fi.com buymeacoffee.com shopify.com webflow.com
dropbox.com leetcode.com coursera.org udemy.com yelp.com`.split(/\s+/)

const BRAND_ICON_HOSTS = new Set(DEFERRED_BRAND_ICON_HOSTNAMES)

function normalizedHost(hostname: string): string {
  return hostname.trim().toLowerCase().replace(/^www\./, '')
}

export function hasBrandIconHost(hostname: string): boolean {
  const host = normalizedHost(hostname)

  if (!host.includes('.')) {
    return false
  }

  const parts = host.split('.')

  for (let i = 0; i < parts.length - 1; i += 1) {
    if (BRAND_ICON_HOSTS.has(parts.slice(i).join('.'))) {
      return true
    }
  }

  return false
}

export function createBrandIconLoader(
  importer: BrandIconImporter = () => import('./brand-icon')
): BrandIconLoader {
  let completed: BrandIconModule | undefined
  let pending: Promise<BrandIconModule> | undefined

  const peek = (hostname: string) => {
    if (!hasBrandIconHost(hostname) || !completed) {
      return null
    }

    return completed.resolveBrandIcon(hostname)
  }

  return {
    async load(hostname) {
      if (!hasBrandIconHost(hostname)) {
        return null
      }

      if (completed) {
        return completed.resolveBrandIcon(hostname)
      }

      if (!pending) {
        const operation = importer().then(module => {
          completed = module
          pending = undefined

          return module
        })

        pending = operation
      }

      const module = await pending

      return module.resolveBrandIcon(hostname)
    },
    peek
  }
}

export const brandIconLoader = createBrandIconLoader()

export function useBrandIcon(
  hostname: string,
  loader: BrandIconLoader = brandIconLoader
): BrandIcon | null {
  const eligible = hasBrandIconHost(hostname)
  const cached = eligible ? loader.peek(hostname) : null
  const [loadedHostname, setLoadedHostname] = useState<string | null>(() => (cached ? hostname : null))
  const icon = cached ?? (loadedHostname === hostname ? loader.peek(hostname) : null)

  useEffect(() => {
    if (!eligible || icon) {
      return undefined
    }

    let active = true

    void loader
      .load(hostname)
      .then(() => {
        if (active) {
          setLoadedHostname(hostname)
        }
      })
      .catch(() => undefined)

    return () => {
      active = false
    }
  }, [eligible, hostname, icon, loader])

  return icon
}
