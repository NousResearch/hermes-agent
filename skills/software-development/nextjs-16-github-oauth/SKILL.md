---
name: nextjs-16-github-oauth
description: GitHub-only OAuth pattern for Next.js 16 with NextAuth v5. Simplifies auth flow for dev tools, avoids multi-provider maintenance. Covers setup, route group conflicts, Vercel deployment, and LGPD compliance integration.
tags: [nextjs, auth, nextauth, github-oauth, oauth, typescript, vercel]
related_skills: [lgpd-compliance, nextjs-16-deployment]
---

# GitHub-Only OAuth — Next.js 16 + NextAuth v5

> **Pattern from HireMe Agent (June 2026):** User preference "é o padrão" — GitHub OAuth is the standard for dev tools. Single provider reduces maintenance, simplifies consent, and fits developer audience.

## When to Use

- Dev-focused SaaS applications
- Tools targeting GitHub users (repositories, issues, PRs)
- Simple auth flow without social login fragmentation
- LGPD compliance simplification (single data processor)

## Prerequisites

- Next.js 16 (App Router)
- NextAuth v5 (beta): `npm install next-auth@beta`
- GitHub OAuth App configured
- Tailwind v4 (optional, for styling)

## GitHub OAuth App Setup

### Create OAuth App

1. Go to https://github.com/settings/developers
2. Click "New OAuth App"
3. Configure:
   - **Application name**: `Your App Name`
   - **Homepage URL**: `https://your-app.vercel.app` (production)
   - **Authorization callback URL**: `https://your-app.vercel.app/api/auth/callback/github`
4. Save `Client ID` and generate `Client Secret`

**Critical:** Callback URL MUST match exactly:
- ✅ `https://your-app.vercel.app/api/auth/callback/github`
- ❌ `https://your-app.vercel.app/api/auth/callback/github/` (trailing slash)
- ❌ `http://localhost:3000/api/auth/callback/github` (for production)

### Environment Variables

```bash
# .env.local (development)
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-here (min 32 chars)
GITHUB_CLIENT_ID=Ov23liTqTBgtWy3UZqw9
GITHUB_CLIENT_SECRET=ghs_xxx... (from GitHub OAuth app)
```

```bash
# Vercel (production) — add in dashboard
NEXTAUTH_URL=https://your-app.vercel.app
NEXTAUTH_SECRET=<openssl rand -base64 32>
GITHUB_CLIENT_ID=same-as-dev
GITHUB_CLIENT_SECRET=same-as-dev
```

**Generate AUTH_SECRET:**
```bash
openssl rand -base64 32
```

**Pitfall: AUTH_SECRET Too Short**

**Symptom:** NextAuth errors about secret length

**Cause:** `AUTH_SECRET` must be at least 32 characters (256 bits)

**Fix:** Generate properly:
```bash
openssl rand -base64 32
# Example output: wuVEdgjREPPVRaOYfZcL5NOadchqt6z0qC8833ZaRKU=
```

**Never:** Use hardcoded secrets, short strings, or placeholder values.

## NextAuth v5 Implementation

### Auth Config

**File:** `src/lib/auth.ts`

```typescript
import NextAuth from "next-auth"
import GitHub from "next-auth/providers/github"

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    GitHub({
      clientId: process.env.GITHUB_CLIENT_ID,
      clientSecret: process.env.GITHUB_CLIENT_SECRET,
    }),
  ],
  secret: process.env.AUTH_SECRET,
  pages: {
    signIn: "/login",
  },
  callbacks: {
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.sub as string
        session.user.email = token.email as string
        session.user.name = token.name as string
        session.user.image = token.picture as string
      }
      return session
    },
  },
})
```

### API Route Handler

**File:** `src/app/api/auth/[...nextauth]/route.ts`

```typescript
import { handlers } from "@/lib/auth"

export const { GET, POST } = handlers
```

### Login Page

**File:** `src/app/login/page.tsx` or `src/app/[locale]/(landing)/login/page.tsx` (with i18n)

**IMPORTANT:** Lucide React exports are case-sensitive — use `Github` (lowercase 'i'), NOT `GitHub`.

```typescript
"use client"

import { signIn } from "next-auth/react"
import { Github } from "lucide-react"  // Note: lowercase 'i'
import { useState } from "react"

export default function LoginPage() {
  const [loading, setLoading] = useState(false)

  const handleGitHubLogin = async () => {
    setLoading(true)
    try {
      await signIn("github", { callbackUrl: "/" })
    } catch (error) {
      console.error("Login error:", error)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-slate-900 rounded-xl p-8 border border-slate-800">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white">Bem-vindo</h1>
          <p className="text-slate-400 mt-2">Entre com sua conta do GitHub</p>
        </div>

        <button
          onClick={handleGitHubLogin}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 bg-slate-800 hover:bg-slate-700 text-white py-3 rounded-lg border border-slate-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <svg className="animate-spin w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : (
            <Github className="w-5 h-5" />  // Lucide React - lowercase 'i'
          )}
          <span>Entrar com GitHub</span>
        </button>

        <div className="mt-6 text-center text-sm text-slate-500">
          Ao entrar, você concorda com nossos{" "}
          <a href="/terms" className="text-slate-400 hover:text-slate-300 underline">
            Termos de Uso
          </a>
          {" e "}
          <a href="/privacy" className="text-slate-400 hover:text-slate-300 underline">
            Política de Privacidade
          </a>
        </div>
      </div>
    </div>
  )
}
```

**Pattern:** Login page links to BOTH `/terms` and `/privacy` for LGPD compliance.

**Lucide Import Pitfall:**
```typescript
// ❌ WRONG - capital 'I'
import { GitHub } from "lucide-react"
// Error: "Export GitHub doesn't exist in target module. Did you mean Github?"

// ✅ CORRECT - lowercase 'i'
import { Github } from "lucide-react"
```

## Route Protection (Optional)

### Middleware for Protected Routes

**File:** `src/middleware.ts`

```typescript
import { auth } from "@/lib/auth"

export default auth((req) => {
  const isLoggedIn = !!req.auth
  const isOnDashboard = req.nextUrl.pathname.startsWith("/dashboard")

  if (isOnDashboard && !isLoggedIn) {
    return Response.redirect(new URL("/login", req.nextUrl))
  }
})

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}
```

**Note:** Next.js 16 prefers `proxy.ts` convention over `middleware.ts`. Check `nextjs-16-migration` skill for details.

## Common Pitfalls

### ❌ Missing Route Handler (500 Error on `/api/auth/github`)

**Symptom:** 
- Error 500 when visiting `/api/auth/github`
- Browser console: "The redirect_uri is not associated with this application"
- NextAuth not initialized properly

**Cause:** NextAuth route handler not exported correctly from route file

**Fix:** NextAuth v5 exports `handlers` object, not the config object directly

```typescript
// ❌ WRONG - importing config directly
import { auth } from "@/lib/auth"
export const { GET, POST } = auth  // WRONG

// ❌ WRONG - exporting config
import NextAuth from "next-auth"
export const config = { ... }  // WRONG

// ✅ CORRECT - export handlers from route
import { handlers } from "@/lib/auth"
export const { GET, POST } = handlers
```

**Verification:**
```bash
# Check route handler exists
curl -s https://your-app.vercel.app/api/auth/providers
# Should return: { "github": { "id": "github", "name": "GitHub", ... } }

# Check server logs for errors
# Next.js 16 error: "Type ... has no properties in common with type 'RouteHandlerConfig'"
```

### ❌ GitHub OAuth Callback URL Mismatch

**Symptom:** OAuth flow returns error `redirect_uri_mismatch` or "The redirect_uri is not associated with this application" (500 error on `/api/auth/github`)

**Cause:** GitHub OAuth app callback URL doesn't match production URL

**Fix:** Ensure GitHub OAuth app callback URL EXACTLY matches:
```
https://your-app.vercel.app/api/auth/callback/github
```

**Common mistakes:**
- ❌ Using `http://localhost:3000/api/auth/callback/github` for production
- ❌ Trailing slash: `/api/auth/callback/github/` (must be NO slash)
- ❌ Missing `/api/auth/` prefix

**Verification:**
```bash
curl https://your-app.vercel.app/api/auth/providers
# Should return: { "github": { "id": "github", "name": "GitHub", ... } }
```

### ❌ AUTH_SECRET Too Short

**Symptom:** NextAuth errors about secret length

**Cause:** `AUTH_SECRET` must be at least 32 characters (256 bits)

**Fix:** Generate properly:
```bash
openssl rand -base64 32
```

**Never:** Use hardcoded secrets, short strings, or placeholder values.

### ❌ Route Group Conflict (Parallel Routes)

**Symptom:** 404 on newly created routes (`/login`, `/privacy`, `/terms`)

**Cause:** Next.js parallel routes conflict when both `(legal)/privacy/page.tsx` AND `privacy/page.tsx` exist. Route groups are for layout sharing, not organizing public routes.

**Fix:** Delete route group directory entirely. Use direct routes:
- `/src/app/login/page.tsx`
- `/src/app/privacy/page.tsx`
- `/src/app/terms/page.tsx`

**Lesson:** If you don't need a different layout for a group of routes, use direct routes. Route groups are ONLY for sharing layouts.

### ❌ Vercel Deploying Stale Branch

**Symptom:** Local works, production shows 404s on new routes

**Cause:** Vercel configured to watch `main` branch, but features on `feat/vercel-deploy`

**Fix:**
```bash
# 1. Create PR
gh pr create --title "feat: GitHub OAuth + LGPD" --body "..."

# 2. Manual merge in GitHub UI (branch protection blocks CLI)
# 3. Vercel auto-deploys after main branch updates
```

**Protected Branch Error:**
```
remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: - Changes must be made through a pull request.
remote: - Cannot force-push to this branch.
```

**Pattern:** With protected branches, feature work → PR → manual merge → Vercel auto-deploys.

## Vercel Deployment Workflow

### Environment Variables

**Required in Vercel Project Settings:**

1. Go to Project → Settings → Environment Variables
2. Add each variable with `Production` environment:
   - `AUTH_SECRET`
   - `GITHUB_CLIENT_ID`
   - `GITHUB_CLIENT_SECRET`
   - `NEXTAUTH_URL` (auto-set by Vercel, verify matches)

**Do NOT:**
- ❌ Commit `.env.local` to git
- ❌ Use development `NEXTAUTH_URL` in production

### Production Branch Configuration

**Vercel Dashboard → Git:**
- **Production Branch:** `main` (or your protected branch)
- **Preview Branches:** All branches (for PR previews)

**Pattern:**
- Feature branch → PR preview → test → merge to main → production deploy

## Verification Steps

### Local Development

```bash
# 1. Start dev server
cd web && npm run dev

# 2. Test routes
curl http://localhost:3000/login     # Should load login page
curl http://localhost:3000/privacy   # Should load privacy policy
curl http://localhost:3000/terms     # Should load terms of use

# 3. Test OAuth flow
# - Visit http://localhost:3000/login
# - Click "Entrar com GitHub"
# - Should redirect to GitHub OAuth authorize page
# - After auth, redirect back to /

# 4. Check providers endpoint
curl http://localhost:3000/api/auth/providers
# Should return: { "github": { ... } }
```

### Production (Vercel)

```bash
# 1. Check deployed routes
curl https://your-app.vercel.app/login
curl https://your-app.vercel.app/privacy
curl https://your-app.vercel.app/terms

# 2. Check OAuth providers
curl https://your-app.vercel.app/api/auth/providers
# Should return: { "github": { "id": "github", "name": "GitHub", ... } }

# 3. Test OAuth flow in browser
# - Visit https://your-app.vercel.app/login
# - Click GitHub login
# - Should work if Vercel env vars are set correctly
```

## LGPD Compliance Integration

GitHub-only OAuth simplifies LGPD compliance compared to multi-provider auth:

### Data Processor Notification

Update Privacy Policy `/privacy`:

```markdown
### Dados Recebidos de Terceiros

Utilizamos o GitHub como provedor de autenticação. Durante o fluxo de login, o GitHub processa os seguintes dados:
- E-mail
- Nome completo
- Foto de perfil (opcional)
- ID da conta GitHub

O GitHub atua como processador de dados, conforme o Contrato de Processamento de Dados (DPA) do GitHub. Para mais informações, consulte a [Política de Privacidade do GitHub](https://docs.github.com/pt/github/site-policy/githubs-privacy-statement).
```

### Consent Link in Login Page

Login page already links to both `/terms` and `/privacy` (see implementation above). This satisfies LGPD Art.8 (informed consent).

### Cookie Banner

See `lgpd-compliance` skill for complete cookie banner implementation. GitHub OAuth doesn't require additional cookies — NextAuth uses secure JWT cookies.

**Verification:** Browser DevTools → Application → Cookies
- ✅ `next-auth.session-token` (HttpOnly, Secure, SameSite=Lax)
- ✅ `next-auth.csrf-token` (HttpOnly, Secure)
- ❌ No third-party tracking cookies (unless added separately)

## Type Safety

### Extend Session Type

**File:** `types/next-auth.d.ts`

```typescript
import NextAuth from "next-auth"

declare module "next-auth" {
  interface Session {
    user: {
      id: string
      email: string
      name: string
      image?: string | null
    }
  }

  interface User {
    id: string
    email: string
    name: string
    image?: string | null
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    sub: string
    email: string
    name: string
    picture?: string | null
  }
}
```

**Usage in components:**
```typescript
"use client"

import { useSession } from "next-auth/react"

export function UserProfile() {
  const { data: session } = useSession()

  if (!session) return null

  return (
    <div>
      <img src={session.user.image || ""} alt={session.user.name} />
      <p>{session.user.name}</p>
      <p>{session.user.email}</p>
    </div>
  )
}
```

## Testing

### Unit Tests (Vitest)

**Test: Login page renders**

```typescript
import { render, screen } from "@testing-library/react"
import LoginPage from "@/app/login/page"

describe("LoginPage", () => {
  it("should render login page with GitHub button", () => {
    render(<LoginPage />)
    
    expect(screen.getByText("Bem-vindo")).toBeInTheDocument()
    expect(screen.getByText(/entrar com github/i)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /termos de uso/i })).toHaveAttribute("href", "/terms")
    expect(screen.getByRole("link", { name: /política de privacidade/i })).toHaveAttribute("href", "/privacy")
  })
})
```

**Test: OAuth providers endpoint**

```typescript
import { GET } from "@/app/api/auth/[...nextauth]/route"

describe("NextAuth API", () => {
  it("should return GitHub provider", async () => {
    const response = await GET(new Request("http://localhost:3000/api/auth/providers"))
    const data = await response.json()
    
    expect(data).toHaveProperty("github")
    expect(data.github).toHaveProperty("id", "github")
    expect(data.github).toHaveProperty("name", "GitHub")
  })
})
```

### E2E Tests (Playwright)

**Test: OAuth flow**

```typescript
import { test, expect } from "@playwright/test"

test("GitHub OAuth flow", async ({ page }) => {
  await page.goto("/login")
  
  // Click GitHub login button
  await page.click('button:has-text("Entrar com GitHub")')
  
  // Should redirect to GitHub (mock or skip in CI)
  // In real tests, mock the OAuth callback or use test accounts
})
```

## Related Skills

- `lgpd-compliance` — Cookie banner, privacy/terms pages, data rights
- `nextjs-16-deployment` — Vercel deployment patterns
- `nextjs-authentication` — NextAuth v5 fundamentals (Credentials provider)

## Key Takeaways

1. **GitHub-only OAuth** simplifies auth flow and reduces OAuth provider maintenance
2. **Route groups** for layout sharing only, not organizing public routes (causes 404s)
3. **Vercel deployment** with protected branches requires manual PR merge → auto-deploy
4. **Environment variables** must match exactly between local and production (callback URLs, secret length)
5. **LGPD compliance** is easier with single provider (one data processor to document)
6. **Type safety** requires extending `next-auth` module types for session/user

## References

- NextAuth v5 docs: https://authjs.dev/
- GitHub OAuth Apps: https://github.com/settings/developers
- Next.js 16 docs: https://nextjs.org/docs
- LGPD (Lei 13.709/2018): https://www.planalto.gov.br
- Vercel Environment Variables: https://vercel.com/docs/projects/environment-variables