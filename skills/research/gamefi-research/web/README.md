# Game Research Workflow for Hermes — Demo Page

A clean, lightweight **showcase** page for the
[`gamefi-research`](../) Hermes Agent skill. It presents the workflow visually:
the problem, the Scan → Review → Score → Classify → Generate flow, current
features, a sample output card, a generated report preview, and the roadmap.

This is a **demo only**. It uses **static sample data** (`lib/sampleData.ts`),
makes **no API calls**, has **no backend**, and contains **no secrets**.

## Tech

- Next.js 14 (App Router) + React 18 + TypeScript
- Tailwind CSS
- Static sample data

## Run locally

```bash
cd skills/research/gamefi-research/web
npm install
npm run dev
# open http://localhost:3000
```

Production build check:

```bash
npm run build && npm run start
```

## Deploy to Vercel

This app lives in a subdirectory of the repo, so point Vercel at it:

1. Import the repository in Vercel.
2. Set **Root Directory** to `skills/research/gamefi-research/web`.
3. Framework preset: **Next.js** (auto-detected); default build settings.
4. No environment variables are required.
5. Deploy.

CLI alternative:

```bash
cd skills/research/gamefi-research/web
npx vercel        # preview
npx vercel --prod # production
```

## Disclaimer

Neutral research showcase. Public repository signals only. All data shown is
illustrative and unverified. Not advice of any kind.
