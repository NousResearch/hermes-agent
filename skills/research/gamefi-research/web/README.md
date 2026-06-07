# GameFi Research Workflow — Demo UI

A lightweight **showcase** web UI for the [`gamefi-research`](../) Hermes Agent
skill. It presents the scanner's output — top projects, Game Research Signal
Score, WATCH / TEST / CONTACT / SKIP categories, score breakdowns, source
links, and a sample Markdown report.

This is a **demo only**. It uses **static sample data** (`lib/sampleData.ts`),
makes no network calls, and contains **no tokens or secrets**.

## Tech

- Next.js 14 (App Router) + React 18 + TypeScript
- No UI framework, no external data — just one CSS file and sample data

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
3. Framework preset: **Next.js** (auto-detected). Build/output settings are
   the defaults.
4. No environment variables are required (the demo has no secrets).
5. Deploy.

CLI alternative:

```bash
cd skills/research/gamefi-research/web
npx vercel        # preview
npx vercel --prod # production
```

## Updating the sample data

Edit `lib/sampleData.ts`. The shape mirrors the real output of
`scripts/gamefi_scan.py` (projects, scores, categories, breakdowns, sources,
and the raw Markdown report string), so a real report can be dropped in later.

## Disclaimer

Neutral research showcase. Not financial advice, not a trading tool, not an
investment recommendation. Public signals only; all data shown is illustrative
and unverified.
