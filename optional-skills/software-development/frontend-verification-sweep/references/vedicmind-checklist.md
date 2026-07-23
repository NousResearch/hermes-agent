# <your-project> Verification Checklist

Project: C:\Users\<username>\<your-project>\vedic-math-platform
Tech: React 19, Vite 7, TypeScript 5.9, Supabase, Tailwind 4, shadcn/ui, TanStack Query 5, GSAP 3.15, Lenis 1.3, KaTeX 0.17, Nivo 0.99, Howler 2.2, canvas-confetti 1.9, Sonner 2.0, cmdk 1.1

## Expected Packages (25 deps + 15 devDeps)
- @base-ui/react, @fontsource-variable/geist, @nivo/core, @nivo/radar
- @supabase/supabase-js, @tanstack/react-query, @tanstack/react-query-devtools
- canvas-confetti, class-variance-authority, clsx, cmdk, framer-motion
- gsap, howler, katex, lenis, lucide-react, react, react-dom, react-katex
- recharts, shadcn, sonner, tailwind-merge, tw-animate-css, zustand
- Dev: @tailwindcss/vite, @testing-library/jest-dom, @testing-library/react
- Dev: @types/canvas-confetti, @types/howler, @types/katex, @types/node
- Dev: @types/react, @types/react-dom, @vitejs/plugin-react, jsdom
- Dev: tailwindcss, typescript, vite, vitest

NOTE: vite-plugin-pwa was REMOVED (commit a8de1cd). Do NOT expect it.

## Feature Phases

### shadcn/ui
- Components dir: src/components/ui/
- Required: button, card, dialog, tabs, badge, avatar, skeleton
- Config: components.json, src/lib/utils.ts

### TanStack Query
- Config: src/lib/query-client.ts
- Hooks: src/hooks/queries/{useProfile,usePracticeSessions,useLeaderboard,usePvPSession}.ts
- Docs: MIGRATION.md
- Provider: QueryClientProvider in main.tsx

### Dashboard
- Component: src/components/ProgressDashboard.tsx (uses @nivo/radar + shadcn Tabs)

### KaTeX
- Math component: src/components/ui/math.tsx
- Learn content: src/components/learn/VerticallyAndCrosswise.tsx
- Docs: src/components/learn/README.md
- CSS: katex/dist/katex.min.css imported in VerticallyAndCrosswise.tsx (lazy, not global)

### Sonner + cmdk
- Toaster in App.tsx (sonner)
- CommandPalette.tsx (cmdk) with Sound mute toggle group

### Video Player
- src/components/VideoPlayer.tsx (standalone, 626 lines)
- src/components/MediaPlayer.tsx (legacy, still present)
- Legacy fallback: LearnSection.tsx line 277 has `useLegacyPlayer` — activates via `?legacyPlayer=1` URL param
- When active, renders `<MediaPlayer>` instead of `<VideoPlayer>`

### GSAP + Lenis
- Landing page: src/components/landing/LandingPage.tsx
- Lenis only initialized in LandingPage (isolated)
- LandingPage lazy-loaded in App.tsx, rendered when no user logged in

### Confetti + Audio
- src/lib/sfx.ts: playCorrect, playWrong, playWin, playLose, playTick, playReady, setVolume, mute, unmute
- src/lib/celebrate.ts: victory, streak, firstWin
- Mute toggle in CommandPalette Sound group

### PWA — REMOVED
- vite-plugin-pwa uninstalled, VitePWA config removed from vite.config.ts
- InstallPrompt.tsx and offline.html deleted
- App.tsx: offline event handler removed, toast import removed
- public/manifest.json still exists (pre-existing, not PWA — just browser metadata)
- Build no longer produces sw.js, workbox-*.js, registerSW.js, or manifest.webmanifest

## Friend's Files (<teammate>)
- App.tsx: sonner, appStore, AuthProvider, lazy loading for all routes
- src/components/DailyChallengeSection.tsx (821 lines)
- src/components/auth/Signup.tsx (325 lines)
- src/components/auth/ResetPassword.tsx (305 lines)
- src/context/AuthContext.tsx (348 lines)
- src/store/appStore.ts (413 lines)

## Known Large Components (as of merge verification)
- HomePage.tsx: 1123 lines (CRITICAL)
- LearnSection.tsx: 1100 lines (CRITICAL)
- PvPBattleSection.tsx: 983 lines
- ProgressDashboard.tsx: 916 lines
- BotBattleSection.tsx: 879 lines

## Known Issues
- recharts + @nivo/radar both installed (redundant)
- 28 hardcoded Supabase storage URLs in mathGenerators.ts (should be config)
- KaTeX CSS is lazy-loaded in VerticallyAndCrosswise.tsx (29KB separate chunk, not global)
