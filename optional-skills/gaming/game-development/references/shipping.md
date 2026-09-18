# Building & Publishing

Turning a project into a real, distributable game. Covers building per platform
and publishing to the major stores.

---

## Building per Platform

### Desktop (Windows / macOS / Linux)

| Engine | How |
|---|---|
| **Unity** | File → Build Settings → select platform → Build. Switch platform targets in advance (re-imports assets). |
| **Unreal** | Platforms → Package Project → choose target. Set to **Shipping** config for release (strips debug). |
| **Godot** | Project → Export → add a preset per platform (install export templates first). |

- **Windows:** ship as a folder or installer (Inno Setup / NSIS). Code-sign the
  `.exe` to avoid SmartScreen warnings.
- **macOS:** must be **signed and notarized** by Apple or users can't open it.
  Requires an Apple Developer account ($99/yr).
- **Linux:** ship a tarball or, better, a **Flatpak**/AppImage for distro
  independence.

### Web (HTML5 / WASM)

- **Unity:** WebGL build target. Large initial download — optimize aggressively
  (compression, stripped code). Not great for big games.
- **Godot:** Web export — lightweight, great for jams; note threads/audio
  caveats on some browsers.
- **Custom JS/TS:** already web-native; bundle with Vite/esbuild, host anywhere.
- **Host on:** itch.io (built-in HTML5 player), GitHub Pages, Netlify.

### Mobile (iOS / Android)

- **Android:** export an **AAB** (Android App Bundle) for the Play Store, APK
  for direct/testing. Sign with a keystore (keep it safe — losing it means you
  can't update the app).
- **iOS:** build to Xcode, then archive and upload via Xcode/Transporter.
  Requires Apple Developer account and provisioning profiles.
- **Both:** test on real devices, not just emulators. Handle touch input,
  varied aspect ratios, and performance on low-end hardware.

---

## Publishing to Stores

### Steam (the main PC marketplace)

1. **Steamworks account** — $100 per app (Steam Direct fee, recoupable).
2. **Set up the app** in the Steamworks partner site: store page, depots
   (your build files), branches (default/beta).
3. **Integrate the SDK** if you want achievements, cloud saves, overlay
   (optional but expected).
4. **Upload builds** via SteamPipe (`steamcmd` + a build script / app+depot
   config).
5. **Store page** — capsule images (multiple sizes), trailer, screenshots,
   description, tags. The capsule art is your #1 conversion driver.
6. **Wishlists** — open the page (as "Coming Soon") months early; wishlists at
   launch drive Steam's algorithm. This is the single biggest marketing lever.
7. **Review** — Steam reviews builds + store page before release.

### itch.io (indie-friendly, low-friction)

1. **Create a page**, set pricing (or pay-what-you-want / free).
2. **Upload** with **butler** (itch's CLI) for versioned, patchable uploads:
   ```bash
   butler push ./build user/game:windows
   butler push ./build user/game:html --userversion 1.0.1
   ```
3. No gatekeeping, instant publish, HTML5 games play in-browser. Ideal for
   jams, demos, early builds, and testers.

### Mobile Stores

- **Google Play:** $25 one-time. Upload AAB, fill the store listing, set
  content rating, roll out (staged rollout recommended).
- **Apple App Store:** $99/yr. Upload via Xcode, complete App Store Connect
  listing, submit for review (stricter and slower than Google).

---

## Launch Checklist

Before you hit publish:

- [ ] Builds tested on clean machines (not your dev box) for each platform
- [ ] Save/load works across versions; settings persist
- [ ] All input methods work (keyboard, controller, touch as relevant)
- [ ] Audio has master/music/SFX volume controls that persist
- [ ] Resolution/fullscreen/windowed handled; UI scales to aspect ratios
- [ ] No debug overlays, cheat keys, or console spam in the release build
- [ ] Crash on common edge cases checked (alt-tab, resize, controller unplug)
- [ ] Store page: capsule art, trailer, screenshots, accurate description
- [ ] Age rating / content descriptors set (IARC for most stores)
- [ ] Version number + changelog ready (see `changelog-generator` skill)
- [ ] Backups of signing keys / keystores stored safely
- [ ] A way for players to report bugs / reach you

---

## Marketing Reality (the part devs skip)

A great game nobody hears about doesn't sell. Minimum effort:

- **Build an audience before launch** — devlog, social posts (see
  `social-media/content-creator`), a Steam "Coming Soon" page for wishlists.
- **Trailer** — the first 6 seconds decide if people keep watching. Lead with
  gameplay, not logos.
- **Press/creators** — a short pitch + a press kit (key art, GIFs, fact sheet,
  download key) sent to relevant YouTubers/streamers/journalists.
- **Launch is a moment, not a strategy** — wishlists, community, and steady
  visibility built over months matter more than launch day alone.

---

## Post-Launch

- **Day-one patch readiness** — have the pipeline to push a hotfix fast;
  something always slips through.
- **Analytics** — track funnel drop-off, session length, crash reports to guide
  updates with data, not guesses.
- **Community** — a Discord/forum, respond to feedback, communicate a roadmap.
  Engaged players become your marketing.
