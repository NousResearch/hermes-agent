# Competitive Intelligence Operations

## Tracker Format
The competitive tracker lives at `~/built-lms-agent-system/trackers/competitive-tracker.md`.

## Active Research Requirements
- Web search for competitor product launches, funding, M&A, pricing changes
- RSS feed monitoring for product and company blogs
- Direct career page checks (not LinkedIn/Indeed/G2 — bot protected)

## Delivery
Explicit Slack target channel. Never use `origin`.

## Known Failure Modes
- Cloudflare blocked pages require Cloudflare Browser Rendering (credentials in ~/.env)
- Job discovery cannot use LinkedIn, Indeed, G2, or Capterra — use direct career pages and /academy paths
- Google search fallback: CloakBrowser (when Brave API fails)