# KENSEI User Memory — Sahil Saghir

**Last updated:** 2026-04-29

---

## Identity
- Name: Sahil Saghir
- Location: Nottingham, UK
- Status: Recently exited Kinexio (Senior PM). Job hunting — Senior PM/TPM, contracting open outside IR35.
- Prior: SoftwareOne, ENSEK, E.ON Energy
- Practising Muslim: Ramadan fasting, Quran memorisation, zakat

## Family
- Wife DOB: 27 Jul 1993
- Daughter DOB: 17 Oct 2021 (age 4)
- Son DOB: 2 Jun 2023 (age 2)
- **Never reference children's identifying info in external output.**

## Health
- Currently 115kg, target 85kg
- On low-dose Ozempic (0.25mg)
- Exercise: badminton, swimming, football 2-3x/week
- Self-described "a bit lazy" — be encouraging without being preachy

## Property & Finance
- 2 properties: current residence + property with mother (listing for sale)
- Plan: sell mother's property, convert current to buy-to-let, buy 5/6-bed family home

## Football
- Manchester United supporter (tactical/historical interest, not just match-day)
- Runs @MaestroMatchday content account (currently dormant, being reset from first principles)
- **Do not reference @MaestroMatchday strategy as "active" — it has been reset.**

## Active App Portfolio
1. **Plenishd** — RN+Expo+Convex. Voice/photo kitchen inventory + UK supermarket integration. Pre-launch.
2. **CoachOS** — RN+Expo+Supabase. Grassroots football coaching OS. Beta Aug 2026.
3. **MatchdayMaestro** — RN. Football prediction/trivia/analytics. Approaching Google Play production.
4. **Kick-tionary** — Flutter+Supabase+Riverpod. Football tactics education ages 6-18. Submitted to stores.

Tech stack: Convex over Supabase for new projects. RN+Expo over Flutter for new mobile.
Cost philosophy: free/near-free first, scale to paid only when proven.

## Dev Environment
- Primary: Claude Code in WSL Ubuntu on MSI Modern 15 C13M (i7-13700H, 16GB RAM, no GPU)
- WSL Tailscale IP: 100.69.205.89

## KENSEI Infrastructure
- Hetzner CPX32 VPS, Tailscale IP 100.118.24.103
- Primary model: Kimi K2.6 via Ollama Cloud Pro
- Fallback: Qwen3 Coder via OpenRouter free
- Approval mode: manual — every dangerous command requires explicit approval
- Timezone: Europe/London (BST/GMT)

## Connected Google Accounts (Block 2)
- **Default Gmail:** saghir.sahil@gmail.com
- **Secondary Gmail:** sahilsaghir.ss9@gmail.com
- **Studio Gmail:** fusionfirststudios@gmail.com
- All 3 authenticated via Workspace MCP, all tools exposed (complete tier)
- KENSEI-managed Gmail labels should use `kensei/` prefix
- Token refresh: manual on demand (expiry every 7 days in testing mode)

## Communication
- Direct, casual, sharp. No corporate AI-speak, no em-dashes, no sycophancy, no excessive hedging.
- Profanity OK when it lands.
- **Honest pushback wanted, not agreement.**
- **Do not recommend paid tools without flagging cost first.**
- Hard limits: child safety, dangerous capabilities, sensitive personal context unless raised first.

## Job Hunt Proof Points
- RAG chatbot at SoftwareOne: 20-30% conversion lift, 55% support query reduction
- FM portal at Kinexio: 25-35% cycle time reduction
- Indie app portfolio as differentiator
§
User expects proactive skill updates after every productive session. 'Nothing to save' is only appropriate when no corrections, no new techniques, and no workflow learnings occurred. User explicitly said: "most sessions produce at least one skill update, even if small". Wants a personalised Command Center that feels like his own, not the generic Hermes Workspace. Open to building from scratch or heavily customising — picking best bits from multiple command centers.
§
Prefers "research and clarify" mode before any action is taken on infrastructure, tooling, or skill installations. Will explicitly say "just research, don't action anything yet" when scoping.
§
User refuses to pay for Obsidian Sync subscription. VPS-hosted vault synced to laptop via private GitHub repo and Git. Vault path: /home/kensei/vaults/obsidian-master. Sync script: .gitsync.sh (push/pull).