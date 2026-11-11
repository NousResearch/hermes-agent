# Architecture Decisions

Key design decisions made for the Unitalk Desktop fork, with reasoning. Each decision has a stated **why** — don't reverse one without understanding it.

---

## 1. "AI Collaborator" not "Agent"

**Decision**: The AI entity in the system prompt is called "AI Collaborator" rather than "Agent".

**Reasoning**: The identity line in `DEFAULT_AGENT_IDENTITY` is a placeholder. When users personalize their AI (during onboarding, the "name" beat), this string is replaced with the user-chosen name. "AI Collaborator" is a descriptive, professional fallback that works before personalization and conveys partnership without the technical baggage of "agent". "Agent" implies tool-execution primacy; "Collaborator" implies conversation and assistance — which matches Unitalk's target audience of pro users who want an AI partner, not a CLI tool runner.

**Constraint**: Must remain syntactically compatible with the upstream's identity-line substitution (`S_O_U_L.md` file or custom persona replacing this line). The upstream expects the pattern `"You are <name>, built by <company>."` — we keep this structure intact.

---

## 2. 11 Brand Themes Instead of 13 Geeky Ones

**Decision**: Replace the upstream's 13 developer-culture theme names (Dracula, Nord, Tokyo Night, Catppuccin, etc.) with 11 professional brand-inspired themes.

**Our themes**: Microsoft, Apple, Google, ChatGPT, Claude, Grok, GitHub, Midnight, Slate, Forest, Unitalk.

**Reasoning**: Unitalk's target audience is **professional users, not developers**. A marketing director choosing "Tokyo Night" as their app theme is confusing. Brand-inspired themes (Microsoft blue, Apple silver, Google Material) are instantly recognizable and feel native to the tools they already use. The Unitalk theme (warm paper & magenta) is the default — it's the first thing they see and sets Unitalk's visual identity.

**Downstream effect**: `RETIRED_SKINS` set in `presets.ts` catches any upstream theme names that bleed through. Users who had selected a geeky theme from an older install fall back to `DEFAULT_SKIN_NAME` ('unitalk').

---

## 3. Browser-Auth Providers Only

**Decision**: Hide terminal-auth providers (like Qwen Code) from the onboarding provider picker. Show only providers with browser-based OAuth flows.

**Reasoning**: Unitalk targets non-technical users. Terminal-based authentication (copy-pasting codes into a terminal) is foreign and intimidating to someone who's never used a CLI. Browser-based OAuth is the standard flow they already know from signing into Slack, Notion, or Google. Removing terminal-auth options simplifies the first-run experience and reduces abandonment.

**Implementation**: `HIDDEN_PROVIDER_IDS` in `providers.tsx` contains `'qwen-oauth'`. Terminal-auth providers are still available through the full Settings → Providers panel — they're just not shown during onboarding.

**Also removed**: API-key providers (Fireworks, OpenRouter) from the onboarding screen. These require users to find, copy, and paste API keys from external websites — another non-technical-user friction point. Local models are still offered through a dedicated row.

---

## 4. Unitalk First in Provider List

**Decision**: `PROVIDER_ORDER` puts `nous: 0` (the Unitalk provider, mapping to `nous/welcome`) at position 1 in the onboarding provider picker.

**Reasoning**: Brand visibility and user experience. The free tier (`nous/welcome`) is the easiest entry point — no account needed, no API key, no payment. It should be the first thing a new user sees. The order after Unitalk is: OpenAI Codex, MiniMax OAuth, xAI OAuth, Anthropic, Claude Code — roughly ordered by popularity and ease of setup.

**Provider order**:
```
nous (Unitalk)         → position 0 (first, featured card with arc border)
openai-codex           → position 1
minimax-oauth          → position 2
xai-oauth              → position 3
anthropic              → position 4
claude-code            → position 5
```

---

## 5. `DEFAULT_SKIN_NAME = 'unitalk'`

**Decision**: The default theme (when nothing is persisted, or the stored theme is retired) is `'unitalk'`.

**Reasoning**: The Unitalk brand theme is the first visual impression. It uses warm paper tones and magenta accents — distinct from every other AI desktop app. When a user opens Unitalk for the first time, they should immediately see Unitalk's brand, not a generic dark theme. This also means the onboarding chat (which starts before any theme preference is set) carries the Unitalk brand colors.

**Fallback chain**: unitalk → (user picks another) → (if that theme is retired) → unitalk. The `RETIRED_SKINS` set is currently empty but ready to receive upstream theme names.

---

## 6. Welcome Plugin Replaces the Empty State

**Decision**: A custom welcome plugin (`apps/desktop/src/plugins/welcome/`) replaces the upstream's empty new-session screen with a branded dashboard.

**What it shows**:
- Greeting: "Bonjour, je suis votre Collaborateur IA" with a prompt input and suggestion chips
- Recent conversations (last 5 sessions from `/api/sessions`)
- Info card: connected profile, current model, credits bar, trial countdown, upgrade/buy-credits buttons

**Reasoning**: The upstream's empty state (a blank chat area) is hostile to non-technical users — it gives no guidance on what to do. The welcome dashboard gives immediate orientation: who they're talking to, what they've done before, what their account status is. It reduces the "blank page" anxiety and shows Unitalk's value before the first message is sent.

**Implementation**: Pure SDK consumer — registered as a plugin with `defaultEnabled: true`, routes at `/` and adds a sidebar nav item. No core edits needed. The plugin is French-language (Unitalk's primary market) but supports the `useI18n()` hook for future localization.

---

## 7. Product Name is "Unitalk" Everywhere the User Sees It

**Decision**: All user-facing strings say "Unitalk" — never "Hermes" or "Nous Research".

**Reasoning**: Brand consistency. A user who installs "Unitalk Desktop" should never see "Hermes" in a greeting, a consent dialog, a README, or an error message. The only exceptions are protocol-level identifiers (HTTP headers, env vars, Python module names) and the upstream docs URL (`hermes-agent.nousresearch.com/docs`) which is NOT user-facing — it's read by the AI model, not the human.

**Affected surfaces**: i18n strings (all 8+ locales), shared metrics consent, first-run intro copy, onboarding text, welcome plugin, provider display names, README files, install scripts.

---

## Summary Matrix

| Decision | Target | Trade-off |
|---|---|---|
| Collaborator not Agent | User comprehension | Model must understand "Collaborator" = itself |
| 11 brand themes | Pro familiarity | Fewer themes than upstream (13→11) |
| Browser-auth only | Non-technical UX | Fewer provider options during onboarding |
| Unitalk first | Brand visibility | Pushes OpenAI to position 2 |
| `unitalk` default skin | First impression | Users must actively switch away |
| Welcome plugin | Orientation | Extra JS bundle in every install |
| "Unitalk" everywhere | Brand consistency | Upstream merges are more complex |