# Hermes Agent v0.10.1 (v2026.4.22)

**Release Date:** April 22, 2026

> Focused follow-up release for Mixture of Agents. MoA now defaults to a direct multi-provider stack instead of OpenRouter-only routing.

---

## ✨ Highlights

- **Direct-Provider MoA Default Stack** — `mixture_of_agents` now uses **Xiaomi MiMo v2 Pro** as the aggregator with **MiniMax-M2.7-highspeed** and **DeepSeek Reasoner** as the default reference models. This keeps MoA on first-class direct providers while still preserving support for legacy OpenRouter-style model slugs passed explicitly.

- **Cleaner Provider Routing** — MoA now flows through Hermes' shared provider-aware `async_call_llm()` / `resolve_provider_client()` pipeline instead of maintaining a separate OpenRouter-only client path. That means consistent credential resolution, base URL handling, and fallback behavior with the rest of Hermes auxiliary routing.

---

## ⚠️ Upgrade Note

- **MoA is no longer OpenRouter-only by default.** If you relied on the old default stack, configure `XIAOMI_API_KEY` plus at least one reference-provider key: `MINIMAX_API_KEY` or `DEEPSEEK_API_KEY`.

- **Backward compatibility is preserved for explicit model overrides.** Existing OpenRouter-style model slugs such as `anthropic/claude-opus-4.6` still work when passed directly to MoA.

---

## 🐛 Reliability Improvements

- Empty reasoning-only responses from reference models now fail closed on the final retry instead of being counted as successful empty outputs.

- Setup, tool configuration, and docs now describe the real MoA credential contract: Xiaomi as aggregator plus at least one reference provider.
