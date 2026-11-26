# MiniMax Model Lineup (as of May 2026)

## Text Models (LLMs)

| Model | Released | Context | Input Cost | Output Cost | Notes |
|-------|----------|---------|-----------|------------|-------|
| Text-01 / VL-01 | Jan 2025 | - | - | - | Next-gen text + vision |
| M2 | Oct 2025 | 200K | $0.30/M | $1.20/M | Legacy — use M2.5 |
| M2.1 | Dec 2025 | 196K | $0.27/M | $0.95/M | Budget backup |
| **M2.5** | Feb 2026 | 196K | **$0.12/M** | **$1.00/M** | Default — current sweet spot |
| M2.5 Lightning | Feb 2026 | 1M | $0.30/M | $2.40/M | Speed + long context |
| **M2.7** | Mar 2026 | - | - | - | Latest — recursive self-improvement |

## Other Categories (separate APIs)
- **Speech/TTS:** Speech-02-turbo, Speech-2.5, Speech-2.6, Speech-2.8
- **Video:** Hailuo-02 (1080p/10s), Hailuo-2.3, T2V-01-Director, I2V-01-Director
- **Image:** Image-01
- **Music:** Music-1.5, Music-2.0, Music-2.5, Music-2.6

## Cost Comparison (vs Claude Sonnet)

| Model | Input | Output |
|-------|-------|--------|
| Claude Sonnet 4 | $3/M | $15/M |
| MiniMax M2.5 | $0.12/M | $1.00/M |
| **Savings** | **~25x cheaper** | **~15x cheaper** |

## User's Setup

- Hermes backend: MiniMax M2.5 via OpenCode Zen (`opencode.ai/zen/v1`)
- Claude Code backend: Same — OpenCode Zen → MiniMax M2.5 (user configured)
- Both agents share the same model and provider

## Documentation
- Full model list: https://platform.minimax.io/docs/release-notes/models
- M2.5 benchmarking: https://openhands.dev/blog/minimax-m2-5-open-weights-models-catch-up-to-claude