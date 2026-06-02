<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

> **Fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).** This fork includes several patches not yet merged upstream — see [below](#fork-specific-patches). I dogfood this instance and keep it updated regularly; upstream moves slower due to bureaucratic overhead.

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
</p>

See the **[original README →](https://github.com/NousResearch/hermes-agent#readme)** for full documentation, install instructions, CLI reference, and everything else. This fork only changes what's listed below.

---

## Fork-Specific Patches

This fork maintains the following patches on top of upstream:

| Patch | Description | Status |
|-------|-------------|--------|
| **YOLO mode for computer_use** | Wires the `--yolo` / `approvals.mode: off` setting into the Computer Use tool's approval layer so automated workflows don't get stuck on approval prompts. | PR [#36926](https://github.com/NousResearch/hermes-agent/pull/36926) |
| **Model override for delegate_task** | Adds a `model` parameter (with optional `provider` + `model` fields) to `delegate_task`, letting subagents run under a different model than the parent — e.g., Claude for vision tasks while the parent stays on DeepSeek. Per-invocation and per-task overrides supported. | Various PRs: [#34681](https://github.com/NousResearch/hermes-agent/pull/34681), [#34773](https://github.com/NousResearch/hermes-agent/pull/34773), [#36790](https://github.com/NousResearch/hermes-agent/pull/36790) |

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com). Fork maintained by [@ThatXliner](https://github.com/ThatXliner).
