# Hermes Desktop feature parity

Audit of the Rust port, updated 23 November 2026. The reference is the public NousResearch source release **v0.23.3**, commit `caafe5389ae585ef77387a962aee05b1bcf17029`. The migration branch started at `0284ce17d9bf9ed2f1832ad8950dfac2394d45ab`. The published source tag and its desktop guide define the comparison. Installer release timing can differ by platform. [Release notes](https://github.com/NousResearch/hermes-agent/releases/tag/v0.23.3) · [Desktop guide](https://github.com/NousResearch/hermes-agent/blob/caafe5389ae585ef77387a962aee05b1bcf17029/website/docs/user-guide/desktop.md)

## What runs

The bundle holds two Cargo-built native programs, `hermes-desktop` (Iced 0.14, a multi-window daemon) and `hermes-gateway`. The Rust engine runs turns, streams replies, asks for approval and saves results. There is no Electron, Tauri or Node runtime. HTML artifacts preview in the OS webview (wry: WKWebView, WebView2, WebKitGTK); the global ⌘⇧Space hotkey uses the `global-hotkey` crate. When the Hermes Python agent is installed, two optional features use it, exactly as the Electron app does: Python plugins (`engine/assets/python_plugin_bridge.py`) and the "hey hermes" wake-word model (`desktop/assets/wake_listener.py`). Without it both degrade (Python plugins are refused with a reason; the wake word falls back to spotting the phrase in live dictation).

Evidence levels below: **verified live** means driven against real services or the real window on Apple Silicon macOS; **tested** means automated behaviour tests (158 pass, `cargo test --workspace`); **built** means compiled and reviewed but not exercised end to end. Windows and Linux are built in CI (`.github/workflows/desktop-native-rust.yml`) and have not been driven by hand.

## Published routes

| Upstream route | Rust status | Working behavior and outstanding scope |
| --- | --- | --- |
| `/` and conversation IDs | Done | Create, select, rename, pin, archive and resume chats; drafts; streamed Markdown; approvals; cancel; @file/@folder/@diff/@git/@url references; attachments; prompt queue; group chats; subagents (`delegate_task`); "Open in new window"; cross-window sync of the chat list (tested). |
| `/settings` | Done | Accounts (Nous Portal OAuth, OpenAI Codex, xAI, MiniMax, Qwen device sign-ins, Anthropic/Claude Code), API keys, custom endpoints, model catalog and picker, reasoning/fast presets, appearance, shortcuts, voice, status bar, gateways (local, remote with native OAuth, SSH). Device codes verified live. |
| `/session-import` | Done | Import from the Python agent's SQLite store and other sources into independent native copies. |
| `/capabilities` | Done | Skills catalog with enable/disable, MCP servers (HTTP, stdio, OAuth 2.1), plugin catalog (2,490 pinned entries): install, enable, settings form, sign-in, update, remove. Agent Plugins v1 packages verified live (Aave); Python plugins tested through the installed agent. |
| `/artifacts` | Done | Gallery, search, filters, previews including live HTML in the OS webview, origin navigation. |
| `/messaging` | Done | Channel setup and live conversations through the gateway adapters the Rust gateway implements. |
| `/webhooks` | Done | Create, enable, delete; HMAC-SHA256 verified deliveries (`/webhooks/{name}`) run the agent into a "Webhook · name" chat (tested). |
| `/cron` | Done | Routines page with 16 templates, scheduler, run history, approvals; sidebar "Cron jobs" section with countdowns and run peek; each routine has a chat that shows its run live while it streams. |
| `/profiles` | Done | Create, rename, delete, SOUL editing, export/import (tar.gz). |
| `/agents` | Done | Live subagent tree (`subagent.progress`), open a child's transcript. |
| `/command-center` | Done | Command Center screen and actions. |
| `/starmap` | Done | Canvas graph of enabled skills, related links, clusters and memory cards. |
| Contributed routes | Partial | Plugins add tools, skills and MCP servers. Desktop UI contributions from plugins (extra pages) are not loaded. |

## Workflows within and outside those routes

| Published behavior | Native scope |
| --- | --- |
| HUD mode (⌘⇧H) | Own frameless always-on-top window with native frost (NSVisualEffectView / Windows 11 acrylic), idle fade, click-through while faded, hover or a reply brings it back; composer, + menu, model picker and reasoning menu work inside it. Verified live on macOS with real AppKit click and key events. Linux keeps a solid band. |
| Quick entry, pet, wake pill | ⌘⇧Space quick-entry window, desktop pet (petdex gallery and AI pet generation via Nous Portal), wake indicator. Verified live. |
| Voice | Dictation, read-aloud and voice conversation on macOS (Speech + AVFoundation). Wake word: the Python agent's "hey hermes" openWakeWord model (tested, installs its packages on first use), else phrase spotting. Windows/Linux dictation is not implemented. |
| Cloud | Hermes Cloud agent discovery through the Portal (verified live: listing works). Connecting to a cloud agent needs one that is reachable. |
| Connections | Local gateway, remote gateway (token or native OAuth), SSH (remote `hermes serve` + port forward). SSH has not been run against a real host. |
| Terminal, files, Git | PTY tabs, project file tree and previews, repository status and diffs. |
| Packaging | Signed macOS app bundle; Windows zip and Linux tarball from CI. No notarization, installers or auto-update. |

## Gateway contract inventory

_Counted on 21 November 2026 and not recounted since._

The published OpenRPC file declares **263 unique method names**. The Rust gateway advertises **25** methods, **18** of which appear in that file. **245** published names are absent from the Rust method table. A matching name establishes dispatch coverage only. It does not certify complete request schemas or behavior, and the count is not a desktop feature percentage. [Published OpenRPC registry](https://github.com/NousResearch/hermes-agent/blob/caafe5389ae585ef77387a962aee05b1bcf17029/apps/shared/src/gateway-contract.openrpc.json) · [Native handlers](crates/gateway/src/lib.rs) · [Gateway wire and tests](crates/gateway/README.md)

| Published namespace | Advertised names in the Rust gateway | Outstanding published names |
| --- | --- | --- |
| agents | — | `agents.list` |
| approval | `approval.pending`, `approval.received`, `approval.respond` | — |
| billing | — | `billing.auto_reload`, `billing.charge`, `billing.charge_status`, `billing.keep_going_on_credits`, `billing.state`, `billing.step_up` |
| bot_delivery | — | `bot_delivery.list` |
| bot_relay | — | `bot_relay.deliver`, `bot_relay.outbox.drain`, `bot_relay.reply`, `bot_relay.roster.sync` |
| browser | — | `browser.manage` |
| clarify | — | `clarify.lock` |
| cli | — | `cli.exec` |
| client | `client.capabilities` | — |
| clipboard | — | `clipboard.paste` |
| command | — | `command.dispatch`, `command.resolve` |
| commands | — | `commands.catalog` |
| complete | — | `complete.path`, `complete.slash` |
| config | — | `config.get`, `config.set`, `config.show` |
| connection | — | `connection.respond` |
| connectors | — | `connectors.accounts`, `connectors.accounts.remove`, `connectors.accounts.rename`, `connectors.catalog`, `connectors.connect`, `connectors.list`, `connectors.operation.status`, `connectors.operation.wake`, `connectors.policy.get`, `connectors.policy.set`, `connectors.tools` |
| cron | — | `cron.manage` |
| delegation | — | `delegation.pause`, `delegation.status` |
| diagnostics | — | `diagnostics.share_nous` |
| display | — | `display.install`, `display.lease.acquire`, `display.lease.release`, `display.observe`, `display.start`, `display.status`, `display.stop`, `display.switchSandboxImage`, `display.thumbnail` |
| file | — | `file.attach` |
| free_tier | — | `free_tier.ack_notice`, `free_tier.challenge_result`, `free_tier.claim_nudge`, `free_tier.provision`, `free_tier.status` |
| gateway | `gateway.capabilities` | — |
| groups | — | `groups.approve`, `groups.attachment.download`, `groups.attachment.upload`, `groups.capabilities`, `groups.context`, `groups.continue`, `groups.create`, `groups.demote`, `groups.disband`, `groups.discard`, `groups.list`, `groups.log`, `groups.peer.invite`, `groups.peer.register`, `groups.peer.revoke`, `groups.promote`, `groups.rename`, `groups.replica_state`, `groups.replicate`, `groups.retry`, `groups.send`, `groups.state`, `groups.stop` |
| handoff | — | `handoff.fail`, `handoff.request`, `handoff.state` |
| i18n | — | `i18n.catalog`, `i18n.languages` |
| image | — | `image.attach`, `image.attach_bytes`, `image.detach`, `image.generate`, `image.render` |
| input | — | `input.detect_drop` |
| insights | — | `insights.get` |
| learning | — | `learning.delete`, `learning.detail`, `learning.edit`, `learning.frames` |
| llm | — | `llm.oneshot` |
| machine | — | `machine.facts` |
| mcp | — | `mcp.catalog`, `mcp.servers.add`, `mcp.servers.list`, `mcp.servers.oauth.callback`, `mcp.servers.oauth.cancel`, `mcp.servers.oauth.poll`, `mcp.servers.oauth.start`, `mcp.servers.remove`, `mcp.servers.set_api_key`, `mcp.servers.status`, `mcp.servers.test` |
| message | — | `message.react` |
| model | — | `model.disconnect`, `model.options`, `model.save_key` |
| onboarding | — | `onboarding.set_run`, `onboarding.skip`, `onboarding.start`, `onboarding.state` |
| paste | — | `paste.collapse` |
| pdf | — | `pdf.attach` |
| pet | — | `pet.cancel`, `pet.cells`, `pet.disable`, `pet.export`, `pet.gallery`, `pet.generate`, `pet.generate.status`, `pet.hatch`, `pet.info`, `pet.info.meta`, `pet.remove`, `pet.rename`, `pet.scale`, `pet.select`, `pet.thumb` |
| ping | `ping` | — |
| plugins | — | `plugins.catalog.lookup`, `plugins.list`, `plugins.manage` |
| preview | — | `preview.restart` |
| process | — | `process.kill`, `process.list`, `process.stop` |
| profiles | — | `profiles.configure`, `profiles.create`, `profiles.describe`, `profiles.get_asset`, `profiles.list`, `profiles.set_asset` |
| project | — | `project.facts` |
| projects | — | `projects.add_folder`, `projects.archive`, `projects.create`, `projects.delete`, `projects.discover_repos`, `projects.for_cwd`, `projects.get`, `projects.list`, `projects.project_sessions`, `projects.record_repos`, `projects.remove_folder`, `projects.set_active`, `projects.set_primary`, `projects.tree`, `projects.update` |
| prompt | `prompt.submit` | `prompt.background`, `prompt.btw` |
| reload | — | `reload.env`, `reload.mcp` |
| request | — | `request.answer` |
| rollback | — | `rollback.diff`, `rollback.list`, `rollback.restore` |
| session | `session.activate`, `session.active_list`, `session.close`, `session.create`, `session.events.since`, `session.history`, `session.interrupt`, `session.list`, `session.resume`, `session.status`, `session.title` | `session.archive`, `session.branch`, `session.branch_stored`, `session.branch_whole`, `session.compress`, `session.context_breakdown`, `session.control`, `session.control.read`, `session.cwd.set`, `session.delete`, `session.events.stats`, `session.foreign.import`, `session.foreign.list`, `session.foreign.preview`, `session.most_recent`, `session.redirect`, `session.save`, `session.set_hidden`, `session.steer`, `session.undo`, `session.usage`, `session.workspace.move` |
| setup | — | `setup.runtime_check`, `setup.status` |
| shared_metrics | — | `shared_metrics.desktop_daily`, `shared_metrics.desktop_dislike`, `shared_metrics.desktop_feature_use`, `shared_metrics.desktop_friction`, `shared_metrics.desktop_onboarding`, `shared_metrics.set`, `shared_metrics.slash_command`, `shared_metrics.startup_latency`, `shared_metrics.status`, `shared_metrics.update_run` |
| shell | — | `shell.exec` |
| skills | — | `skills.manage`, `skills.reload` |
| slash | — | `slash.exec` |
| spawn_tree | — | `spawn_tree.list`, `spawn_tree.load`, `spawn_tree.save` |
| subagent | — | `subagent.interrupt`, `subagent.list`, `subagent.roster`, `subagent.steer`, `subagent.tail` |
| subscription | — | `subscription.change`, `subscription.preview`, `subscription.resume`, `subscription.state`, `subscription.upgrade` |
| system | — | `system.battery` |
| terminal | — | `terminal.resize` |
| tools | — | `tools.configure`, `tools.list`, `tools.show` |
| toolsets | — | `toolsets.list` |
| usage | — | `usage.bars` |
| vault | — | `vault.add`, `vault.list`, `vault.lock`, `vault.preview.fill`, `vault.remove`, `vault.source.set`, `vault.sources`, `vault.unlock` |
| verification | — | `verification.status` |
| voice | — | `voice.record`, `voice.toggle`, `voice.tts` |
| wake | — | `wake.feed`, `wake.pause`, `wake.resume`, `wake.start`, `wake.status`, `wake.stop` |

Rust also advertises `gateway.ping`, `prompt.cancel`, `prompt.receipt`, `rpc.discover`, `runtime.describe`, `session.info`, `session.mutate`. These are outside this generated registry.
