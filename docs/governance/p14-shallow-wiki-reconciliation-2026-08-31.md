# P1.4 Shallow Wiki Reconciliation Report

Programme: evidence-spine-programme-2026-08-30, Phase P1.4 (lines 75-79).
Date: 2026-08-31. Status: read-only reconciliation; **no archive, symlink, import, edit, or rename has been performed on either tree**.

## Purpose

The Git-backed wiki at `/home/kensei/docs/wiki` (origin `https://github.com/Sahil-SS9/kensei-wiki.git`) is the canonical wiki authority. A shallow duplicate tree at `/home/kensei/wiki` still exists and a live LLM cron (gitradar ingest) still writes new entries into it. Before any future archive/symlink migration, this report freezes a per-item classification of every file that exists ONLY in the shallow tree, so no content can be silently lost.

## Method

1. Diffed real file listings of both live trees (read-only): `find <tree> -type f -name '*.md'` sorted and `comm -23` to get shallow-only paths.
2. For each shallow-only file, extracted `upstream_url:` from the YAML frontmatter (all 110 files carry one).
3. Built a canonical index of `repos/*.md` pages keyed by normalised `upstream_url` (lowercased, trailing slash stripped).
4. Matched each candidate by URL; a slug (filename stem) fallback match was also checked. Any candidate with neither a URL nor a match is classified `no-url` / `unique` and would require manual salvage before any archive step.
5. Re-ran the same computation deterministically as a controller cross-check (independent rerun, same classification).

## Counts (current run, 2026-08-31)

| Metric | Value |
|---|---|
| Baseline candidate count (programme intake) | 110 |
| Current shallow-only candidate count | 110 |
| Delta vs baseline | **0** |
| Classified `mapped` (canonical page exists for the upstream URL) | 110 |
| Classified `genuinely unique` (no canonical page for its URL) | 0 |
| Classified `no-url` (no upstream_url in frontmatter) | 0 |

The delta is zero: no shallow-only files were added or removed since the baseline manifest was captured. Every one of the 110 candidates resolves to an existing canonical `repos/<slug>.md` page by normalised GitHub upstream URL (110/110 by URL match; slug fallback never needed; all 110 canonical target files verified present on disk).

Tree sizes at time of report: canonical `/home/kensei/docs/wiki/repos/` = 8,938 repo pages (9,923 md files tree-wide); shallow `/home/kensei/wiki/repos/` = 493 files (500 files tree-wide).

## Canonical sync path evidence

- `scripts/wiki_daily_sync.sh` target: `WIKI_DIR="${WIKI_SYNC_DIR:-/home/kensei/docs/wiki}"` — defaults to the Git-backed canonical tree; overridable only for disposable runs (`WIKI_SYNC_DIR`).
- Guard rail in the same script: refuses unsafe origin — `EXPECTED_REMOTE="${WIKI_SYNC_EXPECTED_REMOTE:-https://github.com/Sahil-SS9/kensei-wiki.git}"` is enforced against the live `git remote get-url origin` before any commit/push; fail-closed on mismatch.
- Live tree verified: `/home/kensei/docs/wiki` origin = `https://github.com/Sahil-SS9/kensei-wiki.git`, branch `main`, HEAD `1d8cba8` (`wiki sync: 2026-08-31_04:45:34`) — the daily sync is actively committing to the canonical tree.
- P1.4 authority change (this commit): `wiki_on_complete.py`, `approval_handler.py`, `brain-to-wiki-synthesis.py`, `kensei_review_daily.py` default to `~/docs/wiki` when `WIKI_DIR` / `KENSEI_WIKI_ROOT` are unset (overrides preserved, no new env var). Per controller scope correction, the bundled active contract `skills/research/llm-wiki/SKILL.md` (all `WIKI_PATH` default + Git-sync/Obsidian examples), `scripts/research_paper_preprocess.py` (WIKI_PATH default), and `idea_box/flow.py` (wiki fallback default) were aligned to `$HOME/docs/wiki` in the same commit. The live-root skill at `/home/kensei/.hermes/skills` and live cron prompts are deployment surfaces — untouched here.

## Consumer / writer inventory (candidate repo)

Every current wiki consumer and writer found in the candidate worktree `scripts/`, plus the live cron jobs that invoke them (`/home/kensei/.hermes/cron/jobs.json`, read-only):

| Consumer/writer | Tree it resolves by default | Role | P1.4 action |
|---|---|---|---|
| `scripts/wiki_on_complete.py` (cron `wiki-on-complete`) | was `~/wiki`, now `~/docs/wiki` (WIKI_DIR override kept) | writer (repo page frontmatter updates) | **changed** |
| `scripts/approval_handler.py` (cron `discord-approval-handler`) | was `~/wiki`, now `~/docs/wiki` (WIKI_DIR override kept) | writer (wiki page rewrite on approval) | **changed** |
| `scripts/brain-to-wiki-synthesis.py` (cron `brain-to-wiki-synthesis`) | was `~/wiki`, now `~/docs/wiki` (WIKI_DIR override kept) | writer (concepts/comparisons/index/log) | **changed** |
| `scripts/kensei_review_daily.py` (cron `kensei-review-daily`) | was `/home/kensei/wiki`, now `~/docs/wiki` (KENSEI_WIKI_ROOT override kept) | reader (`_meta/paper-mashups.md`) | **changed** |
| `scripts/wiki_daily_sync.sh` (cron `wiki-daily-sync`) | already `/home/kensei/docs/wiki` + remote-pinned | writer (git commit+push of canonical tree) | verified, no change |
| `scripts/archive/gitradar-upstream-monitor.py` | primary: `<repo>/runbooks/github-radar/repos`; legacy fallback read: `~/wiki/repos` | reader (archived/backward-compat path only) | none (archived; legacy read only) |
| `scripts/research_paper_preprocess.py` (cron `research-paper-synthesis-daily`) | was `~/wiki`, now `~/docs/wiki` (WIKI_PATH override kept) | reader (memory-gate wiki evidence lookup) | **changed** (controller scope correction) |
| `skills/research/llm-wiki/SKILL.md` (bundled active contract) | was `~/wiki`, now `$HOME/docs/wiki` (WIKI_PATH override kept) | authority contract steering agents/crons | **changed** (controller scope correction) |
| `skills/research/research-paper-synthesis/SKILL.md` | was `~/wiki`, now `~/docs/wiki` | active synthesis contract steering paper and mashup writes | **changed** (controller completion correction) |
| `idea_box/flow.py` wiki fallback | was `~/wiki`, now `~/docs/wiki` (WIKI_PATH override kept) | writer (idea provenance fallback) | **changed** (controller scope correction) |
| `content_engine/kb_retrieve.py` | was `~/wiki`, now `~/docs/wiki` (WIKI_PATH override kept) | reader (personal-content knowledge retrieval) | **changed** (controller completion correction) |
| `scripts/ceecee_approval_handler.py`, `scripts/proposal_approval_handler.py` | n/a — no wiki references | not wiki consumers | none |

Live cron prompts that reference a wiki tree (LLM-driven, no script): `gitradar` repo-ingest prompt **still instructs writing new entries to `~/wiki/repos/`** — this is the remaining shallow-tree writer producing new shallow-only files; `kensei-mashup-review` reads `/home/kensei/wiki/_meta/paper-mashups.md`; `MrHermagi Daily Lesson` and `kensei-librarian-daily` read `~/wiki`; `research-paper-synthesis-daily` reads `~/wiki` via its prompt text but its preprocess script now defaults to the canonical tree; `knowledge-weekly-digest` already reads `~/docs/wiki`. These prompt/config-only live consumers are documented for the P1.6 live-runtime phase — live cron state was deliberately NOT edited during P1.4 (candidate repo change only; deployment is separate).

## Per-item mapping (all 110 shallow-only candidates)

Columns: shallow file (relative to `/home/kensei/wiki/`), upstream URL from its frontmatter, canonical page (relative to `/home/kensei/docs/wiki/repos/`), match basis. All rows are `mapped`; the unique/no-url rows would appear here with their reason and are absent because none exist.

| # | Shallow file | Upstream URL | Canonical page | Match |
|---|---|---|---|---|
| 1 | repos/0xshug0-audio.cpp.md | https://github.com/0xShug0/audio.cpp | repos/audio.cpp.md | url |
| 2 | repos/aaryanverma-graybox.md | https://github.com/Aaryanverma/graybox | repos/Aaryanverma-graybox.md | url |
| 3 | repos/aaswordman-operit2.md | https://github.com/AAswordman/Operit2 | repos/operit2.md | url |
| 4 | repos/about-intelligence-soku-cli.md | https://github.com/About-Intelligence/soku-cli | repos/soku-cli.md | url |
| 5 | repos/accio-lab-dressage.md | https://github.com/Accio-Lab/Dressage | repos/Accio-Lab-Dressage.md | url |
| 6 | repos/alpha-dojo-dojoagents.md | https://github.com/Alpha-Dojo/DojoAgents | repos/dojoagents.md | url |
| 7 | repos/alphaxiv-openresearch-cli.md | https://github.com/alphaXiv/openresearch-cli | repos/openresearch-cli.md | url |
| 8 | repos/anionex-codex-deepseek-vision.md | https://github.com/Anionex/codex-deepseek-vision | repos/codex-deepseek-vision.md | url |
| 9 | repos/archastro-scopey.md | https://github.com/ArchAstro/scopey | repos/ArchAstro-scopey.md | url |
| 10 | repos/asdecided-wayfinderrouter.md | https://github.com/asdecided/WayfinderRouter | repos/asdecided-WayfinderRouter.md | url |
| 11 | repos/asterove-astermem.md | https://github.com/Asterove/AsterMem | repos/Asterove-AsterMem.md | url |
| 12 | repos/atomic-mail-atomic-mail-agentic.md | https://github.com/Atomic-Mail/atomic-mail-agentic | repos/atomic-mail-agentic.md | url |
| 13 | repos/autoloops-greplica.md | https://github.com/Autoloops/greplica | repos/greplica.md | url |
| 14 | repos/auucoder-gptgrok2api.md | https://github.com/AuuCoder/gptGrok2api | repos/AuuCoder-gptGrok2api.md | url |
| 15 | repos/benjam1ncup-polymarket-trading-bot-python-v2.md | https://github.com/Benjam1nCup/Polymarket-trading-bot-python-V2 | repos/polymarket-trading-bot-python-v2.md | url |
| 16 | repos/bino5150-lumina.md | https://github.com/Bino5150/lumina | repos/Bino5150-lumina.md | url |
| 17 | repos/capitalone-vulnhunter.md | https://github.com/capitalone/VulnHunter | repos/vulnhunter.md | url |
| 18 | repos/chi11321-crabport.md | https://github.com/chi11321/CrabPort | repos/crabport.md | url |
| 19 | repos/cogeto-cogeto.md | https://github.com/Cogeto/cogeto | repos/Cogeto-cogeto.md | url |
| 20 | repos/dannymac180-sol-advisor.md | https://github.com/DannyMac180/sol-advisor | repos/sol-advisor.md | url |
| 21 | repos/deepender25-edge-drop.md | https://github.com/Deepender25/Edge-Drop | repos/Deepender25-Edge-Drop.md | url |
| 22 | repos/deviffyy-openquota.md | https://github.com/deviffyy/OpenQuota | repos/deviffyy-OpenQuota.md | url |
| 23 | repos/echovic-orca-agent.md | https://github.com/echoVic/orca-agent | repos/echoVic-orca-agent.md | url |
| 24 | repos/encod3d-sec-claudebrain.md | https://github.com/Encod3d-Sec/ClaudeBrain | repos/claudebrain.md | url |
| 25 | repos/extraltodeus-j-wash.md | https://github.com/Extraltodeus/J-Wash | repos/j-wash.md | url |
| 26 | repos/ferroxlabs-wayland.md | https://github.com/FerroxLabs/wayland | repos/wayland.md | url |
| 27 | repos/francis1998-nexus-llm-router.md | https://github.com/Francis1998/nexus-llm-router | repos/nexus-llm-router.md | url |
| 28 | repos/fuji-mak-capsomnia.md | https://github.com/fuji-mak/Capsomnia | repos/fuji-mak-Capsomnia.md | url |
| 29 | repos/fullive-ai-anima.md | https://github.com/Fullive-AI/Anima | repos/Fullive-AI-Anima.md | url |
| 30 | repos/getbusbar-busbar.md | https://github.com/GetBusbar/busbar | repos/GetBusbar-busbar.md | url |
| 31 | repos/guoyijia22-crabrag.md | https://github.com/guoyijia22/CrabRAG | repos/crabrag.md | url |
| 32 | repos/haseebkhalid1507-myx.md | https://github.com/HaseebKhalid1507/Myx | repos/HaseebKhalid1507-Myx.md | url |
| 33 | repos/hikari-systems-slater.md | https://github.com/Hikari-Systems/slater | repos/Hikari-Systems-slater.md | url |
| 34 | repos/hongjin-he-microworld.md | https://github.com/hongjin-he/MicroWorld | repos/microworld.md | url |
| 35 | repos/iamcorey-kooky.md | https://github.com/iAmCorey/kooky | repos/kooky.md | url |
| 36 | repos/ilhamrisky-tedi.md | https://github.com/IlhamriSKY/TEDI | repos/IlhamriSKY-TEDI.md | url |
| 37 | repos/inakitajes-convoy.md | https://github.com/Inakitajes/convoy | repos/Inakitajes-convoy.md | url |
| 38 | repos/inclusionai-areno.md | https://github.com/inclusionAI/AReno | repos/inclusionAI-AReno.md | url |
| 39 | repos/inclusionai-avernet.md | https://github.com/inclusionAI/Avernet | repos/inclusionAI-Avernet.md | url |
| 40 | repos/jayleebot-forgeflow.md | https://github.com/JayleeBot/ForgeFlow | repos/forgeflow.md | url |
| 41 | repos/joeynyc-grok-ui.md | https://github.com/joeynyc/Grok-UI | repos/joeynyc-Grok-UI.md | url |
| 42 | repos/juanjuandog-finsight-ai.md | https://github.com/juanjuandog/FinSight-AI | repos/finsight-ai.md | url |
| 43 | repos/jyh1878-kimicodebar-windows.md | https://github.com/JYH1878/KimiCodeBar-Windows | repos/JYH1878-KimiCodeBar-Windows.md | url |
| 44 | repos/kiyoshithedevil-kodama.md | https://github.com/KiyoshiTheDevil/Kodama | repos/kodama.md | url |
| 45 | repos/kmno4-zx-llm-agent-rl-lab.md | https://github.com/KMnO4-zx/llm-agent-rl-lab | repos/KMnO4-zx-llm-agent-rl-lab.md | url |
| 46 | repos/krishagarwal314-codejury.md | https://github.com/krishagarwal314/CodeJury | repos/codejury.md | url |
| 47 | repos/kritt-ai-open-kritt.md | https://github.com/Kritt-ai/open-kritt | repos/open-kritt.md | url |
| 48 | repos/kvcache-ai-agentenv.md | https://github.com/kvcache-ai/AgentENV | repos/agentenv.md | url |
| 49 | repos/linzwcs-evopolicygym.md | https://github.com/Linzwcs/EvoPolicyGym | repos/evopolicygym.md | url |
| 50 | repos/liuyanghejerry-clausura.md | https://github.com/liuyanghejerry/Clausura | repos/liuyanghejerry-Clausura.md | url |
| 51 | repos/lucifer1004-veloq.md | https://github.com/lucifer1004/VeloQ | repos/lucifer1004-VeloQ.md | url |
| 52 | repos/marktechpost-token-saver.md | https://github.com/Marktechpost/Token-Saver | repos/Marktechpost-Token-Saver.md | url |
| 53 | repos/marshallbear1-react-native-system-thumbnails.md | https://github.com/MarshallBear1/react-native-system-thumbnails | repos/MarshallBear1-react-native-system-thumbnails.md | url |
| 54 | repos/matinsenpai-aether-gui.md | https://github.com/MatinSenPai/Aether-GUI | repos/MatinSenPai-Aether-GUI.md | url |
| 55 | repos/mcg-nju-videochat3.md | https://github.com/MCG-NJU/VideoChat3 | repos/videochat3.md | url |
| 56 | repos/menfre01-waveloom.md | https://github.com/Menfre01/waveloom | repos/Menfre01-waveloom.md | url |
| 57 | repos/mighty-alien-ethereum-trading-bot.md | https://github.com/MIgHTy-alIeN/Ethereum-Trading-Bot | repos/MIgHTy-alIeN-Ethereum-Trading-Bot.md | url |
| 58 | repos/mixar-ai-mixar-app.md | https://github.com/Mixar-AI/mixar-app | repos/Mixar-AI-mixar-app.md | url |
| 59 | repos/nadzimouad-the-pit.md | https://github.com/NadziMouad/The-Pit | repos/the-pit.md | url |
| 60 | repos/nanako0129-pilotfish.md | https://github.com/Nanako0129/pilotfish | repos/pilotfish.md | url |
| 61 | repos/nanonets-graft.md | https://github.com/NanoNets/Graft | repos/NanoNets-Graft.md | url |
| 62 | repos/nazzarenogiannelli-tuiboard.md | https://github.com/NazzarenoGiannelli/tuiboard | repos/NazzarenoGiannelli-tuiboard.md | url |
| 63 | repos/neuroaihub-brainpilot.md | https://github.com/NeuroAIHub/BrainPilot | repos/brainpilot.md | url |
| 64 | repos/nextweb4-official-document-ai-assistant.md | https://github.com/NextWeb4/official-document-ai-assistant | repos/NextWeb4-official-document-ai-assistant.md | url |
| 65 | repos/noelo-lab-kuna.md | https://github.com/Noelo-Lab/kuna | repos/kuna.md | url |
| 66 | repos/nuber-dev-ytubic.md | https://github.com/NUber-dev/YTubic | repos/ytubic.md | url |
| 67 | repos/nvidia-nemo-labs-oo-agents.md | https://github.com/NVIDIA-NeMo/labs-OO-Agents | repos/NVIDIA-NeMo-labs-OO-Agents.md | url |
| 68 | repos/onlyterp-ultracode-shim.md | https://github.com/OnlyTerp/UltraCode-Shim | repos/OnlyTerp-UltraCode-Shim.md | url |
| 69 | repos/openbmb-pilotdeck.md | https://github.com/OpenBMB/PilotDeck | repos/pilotdeck.md | url |
| 70 | repos/openhelix-team-robomemarena.md | https://github.com/OpenHelix-Team/RoboMemArena | repos/OpenHelix-Team-RoboMemArena.md | url |
| 71 | repos/org2ai-org2.md | https://github.com/org2AI/ORG2 | repos/org2.md | url |
| 72 | repos/paritok-official-paritok-4b-v1.md | https://github.com/Paritok-official/paritok-4b-v1 | repos/paritok-4b-v1.md | url |
| 73 | repos/petyok-sshub.md | https://github.com/Petyok/SSHub | repos/sshub.md | url |
| 74 | repos/pku-yuangroup-openai4s.md | https://github.com/PKU-YuanGroup/OpenAI4S | repos/PKU-YuanGroup-OpenAI4S.md | url |
| 75 | repos/playa-0v0-cyrene-agent.md | https://github.com/Playa-0v0/Cyrene-Agent | repos/Playa-0v0-Cyrene-Agent.md | url |
| 76 | repos/powerycy-bosshunter.md | https://github.com/powerycy/BossHunter | repos/bosshunter.md | url |
| 77 | repos/prithvi-web-treemap-disk-visualizer.md | https://github.com/Prithvi-Web/TreeMap-Disk-Visualizer | repos/treemap-disk-visualizer.md | url |
| 78 | repos/pulseio76-argusmind.md | https://github.com/pulseio76/ArgusMind | repos/pulseio76-ArgusMind.md | url |
| 79 | repos/qwenaudio-qwen-audio-agent.md | https://github.com/QwenAudio/qwen-audio-agent | repos/QwenAudio-qwen-audio-agent.md | url |
| 80 | repos/rapiercraftstudios-forgedock.md | https://github.com/RapierCraftStudios/ForgeDock | repos/forgedock.md | url |
| 81 | repos/rfhdw0102-youdaonotelm.md | https://github.com/rfhdw0102/YouDaoNoteLM | repos/rfhdw0102-YouDaoNoteLM.md | url |
| 82 | repos/rizzo-ai-academy-rizzo-pii.md | https://github.com/Rizzo-AI-Academy/rizzo-pii | repos/rizzo-pii.md | url |
| 83 | repos/rlinf-rpent.md | https://github.com/RLinf/RPent | repos/rpent.md | url |
| 84 | repos/rodiun-frugon.md | https://github.com/Rodiun/frugon | repos/Rodiun-frugon.md | url |
| 85 | repos/rubric4setwise-rubric4setwise.md | https://github.com/Rubric4Setwise/Rubric4Setwise | repos/rubric4setwise.md | url |
| 86 | repos/rubywang0-windrise.md | https://github.com/RubyWang0/WindRise | repos/windrise.md | url |
| 87 | repos/salomondiei08-oh-my-hermes.md | https://github.com/Salomondiei08/oh-my-hermes | repos/Salomondiei08-oh-my-hermes.md | url |
| 88 | repos/saxy-tellstone.md | https://github.com/Saxy/Tellstone | repos/tellstone.md | url |
| 89 | repos/shaf2665-hermes-router.md | https://github.com/Shaf2665/Hermes-router | repos/hermes-router.md | url |
| 90 | repos/shenseanchen-waku-agent.md | https://github.com/ShenSeanChen/waku-agent | repos/ShenSeanChen-waku-agent.md | url |
| 91 | repos/simonlin1212-tradingagents-astock.md | https://github.com/simonlin1212/TradingAgents-astock | repos/simonlin1212-TradingAgents-astock.md | url |
| 92 | repos/sina2266-gozar.md | https://github.com/sina2266/Gozar | repos/sina2266-Gozar.md | url |
| 93 | repos/syrizelink-openfic.md | https://github.com/syrizelink/OpenFic | repos/syrizelink-OpenFic.md | url |
| 94 | repos/tejas-ta-predikit.md | https://github.com/Tejas-TA/predikit | repos/predikit.md | url |
| 95 | repos/tencent-hunyuan-hyra-results.md | https://github.com/Tencent-Hunyuan/Hyra-results | repos/Tencent-Hunyuan-Hyra-results.md | url |
| 96 | repos/theorcdev-shadscan.md | https://github.com/TheOrcDev/shadscan | repos/shadscan.md | url |
| 97 | repos/tinysuitehq-tinysearch.md | https://github.com/TinySuiteHQ/TinySearch | repos/TinySuiteHQ-TinySearch.md | url |
| 98 | repos/tura-ai-tura.md | https://github.com/Tura-AI/tura | repos/tura.md | url |
| 99 | repos/u-c4n-u-pool.md | https://github.com/U-C4N/U-Pool | repos/u-pool.md | url |
| 100 | repos/ufy2024-auc.md | https://github.com/ufy2024/AuC | repos/ufy2024-AuC.md | url |
| 101 | repos/victortaelin-optmem.md | https://github.com/VictorTaelin/OptMem | repos/VictorTaelin-OptMem.md | url |
| 102 | repos/wanshuiyin-aris-in-ai-offer.md | https://github.com/wanshuiyin/ARIS-in-AI-Offer | repos/aris-in-ai-offer.md | url |
| 103 | repos/whitenightshadow-firefox-reverse.md | https://github.com/WhiteNightShadow/firefox-reverse | repos/WhiteNightShadow-firefox-reverse.md | url |
| 104 | repos/wutishummus-truedeck.md | https://github.com/WutIsHummus/TrueDeck | repos/truedeck.md | url |
| 105 | repos/xiaol-multi-state-rwkv-online-memory.md | https://github.com/xiaol/Multi-state-RWKV-online-memory | repos/xiaol-Multi-state-RWKV-online-memory.md | url |
| 106 | repos/xyz-ai-lab-axrl.md | https://github.com/XYZ-AI-Lab/axrl | repos/axrl.md | url |
| 107 | repos/yliust-tactile.md | https://github.com/yliust/Tactile | repos/yliust-Tactile.md | url |
| 108 | repos/zhuzhaoyun-molio.md | https://github.com/zhuzhaoyun/Molio | repos/zhuzhaoyun-Molio.md | url |
| 109 | repos/zli12321-lhtb.md | https://github.com/zli12321/LHTB | repos/lhtb.md | url |
| 110 | repos/zzliu93-debug-focusd.md | https://github.com/zzliu93-debug/FocuSD | repos/focusd.md | url |

## Data-loss guard for the future migration

When the archive/symlink phase eventually runs, the safe minimum is: for each of the 110 rows above, confirm the canonical page carries the shallow page's `classification` / `adoption_status` / `why_it_matters` fields before deleting the shallow copy — or, more conservatively, `git mv` the shallow copy into `/home/kensei/docs/wiki/_archive/shallow-2026-08/` and reconcile content afterwards. This report is the durable item-level checklist for that step; both trees were left byte-for-byte untouched during P1.4.

## Test evidence

- RED (witnessed, before any source change): `scripts/run_tests.sh tests/scripts/test_wiki_authority_defaults.py -q` → **4 failed** (all four scripts resolved the shallow default).
- GREEN (final authority contract): `scripts/run_tests.sh content_engine/tests/test_kb_retrieve.py tests/scripts/test_wiki_authority_defaults.py tests/idea_box/test_flow.py -q` → **50 passed, 0 failed**. The authority file itself contains 13 tests: 5 canonical-default behaviours, 5 explicit-override behaviours, 1 alternate-HOME guard and 2 bundled-skill contract cases.
- Wider P1.4 regression: nine directly affected test files → **125 passed, 0 failed** using the canonical runner.
- Override contracts preserved: `WIKI_DIR` (three scripts), `WIKI_PATH` (research preprocessor, Idea Box and content knowledge retrieval), and `KENSEI_WIKI_ROOT` (kensei_review_daily.py) still take precedence over the default; no new environment variable introduced.
- Tests use disposable `HOME`/`HERMES_HOME` (`monkeypatch.setenv`) only; no live tree or live `HERMES_HOME` is touched by the test suite.
