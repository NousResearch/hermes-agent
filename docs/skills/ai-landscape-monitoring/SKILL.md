# AI Landscape Monitoring

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Define the framework for evaluating AI industry developments relevant to Wake Robin's investment and operational activities. Two routines \(Friday AI Digest, Weekly AI Research Digest\) actively produce AI landscape intelligence; this skill defines *how* to evaluate AI claims, *what* to watch for, and *how* to distinguish hype from genuine capability.

---

## Monitoring Domains

### Domain 1: AI Tools for Investment Management

| Topic | Key Question | Sources |
| --- | --- | --- |
| Claude Code / AI coding | How does it affect DEM development velocity? | Anthropic blog, LinkedIn |
| AI agents for finance | Which agent architectures produce real alpha? | arXiv, Springer, SSRN |
| LLM for equity research | Can LLMs reliably analyze biotech pipelines? | HBR, MIT Tech Review |
| AI portfolio construction | Does AI improve construction beyond selection? | Academic finance journals |
| Quant finance + AI | Which quant strategies benefit from LLM augmentation? | AQR, SSRN, arXiv |

### Domain 2: AI Infrastructure & Operations

| Topic | Key Question | Sources |
| --- | --- | --- |
| Hermes/OpenClaw ecosystem | What are the security and capability trends? | GitHub, security advisories |
| Town AI Assistant platform | How do Town's capabilities evolve vs competitors? | town.com, BetaList, App Store, LinkedIn |
| Local vs cloud LLMs | When does local deployment break even? | Benchmarks, cost analyses |
| Agent memory systems | Which persistent memory architectures work? | arXiv, Nous Research |
| Model routing | How to select models by task type? | Benchmarks, production data |
| AI security | What attack vectors affect agent systems? | Texas A&M taxonomy, CVEs |
| Parallel agent orchestration | When does N-bot parallelism ("AI vampires") outperform serial workflows? | Practitioner reports, benchmarks, Hermes production data |

### Domain 3: AI in Wake Robin's Business Domains

| Topic | Key Question | Sources |
| --- | --- | --- |
| AI for real estate | How is AI changing multifamily/seniors housing? | ULI, NMHC, proptech news |
| AI for family offices | Which AI tools are SFOs actually adopting? | Family office surveys, FOX |
| AI biotech stock selection | Can AI improve clinical trial outcome prediction? | arXiv, biotech journals |
| AI regulatory impact | How are FDA, SEC, DOL responding to AI? | Federal Register, regulatory news |
| Medical AI consumer adoption | How fast are clinicians and consumers adopting LLM-based diagnostics? Investment implications for GH, VCYT, digital health, precision medicine. | FDA guidance, app store data, provider surveys, earnings calls |

---

## Evaluation Framework: Hype vs Substance

### The Institutional Rigor Test

Apply CFA-grade skepticism to AI performance claims:

| Question | Red Flag | Green Flag |
| --- | --- | --- |
| Is there out-of-sample evidence? | Only in-sample or backtest | True OOS or live deployment |
| Is it net of costs? | Gross returns only | Net of transaction costs, API costs |
| Is sample size sufficient? | n < 30 or unclear | n > 100, statistically significant |
| Is there regime conditioning? | Only tested in bull markets | Tested across regimes |
| Is the methodology replicable? | Black box, no code | Open source or detailed methodology |
| Is there survivorship bias? | Only surviving companies | Full universe at decision time |
| Who funded the research? | AI vendor self-reporting | Independent academic or practitioner |
| Is the benchmark appropriate? | vs cash or wrong benchmark | vs relevant investable alternative |

### Claim Classification

| Category | Definition | Action |
| --- | --- | --- |
| VALIDATED | Multiple independent confirmations, live deployment, open methodology | Monitor for integration opportunities |
| PROMISING | Strong methodology, limited OOS, no obvious red flags | Track, evaluate in 6 months |
| UNPROVEN | In-sample only, vendor claims, no independent validation | Note, do not cite as evidence |
| HYPE | No methodology, unreplicable claims, obvious conflicts of interest | Ignore, note as cautionary example |
| DEAD | Tested and failed \(like Spec 053 options-as-alpha\) | Do not revisit without new data |

---

## Notable Industry Signals Log

High-profile claims captured for tracking through the Institutional Rigor Test. Each entry records the source, the claim, initial classification, and what evidence would move it up or down.

### Marc Andreessen on Joe Rogan (May 2026, ~3h20m)

Source doc: [Andreessen/Rogan AI Alpha Notes](https://www.town.com/content/file/sh7ax76c78x9d6d397dt8b7g2s878d4j) in ai-projects collection.

| # | Claim | Initial Classification | Evidence Needed to Upgrade/Downgrade |
| --- | --- | --- | --- |
| 1 | "AGI is here" — crossed ~3 months ago with GPT-5.5, Claude 4.6, Gemini 3, Grok 4.3 | HYPE | No agreed definition of AGI applied; no benchmark cited; speaker is a16z GP with massive portfolio exposure to the claim being true. Would need independent benchmark suite showing qualitative capability break vs prior generation. |
| 2 | Top AIs give better answers than world-class human experts on almost any topic | UNPROVEN | Anecdotal (n=1 user). Would need controlled expert-vs-LLM comparison studies across domains. Some domain-specific evidence exists (radiology, legal Q&A) but "almost any topic" is unfalsifiable as stated. |
| 3 | Doctors already secretly using ChatGPT in exam rooms | PROMISING | Consistent with AMA 2025 survey (45% of physicians experimenting with LLMs). "Secretly" and "already" overstate adoption curve but directional signal is real. Watch for FDA guidance on clinical LLM use. Investment signal: bullish GH, VCYT, diagnostic AI. |
| 9 | "The only real skill left is knowing what to ask" — bottleneck is in the user's head | VALIDATED | This is the core thesis behind DEM's skill-encoded domain expertise. The moat for institutional investors is encoding the right questions (clinical pipeline, 13F filing analysis, catalyst timing) into agent prompts. Independently confirmed by production experience. |
| 12 | AI solving 100+ year open math problems; expect cancer cures, new drugs, physics breakthroughs | PROMISING (math) / UNPROVEN (applied science) | AlphaProof/AlphaGeometry IMO results are real. Protein folding (AlphaFold) is real. "Cancer cures" and "weird new physics" within "next few years" is aspirational. Track drug discovery pipeline (Recursion, Insilico) for evidence. |
| 13 | Best AI coders make $50M/year — signals how big this is | UNPROVEN | No named examples or comp data cited. Directionally consistent with a16z portfolio company comp trends but $50M is an extraordinary claim requiring extraordinary evidence. Watch for public comp disclosures. |
| 16-17 | "AI vampires" running 20 parallel bots; bots-running-bots (1000 AI workers) months away | PROMISING (parallel bots) / UNPROVEN (recursive delegation at scale) | Parallel coding bots are real and in use (Hermes fleet, Cursor multi-agent, Devin). 20 simultaneous is plausible for code review loops. 1000-agent recursive hierarchies "months away" is speculative — context window limits, error propagation, and coordination overhead are unsolved at that scale. |
| 14 | $200 DNA decode + AI health dashboard from DNA + blood + wearable data | PROMISING | Whole genome sequencing is ~$200 (Nebula, Dante Labs). LLM interpretation of combined genomic + phenotypic data is early but real. Clinical validity of AI-generated health recommendations from consumer genomics is UNPROVEN. Watch FDA regulation of DTC genomic AI interpretation. |

**Net assessment:** Andreessen is a16z's chief evangelist; every claim benefits his portfolio. Apply maximum conflict-of-interest discount. That said, claims #3 (medical AI adoption), #9 (question-asking as bottleneck), and #16 (parallel agent orchestration) are directionally validated by independent evidence and DEM production experience. Track #12 and #14 for biotech investment signal.

---



## Competitive Positioning

### Hermes vs Commercial Alternatives \(May 2026\)

| System | Architecture | Differentiator | Concern |
| --- | --- | --- | --- |
| Hermes/OpenClaw \(DEM\) | 30-agent fleet, repo-governed, CCFT | Full governance control, audit trail | Operator-dependent, WSL2 infrastructure |
| Town AI Assistant | Cloud SaaS, routine-based, multi-platform | Graduated autonomy, meeting capture, audio briefings, 39 skills, persistent memory | Integration breadth \(\~12 vs 200+\), near-zero market visibility, post-beta pricing unknown, platform dependency risk |
| Nous Research Hermes | Self-learning skills, MIT license | Skill auto-creation, large community | Governance-incompatible with CCFT \(unreviewed skill mutations\) |
| ChatGPT/Claude direct | Single model, conversational | Ease of use, frontier capability | No persistent memory, no governance |
| Bloomberg Terminal AI | Integrated data + AI | Data advantage | Closed ecosystem, no customization |
| Kensho/S&P AI | NLP on financial documents | Scale, data coverage | Enterprise pricing, no biotech specialization |

### DEM Competitive Advantages

1. **Governance-first:** Every signal change goes through Checklist v2, spec system, and operator approval
2. **PIT-safe by design:** No other system enforces point-in-time as rigorously
3. **Domain-specialized:** 48-manager biotech hedge fund registry, clinical trial pipeline integration
4. **Transparent:** All scoring rules encoded in skills, all evidence documented

### DEM Competitive Disadvantages

1. **Single-operator dependency:** Darren is sole governance authority
2. **Infrastructure fragility:** WSL2 on Windows, sleep-cliff risk, Herald DARK 5+ weeks
3. **No live trading integration:** Shadow portfolio only \(Alpaca API available but not wired\)
4. **Limited NLP:** No earnings call NLP, no news sentiment \(Herald pipeline is down\)

---

### Town AI Assistant — Platform Profile \(May 2026\)

**Company:** Town \(town.com\), San Francisco, \~29 employees.
**Founders:**

- Jean-Denis Greze \(CEO\) — Former CTO of Plaid \(2017-2024\), Director of Engineering at Dropbox \(2013-2017\). Columbia CS + Harvard Law.
- Tony Vincent \(CPO\) — Former Director of Product, Applied AI at Google; Head of Product Design at Dropbox. MS HCI from MIT.

**Funding:** $18M Series A led by First Round Capital \(March 2025\). Angels: Adam D'Angelo \(Quora/early Facebook CTO\), William Hockey \(Plaid co-founder\), Immad Akhund \(Mercury\), Christina Cacioppo \(Vanta\), Soleio Cuervo, Helen Min.

**Pivot history:** Originally launched as AI-powered tax advisory for SMBs \(2024\). Pivoted to general-purpose AI work assistant. May 2025 podcast \("Ex-Plaid CTO on Why Tax is the Ultimate AI Challenge"\) captured original vision.

**Current product:** Multi-platform AI work assistant integrating email, calendar, docs, and workflow automation.

#### Core Capabilities

| Capability | Detail |
| --- | --- |
| Email triage | Auto-scan, label by type, identify reply-worthy, draft responses |
| Voice-matched drafting | Persistent style profile; drafts in user's voice from email history |
| Calendar management | Create/edit/RSVP, scheduling, meeting prep |
| Meeting capture | iOS voice recording, transcription, structured summaries, action items with owners |
| Audio briefings | Generated spoken summaries pulling from calendar, email threads, docs |
| Routine automation | Trigger-based \(incoming email, calendar events, schedules, RSVP changes\) with configurable tools and approval modes |
| Content library | Structured content management \(collections, documents, files\) integrated into assistant context |
| Multi-platform access | Web, email \(dedicated address\), Slack, iOS, WhatsApp, desktop \(macOS\) |
| Control modes | Read-only, approval-required \(HITL\), autonomous — per-tool granularity |
| Memory/personalization | Persistent user profile + routine-specific memories across sessions |
| Privacy | No model training on user data; full action logs with reasoning; audit trails |

#### Integrations \(as of May 2026\)

Gmail, Google Calendar, Google Drive, Google Docs, Slack, Notion, Dropbox, GitHub, Linear, HubSpot, Asana, MCP protocol servers \(extensible\). \~12 named integrations vs Lindy's 200+.

#### Town vs AI Email/Productivity Competitor Landscape \(2026\)

**Three competitive camps:**

1. **Full Workflow Automation \(Agents\):** Lindy, Consul, Carly — autonomous agents acting across apps
2. **AI-First Email Clients:** Superhuman, Shortwave, Spark — replace email interface with AI-enhanced version
3. **Assistant Layers:** SaneBox, Fyxer, Gemini, Copilot — augment existing inbox

**Town straddles camps 1 and 3** — works within existing tools \(not a replacement client\) but offers full workflow automation with graduated autonomy.

| Dimension | Town | Lindy \($50/mo\) | Superhuman \($30/mo\) | Shortwave \($8-24/mo\) | Consul | Fyxer \($22/mo\) |
| --- | --- | --- | --- | --- | --- | --- |
| Autonomy levels | 3 modes, per-tool | Full agent | Hybrid \(user-initiated\) | Hybrid | Approval-only | Limited |
| Email triage | Yes | Yes | Yes | Yes | Yes | Yes |
| Voice-matched drafting | Yes \(deep\) | Yes | Yes | Yes | Yes | Yes |
| Workflow/routine builder | Yes \(trigger-based\) | Yes \(visual\) | No | No | Tasklets | No |
| Meeting transcription | Yes \(iOS capture\) | No | No | No | No | No |
| Audio briefings | Yes | No | No | No | No | No |
| Calendar management | Deep \(CRUD + RSVP\) | Basic | No | No | Yes | No |
| Per-tool permissions | Yes | Limited | No | No | No | No |
| Persistent memory | Profile + memories | Some | Limited | Limited | Some | No |
| Integration count | \~12 | 200+ | \~5 | Gmail only | \~6 | \~3 |
| Multi-platform | 6 channels | Web, email | Web, iOS, desktop | Web, iOS | Web | Web |
| Pricing | Free \(open beta\) | $49.99/mo | $30-33/user/mo | $8.50-24/seat/mo | TBD | $22.50/mo |

#### Town Competitive Advantages \(vs field\)

1. **Graduated autonomy with per-tool granularity** — 3-mode system more nuanced than any competitor. Lindy is all-or-nothing; Consul is approval-only; Superhuman/Shortwave require human initiation.
2. **Meeting capture + audio briefings** — No major competitor combines voice recording, transcription, structured notes, and generated audio briefings.
3. **Multi-channel access** — Web + email + Slack + iOS + WhatsApp + desktop is broadest access surface in category.
4. **Free during beta** — Every named competitor charges $8.50-$50/month.
5. **Persistent memory architecture** — User profile + routine-specific memories deeper than competitors' tone matching.
6. **Founding team pedigree** — Ex-CTO Plaid + ex-Google AI Director + $18M from First Round Capital.

#### Town Competitive Disadvantages \(vs field\)

1. **Integration breadth** — \~12 named vs Lindy's 200+. Significant gap for cross-app workflow users.
2. **Market visibility** — Absent from every major 2026 AI email assistant comparison guide reviewed \(Zapier, Superhuman blog, Resident, Smartpostly, aitoolbriefing, etc.\).
3. **Review footprint** — No G2/Capterra/TrustRadius listings, no Product Hunt launch, minimal HN/Reddit discussion. Social proof near zero.
4. **Enterprise features** — No visible SSO, admin controls, or compliance features in public materials \(team routines exist but not marketed\).
5. **Post-beta pricing unknown** — Free beta is an advantage now but creates uncertainty for long-term planning.
6. **Pivot residue** — Tax-to-assistant pivot may create brand confusion \(efficient.app 2026 review still describes Town as tax advisory\).

#### Town Relevance to DEM Operations

Town is the **primary operator interface** for Darren's investment workflow. Current operational role:

- Receives Hermes email output \(Herald Digest, Bellringer, PDUFA alerts, 13F scans\) via [djschulz@gmail.com](mailto:djschulz@gmail.com)
- Runs 20+ routines \(morning briefings, catalyst tracking, 13F monitoring, intraday movers, drift baselines, SEC EDGAR scans\)
- Maintains persistent memory \(20+ global preferences, routine-specific memories\)
- Stores structured knowledge in Content Library \(biotech-screener, investment-frameworks, ai-projects collections\)
- Provides chat-based operator interface for ad-hoc research, drafting, calendar management
- Hosts 39 skills \(as of May 2026\) that encode DEM methodology

**Assessment \(Institutional Rigor Test\):**

- Claim classification: **PROMISING** — strong team, funded, differentiated features, but limited independent validation and near-zero market visibility
- Watch for: post-beta pricing, integration expansion, appearance in major comparison guides, enterprise feature development
- Risk: single-platform dependency for DEM operations \(Town + Hermes is the full stack; if Town pivots again or changes pricing, operational disruption is material\)

---

## Cost-Performance Benchmarking

### API Cost Context (updated 2026-05-23)

#### Frontier Models — Standard Pricing (per 1M tokens)

| Provider | Model | Input | Output | Context | Cache Discount |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek** | V4 Pro (75% promo thru May 31) | $0.435 | $0.87 | 1M | 99% (hit = $0.004) |
| **DeepSeek** | V4 Pro (post-promo, Jun 1) | $1.74 | $3.48 | 1M | ~99% |
| **DeepSeek** | V4 Flash | $0.14 | $0.28 | 1M | 99% (hit = $0.003) |
| **OpenAI** | GPT-4.1 | $2.00 | $8.00 | 1.05M | 75% ($0.50 cached) |
| **OpenAI** | GPT-4o | $2.50 | $10.00 | 128K | — |
| **OpenAI** | GPT-4.1 Mini | $0.40 | $1.60 | 1.05M | — |
| **OpenAI** | GPT-4.1 Nano | $0.10 | $0.40 | 1.05M | — |
| **Anthropic** | Claude Opus 4.7 | $5.00 | $25.00 | 1M | 90% cached; 50% batch |
| **Anthropic** | Claude Sonnet 4.6 | $3.00 | $15.00 | 1M | 90% cached; 50% batch |
| **Anthropic** | Claude Haiku 4.5 | $1.00 | $5.00 | 200K | 90% cached; 50% batch |
| **Google** | Gemini 2.5 Pro (<=200K) | $1.25 | $10.00 | 2M | 90% ($0.125 cached) |
| **Google** | Gemini 2.5 Pro (>200K) | $2.50 | $15.00 | 2M | 90% |
| **Google** | Gemini 2.5 Flash | $0.30 | $2.50 | 1M | 90% ($0.03 cached) |
| **Together AI** | Llama 3.3 70B | ~$0.30-0.60 | ~$0.60-0.80 | 128K | — |

#### DeepSeek V4 Pro Cost Advantage (at promo pricing thru May 31)

| vs Model | Input Savings | Output Savings |
| --- | --- | --- |
| GPT-4.1 | 78% cheaper | 89% cheaper |
| Claude Sonnet 4.6 | 86% cheaper | 94% cheaper |
| Claude Opus 4.7 | 91% cheaper | 97% cheaper |
| Gemini 2.5 Pro | 65% cheaper | 91% cheaper |

#### DeepSeek V4 Pro Cost Advantage (post-promo, Jun 1)

| vs Model | Input Savings | Output Savings |
| --- | --- | --- |
| GPT-4.1 | 13% cheaper | 57% cheaper |
| Claude Sonnet 4.6 | 42% cheaper | 77% cheaper |
| Claude Opus 4.7 | 65% cheaper | 86% cheaper |
| Gemini 2.5 Pro | 39% MORE expensive | 65% cheaper |

#### Best Value By Use Case

| Use Case | Best Value |
| --- | --- |
| Output-heavy (generation, code, long-form) | DeepSeek V4 Pro |
| Input-heavy with caching (RAG, repeated prompts) | DeepSeek V4 Flash or Gemini 2.5 Flash |
| Maximum reasoning quality | Claude Opus 4.7 or GPT-4.1 |
| Budget batch processing | GPT-4.1 Nano ($0.10/$0.40) |
| Long-context (>1M tokens) | Gemini 2.5 Pro (2M context) |
| Enterprise compliance / US-based | GPT-4.1 or Claude Sonnet 4.6 |
| Hermes agent fleet (daily driver) | Claude Sonnet 4.6 (tool-call reliability) |
| Hermes fallback / cost-conscious runs | DeepSeek V4 Flash |

#### DeepSeek V4 Key Specs

- **Models:** `deepseek-v4-flash` (budget), `deepseek-v4-pro` (reasoning)
- **Context:** 1M tokens; Max output: 384K tokens (largest available)
- **Thinking mode:** Both models support non-thinking and thinking (default) modes
- **API compatibility:** OpenAI format (`https://api.deepseek.com`) AND Anthropic format (`https://api.deepseek.com/anthropic`) — true drop-in replacement
- **Concurrency:** V4 Flash 2,500 / V4 Pro 500
- **Legacy model names:** `deepseek-chat` and `deepseek-reasoner` map to non-thinking/thinking modes of V4 Flash (will be deprecated)
- **Promo:** V4 Pro at 75% off through 2026-05-31 15:59 UTC; post-promo = 1/4 original price permanently
- **Data sovereignty note:** China-based infrastructure — do NOT route portfolio-sensitive or PII-containing prompts without operator review

#### Key Pricing Trends (2026)

1. **Output tokens are the real cost driver** — DeepSeek's output advantage (57-97% cheaper) matters more than input for generation-heavy agent workloads
2. **Cache pricing is converging toward free** — DeepSeek 99% discount, Anthropic 90%, Google 90%, OpenAI 75%
3. **Context windows standardized at 1M** — no longer a differentiator (except Gemini 2M)
4. **Price war accelerating** — DeepSeek forcing all providers to cut; expect further reductions Q3 2026

### Local LLM Decision (D-2026-007)

DEM Tier 6 decision: local LLMs deferred. Breakeven vs frontier APIs is 3-6 months on DGX Spark ($3K). Against cheap open-weight cloud APIs (DeepSeek V4 Flash at $0.14/M input), local never breaks even within hardware life. Qwen 2.5 Coder 32B scores 9.3/10 on tool calling but requires 24GB+ VRAM (current: 16GB).

---

## Key Sources \(Standing Watch List\)

### Academic / Research

| Source | Focus | Frequency |
| --- | --- | --- |
| arXiv \(cs.AI, q-fin\) | Frontier AI + quant finance research | Weekly scan |
| SSRN | Finance-specific academic papers | Weekly scan |
| Journal of Finance | Peer-reviewed finance research | Quarterly |
| Springer AI/Finance | Applied AI in finance | Monthly |

### Industry / Practitioner

| Source | Focus | Frequency |
| --- | --- | --- |
| HBR | AI strategy, management implications | Monthly |
| MIT Technology Review | Technology breakthroughs | Monthly \(10 Breakthrough Technologies\) |
| The Economist | AI policy, economic impact | Weekly |
| AQR Capital | Systematic/quant strategy insights | Quarterly |
| Anthropic Blog | Claude capabilities, safety research | On release |

### Biotech + AI Specific

| Source | Focus | Frequency |
| --- | --- | --- |
| BioPharm IQ \(@BioPharmIQ\) | Biotech news, clinical data | Daily \(Twitter\) |
| BiotechEdge | 13F tracking, fund convergence | Weekly |
| CatalystAlert | PDUFA/catalyst tracking | Continuous |
| BioCatalysts.AI | Catalyst volatility prediction | Continuous |
| PDUFA.BIO + ODIN | FDA decision prediction | Per-event |

---

## Routine Integration

### Friday AI Digest

- **Schedule:** Friday 8 AM ET
- **Scope:** Self-forwarded "Friday AI" links from past week + web research
- **Themes:** Claude Code/AI Agents, Hermes, Quant Finance + AI, Biotech + AI
- **Output:** Email to [dschulz@wakerobin.co](mailto:dschulz@wakerobin.co) \+ [djschulz@gmail.com](mailto:djschulz@gmail.com)

### Weekly AI Research Digest

- **Schedule:** Weekly
- **Scope:** 11 AI + investment management research topics
- **Output:** Compiled document saved to ai-projects Content Library collection + email summary to [dschulz@wakerobin.co](mailto:dschulz@wakerobin.co)

---

## Key Constraints

1. AI landscape intelligence is for INFORMING decisions, not for automated action
2. All AI performance claims must pass the Institutional Rigor Test before being cited
3. Hermes competitive analysis must be honest about disadvantages, not just advantages
4. Cost-performance benchmarking should be updated quarterly as API pricing changes
5. The AI bubble/skeptic perspective is tracked alongside positive developments -- both lenses matter
