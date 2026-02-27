---
name: ai-agent-frameworks
description: >
  Comprehensive reference for choosing a Python AI agent framework.
  Covers top 5 frameworks by GitHub stars: LangChain, MetaGPT, AutoGen,
  LlamaIndex, and CrewAI. Includes star counts, main features, design
  philosophy, honest community criticisms, use-case fit matrix, and a
  decision flowchart. Research from GitHub, Reddit, HN, and dev blogs
  (Feb 2026). Use this before starting any LLM agent project.
triggers:
  - user wants to build an AI agent, multi-agent system, or LLM pipeline
  - user asks which AI agent framework to use
  - user mentions LangChain, AutoGen, CrewAI, LlamaIndex, or MetaGPT
  - user asks about RAG, tool-use agents, or orchestrating multiple LLMs
  - comparing agent frameworks, pros and cons, "which framework should I use"
---

# AI Agent Framework Selection Guide

Researched Feb 2026. Data from GitHub star counts, official docs, Reddit
(r/LangChain, r/LocalLLaMA, r/AI_Agents, r/MachineLearning, r/AutoGenAI),
Hacker News, and developer blogs.

---

## Quick Decision Matrix

| Use Case                              | Recommended        | Avoid                |
|---------------------------------------|--------------------|----------------------|
| Production RAG over private docs      | LlamaIndex         | MetaGPT              |
| Multi-agent research / exploration    | AutoGen            | CrewAI               |
| Rapid PoC / demo with role agents     | CrewAI             | MetaGPT              |
| Widest integration ecosystem needed   | LangChain          | MetaGPT              |
| Software generation / code agents     | AutoGen or MetaGPT | LlamaIndex           |
| Complex stateful agent graphs         | LangGraph (LC)     | CrewAI               |
| Cost-sensitive production workload    | LlamaIndex / raw SDK | MetaGPT / AutoGen  |
| Easy onboarding, small team           | CrewAI             | LangChain            |
| Local / open-source model usage       | LlamaIndex / Ollama | MetaGPT            |

---

## Framework Profiles

---

### 1. LangChain
- Repo:   https://github.com/langchain-ai/langchain
- Stars:  ~97,900 (largest in the space by far)
- Forks:  ~15,900
- PyPI:   langchain, langchain-core, langchain-community

#### Design Philosophy
General-purpose LLM application framework. Build anything from simple
chat to complex multi-step agentic workflows. Emphasis on composability
via the pipe operator (LCEL).

#### Key Features
1. LCEL (LangChain Expression Language): declarative chain composition
   using | operators -- strings together prompts, LLMs, tools, retrievers
   in a functional, lazy-evaluated pipeline.
2. Massive integration surface: 100+ LLM providers, 50+ vector stores,
   document loaders for PDF, HTML, Notion, Slack, SQL, and more.
3. LangGraph: first-class support for cyclical, stateful agent graphs
   with branching logic, human-in-the-loop, and checkpointing. This is
   now the recommended path for complex agents.
4. Full RAG toolkit: text splitters, embedding wrappers, retrievers,
   re-rankers, and multiple retrieval strategies (MMR, similarity, hybrid).
5. LangSmith: observability, tracing, evaluation, and monitoring for
   every LLM call and chain step. The de facto debugging layer.

#### Biggest Weaknesses (community consensus)
1. Over-abstraction / leaky abstractions: LangChain wraps simple API
   calls in many layers that obscure what prompts are actually sent.
   Debugging stack traces are cryptic. Max Woolf famously showed that
   3-line raw OpenAI calls become 20+ lines in LangChain with less clarity.
   HN "LangChain is Bad" thread (38548872) got 302 comments of agreement.
2. Extreme version churn: breaking changes every few weeks. LCEL
   replaced the old Chain API mid-stream. Packages split into langchain-
   core, langchain-community, langchain-openai causing widespread import
   hell. Production deployments routinely break on upgrades.
3. Package fragmentation: functionality split across langchain, langchain-
   core, langchain-community, langgraph, langsmith -- confusing deps.
4. Overengineering for simple use cases: most teams building simple
   chat or single-call apps end up ripping out LangChain and rewriting
   with raw SDKs. Quote: "I spent a weekend debugging LangChain; I could
   have written the whole thing from scratch in that time."
5. Docs lag codebase: examples use deprecated APIs frequently. The
   sheer volume of concepts (40+ chain types, multiple agent frameworks)
   makes it hard to find the right approach.
6. LangSmith funnel: the recommended debugging/eval path leads toward
   a paid product, creating subtle vendor lock-in.

#### When to Use LangChain
- You need the widest possible integration ecosystem (obscure vector DB,
  unusual doc loader, niche LLM provider)
- You are building a complex stateful agent graph (use LangGraph)
- Your team already knows LangChain well and migration cost is high
- You need battle-tested RAG components (splitters, retrievers)

#### When NOT to Use LangChain
- Simple single-LLM-call applications (use the raw SDK)
- Small teams who need stable, debuggable code
- Production systems where you can't afford frequent migration work
- Anyone new to LLMs (the learning curve is high and can mislead)

---

### 2. MetaGPT
- Repo:   https://github.com/geekan/MetaGPT
- Stars:  ~47,400
- Forks:  ~5,600
- PyPI:   metagpt

#### Design Philosophy
Simulate a software engineering company using multiple LLM-powered agents
in formal roles. Agents produce structured artifacts (PRDs, design docs,
code, test plans) following Standard Operating Procedures (SOPs).

#### Key Features
1. Software company simulation: agents act as Product Manager, Architect,
   Lead Engineer, QA Engineer, each with distinct responsibilities.
2. Structured artifact output: produces formal deliverables (PRDs, system
   design, UML diagrams, API specs, source code, test suites) not just
   raw text.
3. SOP-driven workflows: agents follow predefined organizational processes,
   encoding human software development workflows into agent behavior.
4. Code review / debug loop: dedicated QA and Engineer agents iterate to
   catch and fix code-level issues automatically.
5. DataInterpreter: specialized mode for data science -- generates
   analysis code, executes it, interprets results, produces charts.
6. Incremental development: can pick up from an existing codebase.

#### Biggest Weaknesses (community consensus)
1. Prohibitively expensive: a simple coding task costs $5--$20+ in API
   tokens due to sequential multi-agent round-trips. Complex projects
   can cost $50+. Shocking API bills are a common Reddit complaint.
2. Very slow: full pipeline runs take 15--30+ minutes. Sequential SOP
   structure, API rate limits, and token volume create painful latency.
3. Narrow use case: deeply optimized for software generation. Adapting
   to non-coding domains (data analysis, customer service, research)
   requires invasive rework and the framework resists generalization.
4. Code quality requires heavy human review: generated code often works
   for toy inputs but fails on edge cases, missing libraries, or complex
   logic. It is a starting point, not finished product.
5. Hallucination compounding: errors in early agent outputs (e.g., a
   wrong assumption in the PRD) cascade through all downstream agents,
   producing confidently wrong code with no self-correction.
6. English docs/community gaps: primary dev team and active community
   are Chinese-speaking. English docs can be incomplete or machine-
   translated, and Western community support is thin.
7. Academic codebase: research-first priorities mean poor logging, error
   handling, and production deployment tooling.

#### When to Use MetaGPT
- Research into multi-agent software generation
- Generating boilerplate/scaffold code for well-defined, narrow specs
- DataInterpreter for automated data analysis pipelines
- Academic exploration of agentic systems

#### When NOT to Use MetaGPT
- Any cost-sensitive application
- Production software development (output requires too much human work)
- Non-software use cases
- Real-time or interactive applications (too slow)

---

### 3. AutoGen (Microsoft)
- Repo:   https://github.com/microsoft/autogen
- Stars:  ~40,800
- Forks:  ~5,900
- PyPI:   autogen-agentchat (v0.4+), pyautogen (v0.2 community fork)

#### Design Philosophy
Agent framework centered on structured conversations between multiple
agents. Agents solve tasks by exchanging messages, with the key insight
that a generate-execute-feedback loop (write code, run it, fix errors)
can automate complex workflows.

#### Key Features
1. Conversational multi-agent model: agents exchange natural language
   messages; the conversation IS the computation.
2. Code generation + sandboxed execution loop: AssistantAgent writes
   Python; UserProxyAgent runs it in Docker/local sandbox; errors feed
   back automatically for self-correction.
3. Flexible topologies: two-agent chats, GroupChat (N agents + moderator),
   hierarchical nested chats for complex orchestration, and Swarm (v0.4).
4. Event-driven actor model (v0.4+): async, distributed message passing
   with typed events. Agents are isolated actors -- better for scale.
5. Model agnostic: OpenAI, Azure OpenAI, Anthropic, Gemini, local models
   via Ollama/LiteLLM.
6. AutoGen Studio: drag-and-drop visual interface for prototyping agent
   workflows without code.

#### Biggest Weaknesses (community consensus)
1. Catastrophic v0.2 -> v0.4 breaking changes: v0.4 is an almost total
   rewrite. All v0.2 code, tutorials, and third-party plugins are
   incompatible. Community is fractured: many stay on pyautogen (v0.2
   fork) rather than migrate. Microsoft did not provide a smooth
   migration path.
2. Infinite loops and cost blowouts: multi-agent conversations frequently
   loop endlessly or make far more LLM calls than expected. Termination
   conditions are hard to get right. Same-input runs can produce
   dramatically different token counts.
3. Non-deterministic GroupChat: agents in group conversations don't
   reliably follow turn-taking rules or stay on task. The speaker
   selection mechanism (LLM-based by default) adds unpredictability.
4. Microsoft abandonment risk: community cites MSFT's history of
   abandoning developer tools (Cortana, etc.) and the v0.4 rewrite
   signals instability. The v0.4 split fractured ecosystem momentum.
5. Requires strong (expensive) models: reliable code generation and
   agent coordination degrades significantly below GPT-4 class models.
   Cheaper alternatives (GPT-3.5, local models) produce unreliable
   agent behavior.
6. Security concerns around code execution: running LLM-generated code,
   even in sandboxes, is a real attack surface in production.

#### When to Use AutoGen
- Research and exploration with agentic coding tasks
- Code generation workflows where self-correction is valuable
- Prototyping multi-agent conversation workflows
- Academic benchmarking, internal tooling, R&D
- When you need the generate-execute-fix loop (software debugging, data
  science automation)

#### When NOT to Use AutoGen
- Cost-sensitive production (hard to bound token spend)
- Applications that need deterministic, predictable outputs
- Long-running agents where conversation loops are risky
- Teams that need a stable, long-term API

---

### 4. LlamaIndex
- Repo:   https://github.com/run-llama/llama_index
- Stars:  ~40,000
- Forks:  ~5,700
- PyPI:   llama-index-core (+ integration packages)

#### Design Philosophy
Data-centric framework purpose-built for connecting LLMs to custom data
sources. The core mission: make any data source queryable by an LLM.
Best-in-class for RAG; agents and workflows are a secondary layer built
on top.

#### Key Features
1. Data-centric / RAG-first: the best framework for building
   production-quality Retrieval-Augmented Generation over private data.
2. Advanced retrieval strategies: BM25, dense vector, hybrid retrieval,
   re-ranking (Cohere, BGE), HyDE (hypothetical document embeddings),
   small-to-big retrieval, recursive retrieval, and more.
3. LlamaHub: 100+ data connectors (PDF, Word, Notion, Slack, Confluence,
   SQL, REST APIs, Google Drive) maintained by the community.
4. Agentic workflows: ReAct agents, OpenAI function-calling agents, and
   the Workflow API (event-driven, async, stateful) for complex
   multi-step data pipelines.
5. Built-in evaluation: measures faithfulness, answer relevance, context
   relevance, and groundedness -- the best evaluation story of any
   framework.
6. LlamaCloud: managed data pipeline and retrieval service for production
   deployments at scale (paid).

#### Biggest Weaknesses (community consensus)
1. Steep learning curve for RAG customization: basic RAG is easy but
   tuning for real-world quality requires understanding NodeParser,
   Retriever, QueryEngine, ResponseSynthesizer, and how they interact --
   poorly documented.
2. v0.10 API reorganization: massive restructure split everything into
   llama-index-core + dozens of integration packages, breaking existing
   code. Migration guides were incomplete. Described as traumatic by
   frequent users.
3. Black-box retrieval quality: when RAG returns poor results, diagnosing
   whether the issue is chunking, embedding, storage, or query formulation
   is non-trivial. Pipeline layers obscure root causes.
4. Performance and scale issues: default VectorStoreIndex loads all
   embeddings in memory. Indexing large corpora (100k+ docs) is slow and
   memory-hungry. Requires external vector DB + careful tuning for scale.
5. LlamaCloud funnel concerns: community perception that best-in-class
   production features are being reserved for the paid LlamaCloud tier,
   with OSS increasingly treated as top-of-funnel.
6. Agent layer feels secondary: LlamaIndex added agents after RAG.
   The agent implementation is less mature and more buggy than AutoGen
   or LangGraph for complex agentic workflows.
7. OpenAI-centric defaults: despite nominal agnosticism, defaults and
   integrations are tuned for OpenAI. Local/alternative models require
   extra manual configuration.

#### When to Use LlamaIndex
- RAG is the core use case (Q&A over private docs, knowledge bases)
- You need advanced retrieval strategies beyond basic vector similarity
- Data is in diverse formats (PDFs with tables, databases, APIs)
- You want built-in RAG evaluation metrics
- Production data pipelines connecting LLMs to structured/unstructured data

#### When NOT to Use LlamaIndex
- You don't have a data retrieval use case (pure chat, code gen, etc.)
- You need a mature, stable agent framework (use AutoGen or LangGraph)
- Your data corpus is small and simple (raw LLM call may suffice)
- You have a small team and can't invest in RAG tuning

---

### 5. CrewAI
- Repo:   https://github.com/crewAIInc/crewAI
- Stars:  ~28,100
- Forks:  ~3,800
- PyPI:   crewai

#### Design Philosophy
Human-readable, role-based multi-agent orchestration. Agents are defined
as role-playing entities (Researcher, Writer, Analyst) with goals and
backstories. Designed for intuitive, accessible multi-agent apps.

#### Key Features
1. Role-based agent design: agents have role, goal, backstory, and
   optional tools -- the most intuitive mental model for non-experts.
2. Tasks as first-class objects: tasks are separate from agents, with
   explicit expected output, tool access, and optional delegation to
   other agents.
3. Multiple process modes: Sequential (default pipeline), Hierarchical
   (manager agent delegates to workers), Consensual (agents vote).
4. Memory system: short-term (within run), long-term (SQLite across
   runs), entity memory (extracts named entities), contextual memory.
5. Tool ecosystem: built-in tools (web search via SerperDev, file ops,
   code interpreter) plus full LangChain tool compatibility.
6. Flows API: newer event-driven pipeline API for structured, conditional
   workflows (introduced to compete with LangGraph).

#### Biggest Weaknesses (community consensus)
1. Agent reliability / hallucination: agents frequently deviate from
   role definitions or claim to complete tasks without doing so.
   Role-playing is just a prompt string -- LLMs can and do ignore it.
   Results are highly non-deterministic. Critical for production use.
2. High cost and slow speed: hierarchical delegation + full context
   passing between agents causes token bloat. Crews with 4+ agents
   can cost several dollars per run on non-trivial tasks.
3. Memory system fragility: the multi-tier memory is unreliable in
   practice. Context is frequently lost between tasks; long-term memory
   retrieval quality is inconsistent. SQLite backend has race conditions
   in parallel runs.
4. Rapid breaking changes: minor version bumps introduce breaking
   changes. Production deployments break. Documentation consistently
   lags the codebase.
5. Enterprise gatekeeping: advanced monitoring, managed execution, and
   collaborative features being pushed toward paid crewai.com platform.
   OSS open-source commitment is uncertain long-term.
6. Limited control flow: complex workflows with conditional branching
   or loops based on agent output are awkward. The framework was designed
   for linear pipelines; arbitrary DAGs require the newer Flows API which
   is less mature.
7. Pydantic schema fragility: when LLMs produce non-conformant structured
   output, error handling is poor and retries aren't always automatic.
   Smaller/local models fail here frequently.

#### When to Use CrewAI
- Rapid prototyping and demos with multi-agent workflows
- Teams new to agent frameworks (most intuitive to get started)
- Use cases where the role metaphor maps naturally to the domain
  (e.g., research crew: researcher + writer + editor)
- PoCs to show stakeholders what's possible

#### When NOT to Use CrewAI
- Production systems requiring deterministic, reliable outputs
- Cost-sensitive applications
- Complex workflows with branching, conditions, or hard exits
- When you need fine-grained control over agent behavior
- Using local/small models (reliability requires GPT-4 class)

---

## Cross-Framework Themes (Affects All)

1. LLM non-determinism: none of these frameworks solve the fundamental
   issue that LLM outputs vary run-to-run. Agentic pipelines can work
   01 time and fail the next with identical inputs.

2. Cost opacity: all frameworks make it difficult to predict or cap API
   costs before running. Token usage is hard to estimate for agentic workflows.
   Always add token budgets, timeouts, and cost monitoring.

3. Commercial lock-in risk:
   - LangChain -> LangSmith (tracing/eval, paid)
   - AutoGen   -> Azure AI services
   - CrewAI    -> crewai.com (managed platform, paid)
   - LlamaIndex -> LlamaCloud (managed data pipeline, paid)

4. Debugging is universally hard: agentic "decide what to do next" loops
   are inherently hard to debug with traditional tooling. Plan for this.

5. Demo-to-production gap: all frameworks produce impressive demos that
   degrade on messy real-world data, requirements, and edge cases.

6. Rapid API churn: the AI agent space is too immature for stable APIs.
   Budget time for migration when you upgrade any of these frameworks.

---

## Star Count Reference (Feb 2026)

| Framework  | Stars   | Forks  | Age    | Company            |
|------------|---------|--------|--------|--------------------|
| LangChain  | ~97,900 | 15,900 | 2022   | LangChain Inc      |
| MetaGPT    | ~47,400 |  5,600 | 2023   | DeepWisdom (CN)    |
| AutoGen    | ~40,800 |  5,900 | 2023   | Microsoft Research |
| LlamaIndex | ~40,000 |  5,700 | 2022   | LlamaIndex Inc     |
| CrewAI     | ~28,100 |  3,800 | 2023   | CrewAI Inc         |

Note: star counts are a marketing signal, not a quality signal.
LangChain's lead partly reflects being first-mover. MetaGPT's high count
reflects viral demos rather than production adoption.

---

## Recommended Alternatives (When Frameworks Are Overkill)

- Raw SDK (openai, anthropic, google-genai): For simple single-LLM apps,
  just use the SDK directly. Simpler, faster, more debuggable.
- smolagents (HuggingFace): Lightweight, minimal agent framework. Good
  for simple tool-use agents without framework overhead.
- DSPy: If you need structured, programmatic prompt optimization rather
  than manual prompt engineering. See dspy skill.
- Instructor: If you just need reliable structured output from LLMs.
  See instructor skill.
- Custom from scratch: For production, many teams find writing a thin
  custom orchestration layer over raw SDKs more maintainable than any
  of the above.

---

## Decision Flowchart

1. Is RAG (retrieval over documents) your primary use case?
   YES -> LlamaIndex
   NO  -> continue

2. Do you need code generation + execution in a loop?
   YES -> AutoGen
   NO  -> continue

3. Do you need structured software artifact generation?
   YES -> MetaGPT (accept high cost/time) or AutoGen
   NO  -> continue

4. Is this a PoC/demo and speed of development matters most?
   YES -> CrewAI
   NO  -> continue

5. Do you need maximum integration ecosystem coverage?
   YES -> LangChain (use LangGraph for complex agents)
   NO  -> continue

6. Is this a simple single-LLM-call application?
   YES -> raw SDK (openai, anthropic), skip all frameworks
   NO  -> re-evaluate LangGraph or LlamaIndex Workflow API
