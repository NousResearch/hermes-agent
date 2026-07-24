# Related Work — Structured Bibliography

> Extraction of all 29 papers/projects from Phase 0 background survey.
> Last updated: 2026-05-10
> Source: phase0-background.md

---

## Category 1: Control Theory × AI Agent Architecture

### 1. AICL: A Control-Loop Architecture for Stable Long-Horizon LLM Agents

- **Citation**: Zenodo (2025), roackb2/cyberloop open-source implementation
- **Summary**: Proposes the "Artificial Intelligence Control Loop (AICL)," formalizing LLM Agent reasoning as a closed-loop process: structured planning → probe-driven monitoring → event orchestration → quantitative stability budget. Aims to suppress drift and inconsistency in long-horizon reasoning. Open-source CyberLoop provides kinematicsMiddleware (EKF+PID), manifoldMiddleware (local PCA+curvature), grassmannianMiddleware (subspace tracking) — differential-geometry-based middlewares.
- **Relevance**: ★★★★
- **Gap**: Does not address Qian Xuesen's hierarchical control thinking. Focuses on single-agent trajectory stability without extension to multi-agent coordination. Does not discuss Open Complex Giant Systems.

### 2. A Control-Theoretic Foundation for Agentic Systems

- **Citation**: arXiv:2603.10779 (2025)
- **Summary**: Proposes a five-level agency hierarchy defining "agency" as runtime decision authority over control architecture elements. From fixed control law → parameter adaptation → policy switching → workflow reconfiguration → goal synthesis. Each level introduces time-varying, switching, delay, and hybrid dynamics. Currently the most systematic formalization of control theory applied to AI Agents.
- **Relevance**: ★★★★★
- **Gap**: Does not address Qian Xuesen's meta-synthetic engineering methodology. Does not discuss human-machine integrated control hierarchy. No connection to concrete Agent systems (Hermes/Claude/GPT).

### 3. Stable Agentic Control: Tool-Mediated LLM Architecture

- **Citation**: arXiv:2605.03034 (2025)
- **Summary**: Tool-mediated architecture — LLM Agent uses deterministic tools; closed-loop stability is a property of the loop, not the Agent. Uses Lean 4 machine verification for Lyapunov stability certificates (controllability, observability, ISS robustness). Key insight: stability is an architectural property, not a model property. First machine-verified stability certificate for tool-mediated LLM controller.
- **Relevance**: ★★★★
- **Gap**: Oriented toward adversarial cybersecurity scenarios, not generalized to general-purpose Agents. Does not address hierarchical control.

### 4. Cybernetic Agents: A Research Proposal

- **Citation**: aixiv.science preprint (research proposal)
- **Summary**: Proposes "cybernetic agent" framework: (i) models LLM Agent as stochastic dynamical system on semantic embedding manifold; (ii) defines semantic stability and safety; (iii) designs explicit observer, planner, regulator, safety module. Core strategy: π = B ∘ R ∘ P ∘ O (safety filter ∘ semantic PID ∘ MPC planning ∘ Kalman observation). Most proximate to our objectives — but remains a proposal, not implemented.
- **Relevance**: ★★★★★
- **Gap**: Does not mention Qian Xuesen or Chinese cybernetics tradition. Does not address meta-synthetic engineering methodology. No discussion of multi-agent collaborative cybernetic framework.

### 5. Your AI Agent Is a Control System

- **Citation**: bolu.dev blog post (2026)
- **Summary**: Agent is an iterative policy embedded in tool-use loops, using new observations to update next actions in a partially observed environment — this is a dynamical system. Best analogy: Model Predictive Control (MPC): estimate → act → observe → re-plan. Inner loop does local work, outer loop manages timeouts, budgets, goals — a multi-rate control system. Emphasizes "sensor engineering" — an Agent can only correct what it can measure.
- **Relevance**: ★★★
- **Gap**: Metaphorical/analogy level, not formalized. Does not address Qian Xuesen's hierarchical systems theory.

### 6. The Control Theory Behind Harness Engineering

- **Citation**: adityashrishpuranik.com blog post (2026)
- **Summary**: Directly maps harness engineering to classical control theory four elements: LLM = plant, harness = controller, observability = sensors, task specification = reference signal. Closed-loop control means output feeds back to input for self-correction — exactly what harness engineering builds. Notes that vocabulary matters — shifting from "prompting" to "building control systems" immediately changes optimization direction.
- **Relevance**: ★★★
- **Gap**: Same mapping/metaphor level, not formalized. Does not address Qian Xuesen.

---

## Category 2: Feedback Mechanisms × Agentic AI

### 7. Feedback Loops for AI Agents

- **Citation**: Medium blog series by mjgmario (2026)
- **Summary**: Feedback loops transform runtime experience into persistent behavioral improvement — without retraining model weights. Connects execution → evaluation → adaptation. Production-grade feedback loops are policy update systems that can apply changes across multiple change surfaces: context structure, tool routing, memory strategies, stop/retry rules.
- **Relevance**: ★★
- **Gap**: Not formalized in control theory language. No discussion of stability, controllability, or other core control concepts.

### 8. Cybernetic Entropy Control

- **Citation**: LessWrong post (2025)
- **Summary**: Builds feedback controller that regulates LLM sampling parameters in real-time based on token-level entropy. Uses PID or fourth-order controller optimizing entropy error to target value. Achieves 3% accuracy improvement on MATH benchmark. Directly applies PID control to LLM inference layer — very specific and actionable.
- **Relevance**: ★★★
- **Gap**: Only addresses sampling parameters, not extended to Agent behavior layer. No connection to Qian Xuesen.

### 9. Agentic Control in Variational Language Models

- **Citation**: arXiv:2604.12513 (2025)
- **Summary**: Studies the minimal measurable form of agentic control in variational language models: internal uncertainty serves not only as diagnostic metric but also as operational signal — regulating training, supporting checkpoint retention, guiding inference-time interventions. Transforms uncertainty from passive measurement to active control interface.
- **Relevance**: ★★★
- **Gap**: Focused on model internals, not extended to Agent system layer.

---

## Category 3: Qian Xuesen × AI / Complex Systems

### 10. Engineering Cybernetics (工程控制论)

- **Citation**: Qian Xuesen, *Engineering Cybernetics*, McGraw-Hill, 1954. 18 chapters, 289 pages. Revised edition with Song Jian, Science Press, 1980 (21 chapters).
- **Summary**: Engineering cybernetics is "an engineering science that aims to organize the design principles used in engineering practice into a discipline, thereby demonstrating similarities between different fields of engineering practice and emphasizing the power of fundamental concepts." 18 chapters covering: non-interacting control of multivariable systems, perturbation-theory-based control design, Von Neumann's error control theory. Chapter 18 discusses building highly reliable systems from relatively unreliable elements.
- **Relevance**: ★★★★★
- **Gap**: 1954 work — naturally does not address AI Agents. But its methodology is fully transferable.

### 11. Open Complex Giant Systems and Its Methodology (开放的复杂巨系统及其方法论)

- **Citation**: Qian Xuesen, Yu Jingyuan, Dai Ruwei (1993). IEEE SMC Conference Invited Paper.
- **Summary**: Open Complex Giant Systems cannot be handled with reductionism. The only viable alternative is meta-synthetic engineering (from qualitative to quantitative). Five characteristics: organic combination of qualitative and quantitative; integration of scientific theory and experiential knowledge; multidisciplinary research; macro-micro unification; requires computer system support.
- **Relevance**: ★★★★★
- **Gap**: Published in 1993 when LLMs did not exist. No one has yet systematically applied this methodology to AI Agent architecture. This is our gap.

### 12. Hall for Workshop of Meta-Synthetic Engineering (综合集成研讨厅体系)

- **Citation**: Qian Xuesen, series of addresses in the 1990s.
- **Summary**: The practical form of meta-synthetic engineering. Organically combines expert groups, data and information, and computing systems into a highly intelligent human-machine interactive system. Possesses comprehensive advantage, overall advantage, and intelligence advantage.
- **Relevance**: ★★★★★
- **Gap**: No one has explicitly proposed using HWME to design multi-Agent systems. The mapping is direct: expert groups = expert Agents; data/information = Agent memory and knowledge systems; computing systems = Agent tools and compute environments; human-machine interaction = Agent-user interface.

### 13. Meta-Synthetic Wisdom Engineering (大成智慧工程)

- **Citation**: Qian Xuesen, later-period thought.
- **Summary**: Integrates thousands of years of human intellectual achievements, knowledge, wisdom, and various intelligence data to achieve "collect great achievement, attain wisdom" (集大成得智慧). Emphasizes human-machine integration with humans as primary.
- **Relevance**: ★★★★
- **Gap**: No one has connected Meta-Synthetic Wisdom Engineering to AI Agent skill learning systems. Hermes Agent's learning cycle is essentially a micro-implementation.

### 14. Qian Xuesen and Engineering Cybernetics

- **Citation**: Zheng Yingping (2001). *Strategic Study of Chinese Academy of Engineering*.
- **Summary**: Systematic review of Qian Xuesen's *Engineering Cybernetics* — its content, paradigm, and important role in the development of control theory from "classical" to "modern." Keywords: cybernetics/engineering cybernetics/engineering science/complex system control/meta-synthetic methodology for complex giant systems.
- **Relevance**: ★★★★
- **Gap**: Does not address AI/Agents. Provides the intellectual genealogy but no technical bridge to AI.

### 15. Between the Human and the Machinic: Qian Xuesen and AI

- **Citation**: Urbanomic publication, Bo An.
- **Summary**: Qian Xuesen's cybernetics and AI thought should be understood as "engineering of systems discourse and design" — systems for governing society and the state. Lingjing (灵境, virtual reality) is not merely a translation but part of a theory of mega socio-technical management systems.
- **Relevance**: ★★★
- **Gap**: Philosophical/intellectual history oriented. Does not address concrete technical architecture.

---

## Category 4: Hierarchical Control × Multi-Agent Systems

### 16. POLARIS: Multi-Agentic Reasoning for Self-Adaptive Systems

- **Citation**: arXiv (2024)
- **Summary**: Three-layer multi-Agent adaptive framework: low-latency Adapter layer (monitoring + safety execution) → Reasoning layer (tool-aware, explainable Agent) → Meta layer (experience logging + meta-learning to improve adaptation strategy). Introduces Self-Adaptation 3.0 paradigm.
- **Relevance**: ★★★
- **Gap**: Deep resonance with Qian Xuesen's hierarchical systems thinking (execution layer = applied technology, reasoning layer = technical science, meta layer = fundamental theory), but does not cite Qian Xuesen or the cybernetics tradition.

### 17. CTHA: Constrained Temporal Hierarchical Architecture

- **Citation**: arXiv:2601.10738 (2025)
- **Summary**: Temporal hierarchical architecture — introduces temporal layers (different cognitive layers), uses message contract constraints, permission manifold constraints, and arbitrator conflict resolution to restore coordination stability. 47% fault cascade reduction.
- **Relevance**: ★★★
- **Gap**: Directly addresses stability in multi-Agent hierarchical architectures. Does not connect to cybernetics or Qian Xuesen.

### 18. HiMAC: Hierarchical Macro-Micro Agentic Control

- **Citation**: arXiv (2025)
- **Summary**: Macro planning (blueprint generation) + micro execution (atomic actions), with iterative co-evolutionary training. Explicitly separates decisions at different time scales. Name contains "control" but uses RL rather than classical control theory.
- **Relevance**: ★★★
- **Gap**: Does not reference classical cybernetics. Does not connect to Qian Xuesen.

### 19. REDEREF: Probabilistic Control for Multi-Agent LLM Systems

- **Citation**: arXiv:2603.13256 (2025)
- **Summary**: Training-free probabilistic controller: Thompson sampling-guided Agent delegation + reflection-driven rerouting + evidence selection + memory priors. Reduces token usage by 28% in recursive delegation. Uses probabilistic control theory to manage multi-Agent coordination.
- **Relevance**: ★★★
- **Gap**: Resonates with Qian Xuesen's "reliable systems from unreliable elements" but does not cite him. No formal stability analysis.

---

## Category 5: Self-Organization × AI Agent

### 20. Agentic AI Needs a Systems Theory

- **Citation**: arXiv:2503.00237 (2025). Position paper.
- **Summary**: **Important position paper.** Current AI development over-focuses on individual model capabilities, ignoring broader emergent behavior. Systems need not have every component highly functional — tool use, state maintenance, and environment interaction capabilities can lead to collective agency. Agent contains internal act-sense-adapt loop fed by higher-level feedback loops (agent-human, agent-agent, agent-environment interfaces).
- **Relevance**: ★★★★
- **Gap**: Directly calls for systems theory applied to Agentic AI — fully consistent with Qian Xuesen's systems science. But cites Wiener/Ashby/Von Bertalanffy without mentioning Qian Xuesen.

### 21. Synergetics and LLMs: Emergence, Order, and Self-Organization

- **Citation**: gpt.gekko.de analysis article (2025)
- **Summary**: Uses Haken's synergetics concepts to explain LLM behavior: order parameters (key features emerge and "enslave" others), bifurcation thresholds (emergence of new capabilities), self-organization (complexity produces intelligence). LLMs are composite systems where many simple units produce a whole far greater than the sum of parts.
- **Relevance**: ★★★★
- **Gap**: Synergetics' "slaving principle" (slow variables dominate fast variables) directly corresponds to hierarchical control in Agent architecture (high-level policy dominates low-level execution). But synergetics principles have not been systematically applied to Agent architecture design.

### 22. Self-Organizing Agent Network (SOAN)

- **Citation**: arXiv:2508.13732 (2025)
- **Summary**: Self-organizing Agent network framework: incrementally builds formal Agent networks by identifying and encapsulating structural units as independent Agents, enhancing modularity and orchestration clarity.
- **Relevance**: ★★
- **Gap**: Self-organization applied to Agent workflow orchestration. Does not connect to cybernetics tradition.

---

## Category 6: Observability/Controllability × AI

### 23. Are LLMs Controllable?

- **Citation**: arXiv:2601.05637 (2025)
- **Summary**: Formalizes human-model interaction as a control process. Proposes algorithm for estimating controllable sets with PAC (probably approximately correct) guarantees. Empirical finding: model controllability is surprisingly fragile, highly dependent on model, task, and initial state.
- **Relevance**: ★★★★
- **Gap**: **Important finding** — current LLM controllability is not a given. Provides strong argument for necessity of control layer in Agent architecture. Focuses on dialogue control, not extended to Agent systems. No connection to Qian Xuesen.

### 24. Verifiability-First Agents (OPERA)

- **Citation**: OpenReview (2025)
- **Summary**: Verifiability-first architecture: runtime cryptographic proofs + lightweight audit Agent continuously verifying intent vs. behavior + challenge-response protocol. Shifts evaluation focus from "misalignment probability" to "speed and reliability of detection and repair."
- **Relevance**: ★★★
- **Gap**: Directly corresponds to "observability" in cybernetics. Not framed in control theory language.

### 25. Governance-Aware Agent Telemetry (GAAT)

- **Citation**: arXiv (2025)
- **Summary**: Closes the loop between telemetry collection and automated policy enforcement. Five-layer reference architecture: governance telemetry patterns → real-time policy violation detection → governance enforcement bus (tiered intervention) → trusted telemetry plane. Practical closed-loop governance system — direct engineering implementation of cybernetics in Agent systems.
- **Relevance**: ★★★
- **Gap**: Does not connect to Qian Xuesen's systems engineering thought.

---

## Category 7: Synergetics × AI

### 26. Haken's Synergetics — Foundational Principles

- **Citation**: Hermann Haken, *Synergetics* (1978/1988)
- **Summary**: Science of self-organization in open systems. Core concepts: order parameters, slaving principle (slow variables dominate fast variables), bifurcation, circular causality. Self-organization means massive reduction in degrees of freedom (entropy decrease), macroscopically appearing as "increased order."
- **Relevance**: ★★★★
- **Gap**: Synergetics has never been systematically applied to AI Agent architecture design. This is a major gap. Direct mappings: (1) order parameter = Agent's high-level goal/strategy; (2) slaving principle = high-level planning dominates low-level execution; (3) bifurcation = qualitative shift in Agent behavior mode (e.g., exploration to convergence); (4) control parameters = external conditions affecting Agent behavior (prompt, constraints, toolset).

---

## Category 8: Cybernetics in Software Engineering

### 27. Cybernetic Insights into Software Process and Architecture

- **Citation**: *Systems Research and Behavioral Science* (2010)
- **Summary**: Combines Stafford Beer's Viable System Model (VSM) with Boehm's spiral model. VSM provides adaptive capability for software architecture and self-organizing capability for software processes.
- **Relevance**: ★★★
- **Gap**: VSM's five systems (operation, coordination, control, intelligence, policy) can be directly mapped to different layers of multi-Agent architecture. Does not address AI Agents.

### 28. Cybernetical Intelligence: Engineering Cybernetics with Machine Intelligence

- **Citation**: Wiley textbook (2024)
- **Summary**: First textbook describing the development of machine learning from a cybernetics perspective. Proposes the concept of "Cybernetical Intelligence." Attempts to unify cybernetics and AI, but primarily focuses on the neural network level without addressing Agent architecture.
- **Relevance**: ★★★
- **Gap**: Neural network focused, does not address Agent architecture.

---

## Category 9: Existing Agent Systems — Control Theory Analysis Status

### 29. Direct Analysis of Hermes/Claude/GPT Agents

- **Citation**: Aggregated search results (2026)
- **Summary**: After extensive search, **no one has been found to systematically analyze Hermes Agent, Claude Agent, or GPT Agent architecture using control theory/cybernetics.** Specific findings:
  - **Hermes Agent**: Multiple technical articles discuss its learning loop, memory architecture, skill system — none framed in control theory language. One blog mentions "closed-loop learning" but only descriptively.
  - **Claude Code**: One comparative analysis (Claude Code vs Hermes Agent) discusses respective Agent loops from pure engineering implementation perspective, not elevated to control theory.
  - **GPT Agent**: No cybernetic analysis found.
  - **Agent architecture survey** (arXiv:2604.18071): Analyzes design decision dimensions of 70 open-source Agent systems without using a control theory framework.
- **Relevance**: ★★★★★ (the gap itself)
- **Gap**: **This is a clear gap — our analysis will be the first.**

---

## Gap Summary: 8 Specific Gaps Identified

These are the precise gaps identified in Phase 0 Section 10, representing the innovation space for our research:

1. **No one has applied Qian Xuesen's Engineering Cybernetics / Meta-Synthetic Engineering methodology to AI Agent architecture.** Despite Qian Xuesen's framework being directly applicable, no existing work bridges this Chinese cybernetics tradition to modern Agent design.

2. **No one has used cybernetics/control theory to analyze the specific architectures of Hermes Agent, Claude Agent, or GPT Agent.** All existing control-theoretic work on Agents is abstract — no concrete architecture analysis exists.

3. **No one has applied synergetics' order parameters / slaving principle to Agent hierarchical design.** Despite direct conceptual mappings (order parameter = high-level goal, slaving principle = high-level dominates low-level), this connection remains unmade.

4. **No one has applied the "Open Complex Giant Systems" framework to AI Agent systems.** AI Agent systems are OCGS by definition, yet no one has used this framework to reason about their governance.

5. **No one has applied "from qualitative to quantitative meta-synthesis" to Agent reasoning-action loops.** The qualitative-to-quantitative methodology directly maps to how Agents integrate natural language reasoning with quantitative tool execution.

6. **No one has applied Qian Xuesen's "Overall Design Department" concept to multi-Agent coordination.** The concept of a top-level integrating body coordinating specialized departments maps directly to multi-Agent orchestrator architectures.

7. **No one has used "Meta-Synthetic Wisdom Engineering" to understand Agent skill learning cycles.** Hermes Agent's learning loop (task → evaluation → skill extraction → memory persistence → future reuse) is essentially a micro-implementation of this concept, but the connection has never been articulated.

8. **No one has established a dialogue between the Chinese cybernetics tradition (Qian Xuesen) and the Western cybernetics tradition (Wiener/Ashby/Beer) in the context of AI Agent systems.** These two traditions developed in parallel with little cross-pollination in the AI era. Our work establishes this dialogue.

---

*This bibliography covers all 29 items from Phase 0 background survey. Each entry includes full citation, summary, relevance rating, and identified gap. The 8 specific gaps at the end define the innovation space for our research.*
