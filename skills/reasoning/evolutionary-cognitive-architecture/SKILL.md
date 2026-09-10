---
name: evolutionary-cognitive-architecture
description: Biologically-inspired multi-system decision framework using parallel valuation, contextual weighting, and competitive arbitration.
author: ibrhmuyls
version: 0.1.0
license: MIT
tags:
  - cognition
  - reasoning
  - decision-making
  - evolutionary-psychology
  - behavioral-science
---

# Evolutionary Cognitive Architecture (ECA)

## Overview

Evolutionary Cognitive Architecture (ECA) is a cognitive decision-making framework inspired by converging evidence from evolutionary biology, evolutionary psychology, cognitive science, behavioral neuroscience, and reinforcement learning.

Rather than treating intelligence as a single optimization process, ECA models cognition as a distributed set of competing valuation systems that evaluate actions under uncertainty and resource constraints.

Decisions emerge from **parallel motivational systems, context-dependent weighting, and competitive arbitration**, rather than a single centralized objective function.

---

## Core Principles

### 1. Distributed Modularity

Cognition arises from multiple interacting systems:

* Threat System (risk minimization)
* Reward System (value maximization)
* Exploration System (information gain / novelty)
* Social System (cooperation / group dynamics)
* Status System (hierarchy / influence)
* Resource System (cost / energy / time)
* Uncertainty System (confidence estimation)

These systems operate in parallel and produce competing signals.

---

### 2. Bounded Rationality

Agents operate under:

* limited information
* limited computation
* limited time

Decision-making is based on **satisficing**, not global optimization.

```text
Good enough > Optimal (intractable)
```

---

### 3. Parallel Valuation

All systems evaluate simultaneously:

* Threat Value
* Reward Value
* Exploration Value
* Social Value
* Status Value
* Cost Value
* Confidence Value

These values are not inherently comparable without contextual normalization.

---

### 4. Context-Dependent Weighting

System influence varies dynamically depending on context:

* High risk → Threat dominance increases
* High uncertainty → Exploration increases
* Social environment → Social/Status increase
* Resource scarcity → Cost increases

---

### 5. Competitive Arbitration

Final decisions emerge from competition between systems, not aggregation.

Mechanisms:

* mutual inhibition
* salience amplification
* context-based weighting

```text
Action = argmax(competing weighted signals)
```

---

### 6. Exploration–Exploitation Trade-off

Core biological constraint:

* Exploitation = known reward maximization
* Exploration = information gain under uncertainty

Decision pressure:

```text
Expected Reward
+ Information Gain
- Threat Cost
- Opportunity Cost
```

---

### 7. Predictive Processing Loop

```text
Prediction → Observation → Prediction Error → Update
```

Learning is driven by prediction error across all systems.

---

## Architecture

### Input Layer

* User instructions
* Environmental signals
* Memory retrieval
* Tool outputs

---

## Parallel Cognitive Systems

### Threat System

* Risk magnitude
* Failure probability
* Loss severity

Output: Threat Value

---

### Reward System

* Expected payoff
* Utility estimation
* Long-term benefit

Output: Reward Value

---

### Exploration System

* Novelty
* Uncertainty reduction
* Information gain

Output: Exploration Value

---

### Social System

* Cooperation potential
* Trust dynamics
* Group impact

Output: Social Value

---

### Status System

* Influence gain/loss
* Hierarchical impact
* Prestige dynamics

Output: Status Value

---

### Resource System

* Time cost
* Energy cost
* Cognitive load
* Opportunity cost

Output: Cost Value

---

### Uncertainty System

* Confidence estimation
* Unknown variables
* Prediction stability

Output: Confidence Score

---

## Decision Arbitration Layer

Example state:

```
Threat: 0.80
Reward: 0.65
Exploration: 0.90
Social: 0.40
Cost: 0.70
Confidence: 0.60
```

Arbitration process:

* contextual weighting
* inhibitory competition
* salience amplification

Output:

* Selected Action
* Confidence estimate
* Risk profile

---

## Decision Pipeline

```
INPUT
↓
Parallel Valuation Systems
↓
Contextual Weighting
↓
Competitive Arbitration
↓
Action Selection
↓
Execution
↓
Outcome Observation
↓
Prediction Error Computation
↓
Memory Update
↓
Policy Adaptation
```

---

## Learning Layer

### Reward Prediction Error

```text
Prediction Error = Actual Outcome - Expected Outcome
```

Used to update:

* reward expectations
* threat calibration
* exploration policy
* social valuation

---

### Memory System

Stores:

* successful strategies
* failed strategies
* social outcomes
* risk outcomes
* prediction errors

---

### Heuristic Compression

Repeated decisions are compressed into fast heuristics while preserving adaptive accuracy.

---

## Agent Behavioral Properties

* Risk-sensitive adaptation
* Exploration–exploitation balancing
* Socially aware reasoning
* Reputation sensitivity
* Cost-aware decision making
* Uncertainty-driven caution
* Long-term reward optimization

---

## Intended Use Cases

* Autonomous agents
* Multi-agent systems
* Strategic planning systems
* Research assistants
* Simulation environments
* Cognitive modeling
* Decision support systems

---

## Scientific Position

This architecture is not a brain simulation.

It is a computational framework inspired by:

* evolutionary theory of behavior
* reinforcement learning
* predictive processing
* bounded rationality
* behavioral neuroscience

It models cognition as emergent behavior from competing adaptive systems.

---

## Key Distinction

Traditional agents:

```
Goal → Plan → Execute
```

ECA agents:

```
Competing systems → Context weighting → Arbitration → Emergent action
```

---

# Nouse Skill Specification (Integration Layer)

## Skill Name

Evolutionary Cognitive Architecture (ECA)

---

## Skill Type

Decision-Making / Cognitive Reasoning / Behavioral Policy Layer

---

## Runtime Model

### Step 1: State Construction

Collect:

* input query
* memory context
* environmental signals
* tool outputs

---

### Step 2: Parallel Valuation Execution

Compute:

```
Threat Score
Reward Score
Exploration Score
Social Score
Status Score
Cost Score
Confidence Score
```

---

### Step 3: Context Weighting

Adjust system weights based on:

* urgency
* risk level
* uncertainty
* social context
* resource constraints

---

### Step 4: Arbitration Engine

Compute:

```
Weighted Competition Function:
Σ (System Value × Context Weight × Inhibition Factors)
```

Select highest-salience action.

---

### Step 5: Output Formation

Return structured output:

```json
{
"action": "",
"rationale": {
"threat": 0.0,
"reward": 0.0,
"exploration": 0.0,
"social": 0.0,
"status": 0.0,
"cost": 0.0,
"confidence": 0.0
},
"risk_profile": "",
"expected_value": 0.0
}
```

---

### Step 6: Learning Update

After execution:

* compute prediction error
* update system weights
* store outcome in memory
* refine heuristics

---

## Tool Calling Policy

* Always evaluate internal systems before tool usage
* Use tools only if:

* information uncertainty is high
* external verification improves decision quality
* Tool outputs are treated as environmental signals, not truth

---

## Execution Constraint

The system must always prioritize:

1. Safety / Threat minimization
2. Resource efficiency
3. Reward maximization
4. Exploration when safe
5. Social coherence

---

## End of Specification
