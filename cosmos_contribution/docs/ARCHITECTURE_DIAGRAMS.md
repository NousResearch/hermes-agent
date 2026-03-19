# cosmos System Architecture

## 1. High-Level Architecture
The system follows a modular, event-driven architecture centered around the **Nexus** event bus.

```mermaid
graph TD
    User([User / Claude Code]) <--> MCP[MCP Server]
    MCP <--> Nexus{Nexus Event Bus}
    
    subgraph Core [Core Cognition]
        Nexus <--> Planner[Planner Agent]
        Nexus <--> Proactive[Proactive Agent]
        Nexus <--> Swarm[Swarm Orchestrator]
        Nexus <--> Inference[Inference Engine (LLM)]
    end
    
    subgraph Memory [Unified Memory System]
        Nexus <--> MemSys[Memory System]
        MemSys <--> Working[Working Memory]
        MemSys <--> Archival[(Archival DB)]
        MemSys <--> Graph[(Knowledge Graph)]
        MemSys <--> Context[Virtual Context]
    end
    
    subgraph Evolution [Evolutionary Layer]
        Nexus --> Fitness[Fitness Tracker]
        Fitness --> Optimizer[Genetic Optimizer]
        Optimizer --> Behavior[Behavior Mutation]
    end
    
    subgraph Integration [Integrations]
        Swarm <--> Tools[Tool Router]
        Tools <--> Web[Web Agent]
        Tools <--> Vision[Vision Module]
        Tools <--> Solana[Solana DeGen Mob]
    end
```

## 2. Memory Data Flow
How information moves from short-term to long-term storage via Dreaming.

```mermaid
graph LR
    Input(User Input) --> VC[Virtual Context]
    VC --> WM[Working Memory]
    
    subgraph Active_Processing
        WM <--> Recall[Recall Memory]
    end
    
    WM -- "Consolidation (Dreaming)" --> Dreamer((Memory Dreamer))
    
    subgraph Long_Term_Storage
        Dreamer --> Archival[(Archival Vector DB)]
        Dreamer --> KG[(Knowledge Graph)]
    end
    
    Archival -- "Retrieval (RAG)" --> VC
    KG -- "Retrieval (RAG)" --> VC
```

## 3. Advanced Inference Pipeline
The `InferenceEngine` uses a sophisticated cascade and swarm approach.

```mermaid
flowchart TD
    Request(Prompt) --> Analyzer{Complexity Analyzer}
    
    Analyzer -- "Low Complexity" --> BitNet[BitNet / Small Model]
    Analyzer -- "High Complexity" --> SwarmMode{Swarm Selector}
    
    subgraph Swarm Decisions
        SwarmMode -- "Code/Reasoning" --> MoE[Mixture of Experts]
        SwarmMode -- "Creative/Open" --> Parallel[Parallel Vote]
        SwarmMode -- "Speed Critical" --> Spec[Speculative Decoding]
    end
    
    BitNet -- "Low Confidence" --> Escalate[Escalation Trigger]
    Escalate --> SwarmMode
    
    MoE --> Result
    Parallel --> Result
    Spec --> Result
    BitNet --> Result
    
    Result --> Verifier{Verifier Agent}
    Verifier -- "Approved" --> Final(Output)
    Verifier -- "Rejected" --> Refine[Refinement Loop]
    Refine --> SwarmMode
```

## 4. Evolutionary Feedback Loop
How cosmos improves over time.

```mermaid
stateDiagram-v2
    [*] --> ActiveState
    
    state ActiveState {
        TaskExecution --> OutcomeRecording
        OutcomeRecording --> FitnessUpdate
    }
    
    FitnessUpdate --> EvolutionCheck
    
    state EvolutionCheck <<choice>>
    EvolutionCheck --> EvolutionCycle : Threshold Met
    EvolutionCheck --> ActiveState : Continue
    
    state EvolutionCycle {
        AnalyzeMetrics --> SelectElites
        SelectElites --> MutateBehaviors
        MutateBehaviors --> UpdateConfig
    }
    
    UpdateConfig --> ActiveState
```

## 5. DeGen Mob Integration
The specialized Solana trading subsystem.

```mermaid
graph TD
    Command("Trade Command") --> Router{Tool Router}
    
    Router -- "Scan/Snipe" --> Sniper[Launch Sniper]
    Router -- "Audit" --> Rug[Rug Detector]
    Router -- "Track" --> Whale[Whale Watcher]
    Router -- "Execute" --> Trader[Trading Engine]
    
    subgraph Intelligence
        Rug <--> Helius[Helius API]
        Whale <--> RPC[Solana RPC]
        Sniper <--> RPC
    end
    
    subgraph Execution
        Trader --> Jupiter[Jupiter Aggregator]
        Trader --> Pump[Pump.fun API]
        Trader --> Meteora[Meteora DLMM]
    end
    
    Intelligence --> Nexus
    Execution --> Nexus
```
