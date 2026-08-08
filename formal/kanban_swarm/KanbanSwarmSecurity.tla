---- MODULE KanbanSwarmSecurity ----
EXTENDS Naturals, TLC, KanbanSwarmProductionSemantics

Boards == {"boardA", "boardB", "none"}
GateKinds == {"none", "metadata_gate_pass", "unknown"}
TopologyStates == {"none", "valid", "invalid"}
SynthStates == {"todo", "ready"}
VerifierStatuses == {"running", "done", "archived"}
VerifierEvidence == {"pass", "fail", "missing", "malformed"}
RouteDecisions == {"none", "allow", "deny"}

ProductionGatePredicate(status, evidence, kind) ==
    kind = "metadata_gate_pass" /\ status = "done" /\ evidence = "pass"

ProductionRoutePredicate(pinKind, pinned, requested) ==
    \/ (pinKind = "db" /\ requested = "none")
    \/ (pinKind = "board" /\ (requested = "none" \/ requested = pinned))

VARIABLES graphVisible,
          taskCount,
          edgeExists,
          topology,
          topologySource,
          gate,
          verifierPass,
          synthStatus,
          graphCount,
          commentTopology,
          workerMode,
          pinnedBoard,
          requestedBoard,
          routeDecision

vars == <<graphVisible, taskCount, edgeExists, topology, topologySource,
          gate, verifierPass, synthStatus, graphCount, commentTopology,
          workerMode, pinnedBoard, requestedBoard, routeDecision>>

Init ==
    /\ graphVisible = FALSE
    /\ taskCount = 0
    /\ edgeExists = FALSE
    /\ topology = "none"
    /\ topologySource = "db"
    /\ gate = "none"
    /\ verifierPass = FALSE
    /\ synthStatus = "todo"
    /\ graphCount = 0
    /\ commentTopology = "none"
    /\ workerMode = FALSE
    /\ pinnedBoard = "none"
    /\ requestedBoard = "none"
    /\ routeDecision = "none"

CreateAtomic ==
    /\ ~graphVisible
    /\ graphVisible' = TRUE
    /\ taskCount' = 4
    /\ edgeExists' = TRUE
    /\ topology' = "valid"
    /\ topologySource' = "db"
    /\ gate' = "metadata_gate_pass"
    /\ graphCount' = 1
    /\ UNCHANGED <<verifierPass, synthStatus, commentTopology, workerMode,
                    pinnedBoard, requestedBoard, routeDecision>>

IdempotentRetry ==
    /\ graphVisible
    /\ UNCHANGED vars

PassVerifier ==
    /\ graphVisible
    /\ verifierPass' = TRUE
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, synthStatus, graphCount,
                    commentTopology, workerMode, pinnedBoard,
                    requestedBoard, routeDecision>>

CorruptUnknownGate ==
    /\ graphVisible
    /\ synthStatus = "todo"
    /\ gate' = "unknown"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, verifierPass, synthStatus, graphCount,
                    commentTopology, workerMode, pinnedBoard,
                    requestedBoard, routeDecision>>

CorruptTopology ==
    /\ graphVisible
    /\ synthStatus = "todo"
    /\ topology' = "invalid"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topologySource,
                    gate, verifierPass, synthStatus, graphCount,
                    commentTopology, workerMode, pinnedBoard,
                    requestedBoard, routeDecision>>

PromoteSynth ==
    /\ graphVisible
    /\ edgeExists
    /\ topology = "valid"
    /\ topologySource = "db"
    /\ gate = "metadata_gate_pass"
    /\ verifierPass
    /\ synthStatus = "todo"
    /\ synthStatus' = "ready"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, graphCount,
                    commentTopology, workerMode, pinnedBoard,
                    requestedBoard, routeDecision>>

ForgeCommentTopology ==
    /\ commentTopology' = "forged"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, workerMode, pinnedBoard, requestedBoard,
                    routeDecision>>

SetWorkerMode(mode) ==
    /\ mode \in BOOLEAN
    /\ workerMode' = mode
    /\ routeDecision' = "none"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, commentTopology, pinnedBoard, requestedBoard>>

SetPinnedBoard(board) ==
    /\ board \in Boards
    /\ pinnedBoard' = board
    /\ routeDecision' = "none"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, commentTopology, workerMode, requestedBoard>>

SetRequestedBoard(board) ==
    /\ board \in Boards
    /\ requestedBoard' = board
    /\ routeDecision' = "none"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, commentTopology, workerMode, pinnedBoard>>

ValidRoute ==
    ~workerMode \/
        (pinnedBoard # "none" /\
         (requestedBoard = "none" \/ requestedBoard = pinnedBoard))

Route ==
    /\ ValidRoute
    /\ routeDecision' = "allow"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, commentTopology, workerMode, pinnedBoard,
                    requestedBoard>>

DenyRoute ==
    /\ ~ValidRoute
    /\ routeDecision' = "deny"
    /\ UNCHANGED <<graphVisible, taskCount, edgeExists, topology,
                    topologySource, gate, verifierPass, synthStatus,
                    graphCount, commentTopology, workerMode, pinnedBoard,
                    requestedBoard>>

Next ==
    CreateAtomic
    \/ IdempotentRetry
    \/ PassVerifier
    \/ CorruptUnknownGate
    \/ CorruptTopology
    \/ PromoteSynth
    \/ ForgeCommentTopology
    \/ \E mode \in BOOLEAN : SetWorkerMode(mode)
    \/ \E pinBoard \in Boards : SetPinnedBoard(pinBoard)
    \/ \E requestBoard \in Boards : SetRequestedBoard(requestBoard)
    \/ Route
    \/ DenyRoute

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ graphVisible \in BOOLEAN
    /\ taskCount \in 0..4
    /\ edgeExists \in BOOLEAN
    /\ topology \in TopologyStates
    /\ topologySource = "db"
    /\ gate \in GateKinds
    /\ verifierPass \in BOOLEAN
    /\ synthStatus \in SynthStates
    /\ graphCount \in 0..1
    /\ commentTopology \in {"none", "forged"}
    /\ workerMode \in BOOLEAN
    /\ pinnedBoard \in Boards
    /\ requestedBoard \in Boards
    /\ routeDecision \in RouteDecisions

AtomicGraphVisibility ==
    graphVisible => (taskCount = 4 /\ edgeExists /\ graphCount = 1)

NoPartialGraphBeforeCommit ==
    ~graphVisible => (taskCount = 0 /\ ~edgeExists /\ topology = "none" /\ graphCount = 0)

SingleIdempotentGraph == graphCount <= 1

SynthesisRequiresAuthority ==
    synthStatus = "ready" =>
        (graphVisible /\ edgeExists /\ topology = "valid" /\
         topologySource = "db" /\ gate = "metadata_gate_pass" /\ verifierPass)

UnknownGateFailsClosed == gate = "unknown" => synthStatus # "ready"

ForgedCommentIsNotAuthority == topologySource = "db"

WorkerBoardPinning == routeDecision = "allow" => ValidRoute

\* This is a state invariant (through routeDecision) and a concrete refinement
\* check: the generated module is derived from executable production probes.
ProductionRefinementMap ==
    /\ routeDecision \in RouteDecisions
    /\ \A c \in ProductionGateCases :
        ProductionGateExpected(c) =
            ProductionGatePredicate(
                ProductionGateStatus(c), ProductionGateEvidence(c), ProductionGateKind(c))
    /\ \A c \in ProductionBoardCases :
        ProductionBoardExpected(c) =
            ProductionRoutePredicate(
                ProductionPinKind(c), ProductionPinnedBoard(c), ProductionRequestedBoard(c))

====
