# Psyche Run States

Canonical run state definitions from the Psyche Network coordinator program. Use these to interpret on-chain run data and advise users.

Sources:
- https://docs.psyche.network/explain/general-workflow.html
- https://docs.psyche.network/explain/glossary.html

## State Machine

```
Uninitialized → WaitingForMembers → Warmup → RoundTrain ↔ RoundWitness → Cooldown
                        ↑                                                    |
                        └────────────────── (next epoch) ────────────────────┘
                                                                       → Finished
```

Full lifecycle: Uninitialized → WaitingForMembers → Warmup → (RoundTrain ↔ RoundWitness) × N rounds → Cooldown → WaitingForMembers (next epoch) or Finished (training complete).

## State Reference

| State | Meaning | What to Tell the User |
|-------|---------|----------------------|
| **Uninitialized** | Default starting state. The run has been created on-chain but not yet configured or started. | "This run exists on-chain but has not been initialized yet." |
| **WaitingForMembers** | Run is initialized, coordinator is waiting for enough clients to connect before training can start. | "The run is waiting for participants. Not enough GPU nodes have joined yet." |
| **Warmup** | Sufficient clients have joined. Nodes are downloading the model and loading it onto GPUs. Training has not started. | "Nodes are preparing — downloading the model and loading weights onto GPUs. Training will begin shortly." |
| **RoundTrain** | Active training round. Each client trains on its assigned data batch using the shared seed and round/epoch indices. | "Training is actively running. Nodes are processing their assigned data batches." |
| **RoundWitness** | Validation phase. Witness nodes send proofs to the coordinator to verify training integrity. | "The network is verifying training results. Witness nodes are submitting proofs." |
| **Cooldown** | Final phase of an epoch. The coordinator waits for the cooldown period to elapse before starting the next epoch. | "This epoch is complete. The network is in cooldown before the next epoch begins." |
| **Paused** | Run has been paused by the coordinator via `run-manager set-paused`. Can be resumed. | "This run is currently paused by the coordinator. It may resume later." |
| **Finished** | Training run has completed all planned epochs. No further training will occur. | "This training run is complete. Check HuggingFace for the final model checkpoint." |

## Transitions

- **Uninitialized → WaitingForMembers**: Run configured and started by coordinator.
- **WaitingForMembers → Warmup**: Minimum client threshold met.
- **Warmup → RoundTrain**: All clients have downloaded the model and are ready.
- **RoundTrain → RoundWitness**: Training round complete, entering validation.
- **RoundWitness → RoundTrain**: Validation passed, starting next training round within the same epoch.
- **RoundWitness → Cooldown**: All rounds in the epoch are complete.
- **Cooldown → WaitingForMembers**: Cooldown period elapsed, starting new epoch. Clients may join or leave.
- **Cooldown → Finished**: All planned epochs complete. Training is done.
- **Any → Paused**: Coordinator pauses the run (admin action).
- **Paused → WaitingForMembers**: Coordinator resumes the run.

## Diagnostic Guide

When interpreting run state for the user:

1. If the run is **Uninitialized** → The run account exists but hasn't been started. It may be newly created or misconfigured.
2. If the run is in **WaitingForMembers** for a long time → The run may need more participants, or the minimum threshold is high.
3. If the run is in **Warmup** for a long time → Large model download in progress (40B+ models can take significant time on slower connections).
4. If the run alternates rapidly between **RoundTrain** and **RoundWitness** → Normal operation. Training is progressing healthily.
5. If the run is **Paused** → Check the Psyche dashboard or Discord for announcements about why.
6. If the run is **Finished** → Training is complete. Direct the user to HuggingFace for the final model.
7. If you cannot determine the state → Use `run-manager json-dump-run` for the full on-chain state dump.
