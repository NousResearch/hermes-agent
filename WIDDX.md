# WIDDX

**WIDDX — Neural AI Agent**

WIDDX is the independent evolution of this Hermes-based agent fork toward an event-driven artificial nervous system.

## Neural foundation

The first foundation lives under `neural/` and currently provides:

- immutable `NeuralEvent` and `NeuralSignal` primitives;
- a failure-isolated in-process `NeuralBus`;
- deterministic `Neuron` and bounded `Synapse` primitives;
- observation-only `Perception`, `Guardian`, and `Reflex` boundaries;
- extension points for `Brain`, `Memory`, and `SelfModel`.

The existing Hermes runtime remains the execution core while these components are introduced incrementally.

## Safety boundary

Neural learning is advisory. It may improve prioritization and planning, but it cannot modify approval rules, command blocklists, sandbox policy, credentials, or other security boundaries.

## Compatibility

The `hermes` command and existing Hermes configuration remain compatibility interfaces during the migration. Repository and package identity changes are intentionally staged rather than performed as a breaking rename in one step.
