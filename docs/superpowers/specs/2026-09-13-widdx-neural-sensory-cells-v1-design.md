# WIDDX Neural Sensory Cells v1 Design

## Goal

Turn the existing WIDDX neural foundation into a small, explicit perception layer that can observe Hermes runtime activity, normalize observations into immutable neural events, process them through advisory neurons, and retain bounded learning state without acquiring execution or authorization authority.

## Non-goals

- Replacing Hermes terminal, approval, sandbox, gateway, or tool systems.
- Adding a second permission or authorization system.
- Allowing neural components to execute tools, shell commands, Python code, network operations, or mutate Hermes security policy.
- Building autonomous planning or LLM reasoning inside the neural package in this iteration.

## Architecture

The first loop is:

`runtime observation -> sensory cell -> NeuralEvent -> NeuralBus -> neuron -> optional synapse update -> advisory output`

Sensory cells are adapters only. They accept already-observed runtime facts and create normalized events; they do not execute or authorize anything. Neurons transform bounded events into bounded signals. Synapses retain bounded statistical state. Guardian and Reflex remain advisory components. Hermes remains the authority for action and policy.

## Components

### Sensory cells

Introduce a focused `SensoryCell` protocol/base contract with a stable name and an `observe` operation. Provide small concrete cells for conversation/task, terminal/tool result, error, and environment observations. Each cell accepts structured observation data supplied by Hermes and emits a `NeuralEvent`; no cell imports terminal execution, approval, gateway authorization, subprocess, or network clients.

### Event boundary

Preserve immutable event metadata and bounded confidence/importance. Normalize payloads at the sensory boundary so mutable caller dictionaries are not retained by reference. Event creation remains side-effect-free.

### Neural processing

Add a deterministic processing path that can consume an event, invoke registered neurons, and publish resulting signals without executing tools. Processing failures are isolated per neural observer and never become an execution dependency.

### Learning state

Keep `Synapse` bounded and advisory. Reinforcement must be finite, deterministic, and incapable of changing Hermes policy. Learning state is an optimization/context signal, not authorization state.

## Data flow

1. Hermes observes an event through its existing runtime.
2. A sensory cell validates/normalizes the observation.
3. A `NeuralEvent` is published to the in-process bus.
4. Registered neurons derive bounded `NeuralSignal` values.
5. Optional synapses update bounded learning statistics.
6. Consumers may use the resulting observations for context, prioritization, or future intelligence.
7. Any real action continues through Hermes' existing tool and approval path.

## Error handling

- Invalid sensory input raises a clear validation error at the boundary.
- Neural observer exceptions remain isolated by the bus.
- Neural processing is best-effort and must not break the primary Hermes execution path.
- No fallback may silently execute an action.

## Testing

Use TDD. Tests must prove:

- sensory cells produce normalized events without side effects;
- caller payload mutation after observation cannot mutate the event payload;
- event metadata remains immutable;
- processing produces bounded deterministic signals;
- synapse reinforcement remains bounded and counters stay consistent;
- bus observers remain isolated from one another;
- neural modules have no direct execution/authorization integration;
- the full sense-to-signal path works without invoking terminal or execute-code facilities.

## Performance

The initial implementation must remain synchronous, in-process, dependency-light, and bounded. No background worker, network service, or LLM call is introduced. Neural observation must remain cheap enough to attach to runtime events without becoming a mandatory latency bottleneck.

## Security boundary

The neural layer is advisory. It may observe and infer; it may not grant permission, bypass approval, alter YOLO state, modify security policy, access credentials, or directly invoke execution facilities. This is enforced through module boundaries and regression tests rather than a new permission framework.
