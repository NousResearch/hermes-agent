# Local GPU admission controller (candidate)

This plugin registers one `llm_admission` controller for an individual Hermes profile. It only
allows tasks that have all of these properties:

- a matching capability record with `allow: true`;
- an explicit, independently reviewed mapping from the operational capability ID to the exact
  trial task class referenced by the profile-route evidence;
- an exact profile, model digest, quantization, Ollama runtime version, provider and endpoint match;
- exact agreement between the measured context window, configured `ollama_num_ctx`, and current request;
- an active dispatcher-owned Kanban run whose DB row, claim lock, worker PID and contract digest
  agree with the current request;
- a task-specific verifier and expected fixture whose hashes match the pinned registry; and
- a fresh, unblocked Resource Balancer `gpu` lease.

At completion it compares exactly one JSON artifact to the fixture object captured before the
first model request and refuses the Kanban transition if the task/run/lease or artifact differs.
The fixture is an independent verifier oracle and must not be copied into the task body or model
context; doing so leaks the expected answer and invalidates capability evidence. The controller
uses the fixture's keys and JSON types to constrain response shape, while the answer values remain
available only to trusted completion verification.
The artifact must be inside the task workspace. This version supports only task IDs with an
explicit prepared external verifier in the capability registry; it does not admit general code
generation, repository edits, or profiles/models without matching trial evidence.
Each registry `verifier_path` and `expected_fixture_path` is resolved relative to the directory
containing that registry file; pin the SHA-256 of the full registry bytes in the profile config.

Auxiliary model calls are intentionally excluded from this capability: compression summaries,
memory/search/vision helpers, and auxiliary recovery/fallback destinations that resolve to a local
endpoint are refused before provider I/O because those calls do not carry the task/run-bound GPU
lease. A refusal during compression is terminal and must not retry through the main model. This
keeps the current fixed-JSON envelope tool-free and one-request-only; do not interpret the auxiliary
deny path as support for local auxiliary inference.

## Profile configuration (example only)

Configure the plugin inside each eligible profile's `config.yaml`; this candidate does not edit
any installed profile:

```yaml
plugins:
  entries:
    local_gpu_admission:
      enabled: true
      settings:
        resource_balancer_python_path: /path/to/synapsium/tools
        resource_balancer_state_dir: /path/to/resource-balancer/state
        capability_registry_path: /path/to/local-model-gpu-strategy/capability-registry.json
        capability_registry_sha256: <operator-reviewed SHA-256 of that exact registry>
```

The plugin must run in-process because the current plugin-host protocol does not carry behavior
changing middleware. Missing settings, duplicate controllers, stale status, DB mismatch, or any
exception deny local inference. Pin the registry SHA again whenever its reviewed contents change.

Resource Balancer coordinates only clients that acquire its lease. This plugin does not block
direct Ollama HTTP calls, independent vLLM/llama.cpp servers, or arbitrary CUDA processes. A shared
GPU is not globally exclusive until those paths are restricted to the same admission boundary;
do not claim system-wide GPU ownership from this candidate alone.
