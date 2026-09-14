# platform/

The **platform-owned** half of this repository — the business, tenant and control layer built
around the upstream Hermes Agent runtime.

Nothing here is imported by upstream code. Nothing here may be imported *into* upstream code.

**Start with [`AGENTS.md`](AGENTS.md)** for the working rules, and
[`../docs/platform/ARCHITECTURE_BOUNDARIES.md`](../docs/platform/ARCHITECTURE_BOUNDARIES.md) for
the authoritative boundary definition.

| Document | What it covers |
|---|---|
| `docs/platform/ARCHITECTURE_BOUNDARIES.md` | The upstream/platform line, protected identifiers, control-plane and deployment boundaries, guardrails |
| `docs/platform/IMPLEMENTATION_PLAN.md` | Extension points, module layout, identity layer, dashboard/runtime communication, AWS boundary, tenant config, Phase 1 |
| `docs/platform/PHASE_1.md` | What Phase 1 built, its known limitations, and its extension points |
| `docs/platform/PHASE_2.md` | Policy and governance: how enforcement works, and what it cannot yet express |
| `docs/platform/PHASE_3.md` | Budget controls: what is enforced, what is only observed, and why |
| `docs/platform/PHASE_4.md` | Knowledge: declared corpora, scoped retrieval, and the trust boundary around it |
| `docs/platform/PHASE_5.md` | Supervisor: objectives routed under the tenant's delegation policy |
| `docs/platform/LIVE_RUN.md` | The end-to-end worker run, the four defects it found, and the gap it left |
| `docs/platform/DEPLOYMENT_CONFIG_AUDIT.md` | How provider config, env and secrets actually reach a worker |
| `docs/platform/PHASE_6.md` | The deployment seam: endpoints, credential names, and readiness |
| `docs/platform/BUDGET_ENFORCEMENT_AUDIT.md` | What the runtime exposes for usage and limits, and which of it is enforceable |
| `docs/platform/KNOWLEDGE_CAPABILITY_AUDIT.md` | What exists for knowledge, retrieval and documents; what to reuse versus build |
| `docs/platform/CORE_PATCHES.md` | The patch budget — every upstream file touched, and why |
| `docs/platform/HERMES_PLATFORM_AUDIT.md` | Phase 0 assessment of the existing runtime |
| `docs/platform/BRAND_SURFACE_AUDIT.md` | All 101,368 product-name occurrences, classified |

## Layout

Directories appear as their phase lands.

```
spec/           AgentSpec, IdentitySpec, OrganizationSpec, TenantBundle    [built]
runtime/        AgentRuntime contract, registry, and the Hermes adapter    [built]
audit/          Append-only log enforcing "model-visible means logged"     [built]
policy/         Declaration, compilation, decisions and limit vocabulary   [built]
control/        Control API (read-only) + the dashboard                    [built]
knowledge/      Declare -> chunk -> index -> search, scoped per agent      [built]
supervisor/     Objectives routed under the declared delegation policy     [built]
apply.py        Bundle -> runtime orchestration                            [built]
cli.py          `validate|plan|apply|status|doctor|knowledge|objective`   [built]
examples/       Two agents, two corpora, one objective                     [built]

observability/  JSON logs, correlation IDs, CloudWatch                     [planned]
```

Dependencies are the standard library and PyYAML — nothing else. The layer imports and
its tests run without the runtime installed.

## Try it

```bash
python -m nova validate nova/examples/acme
python -m nova plan     nova/examples/acme
python -m nova apply    nova/examples/acme
python -m nova status
python -m nova doctor  nova/examples/acme   # what each agent still needs

python -m nova knowledge ingest nova/examples/acme
python -m nova knowledge search nova/examples/acme "refund approval" --as-agent customer-support

python -m nova objective plan   nova/examples/acme quarterly-refund-audit
python -m nova objective submit nova/examples/acme quarterly-refund-audit
python -m nova objective status nova/examples/acme

python -m nova serve nova/examples/acme   # dashboard on 127.0.0.1:8787
```

## Three rules, in short

1. **The dependency arrow points one way.** Platform imports Hermes; never the reverse.
2. **Branding is compiled into existing surfaces**, never hard-coded and never read from core.
3. **Add names beside upstream ones; never replace them.**
4. **The package is `nova/`, never `platform/`** — a root `platform/` package shadows
   the stdlib module 40 upstream files import. The guardrail enforces this.
