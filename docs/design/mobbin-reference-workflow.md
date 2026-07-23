# Mobbin Reference Workflow

Mobbin MCP is a reference and prototype accelerator. It should be connected at the agent/tooling layer, not embedded inside Kashi VC, Media Engine, or another single dashboard project.

## Ownership

| Layer | Responsibility |
| --- | --- |
| Mobbin MCP | Search real shipped UI references and return screen-level inspiration. |
| Hermes / Nous Hermes Agent | Convert references into reusable dashboard patterns and prototype directions. |
| `@hermes/dashboard-kit` | Own shared components, contracts, workspace model, and implementation rules. |
| Hermes OS | Govern adoption, dashboard quality, project registry, and cross-project readiness. |
| Individual projects | Consume the kit and expose truthful data contracts. |

## When To Use Mobbin

Use Mobbin when a dashboard needs a design direction before implementation:

- executive command centers
- operational dashboards
- analytics consoles
- monitoring and incident surfaces
- cost/capacity dashboards
- research/evidence dashboards
- media publishing operations
- trading/market intelligence tools

Do not use Mobbin as a shortcut around domain modeling. If the data shape is unknown, first define the `DashboardSnapshotContract`.

## Reference Brief Template

Use this prompt shape when asking for references:

```text
Find real product dashboard references for a dense operational dashboard.
Domain: [Kashi market intelligence / Media Engine publishing / TLC executive OS]
Primary operator: [founder / operator / analyst]
Main decision: [what should I do now?]
Must show: [alerts, trend, table, queue, cost, health, findings]
Avoid: [marketing pages, decorative cards, low-density analytics]
Need 3-4 directions with distinct IA patterns.
```

## Prototype Directions

Every Mobbin-assisted design pass should produce three or four directions:

| Direction | When To Use |
| --- | --- |
| Command First | The operator needs immediate status, blockers, and actions. |
| Intelligence First | The operator needs to understand what the system is learning. |
| Operations First | The operator needs to inspect live work, jobs, queues, and failures. |
| Capacity First | The operator needs to manage budgets, throughput, storage, and API usage. |

Each direction should state:

- which references influenced it
- what problem it solves
- what it hides or deprioritizes
- which kit components it needs
- what data contracts must exist before implementation

## Promotion Rules

Mobbin references can influence layout and interaction patterns, but the final design must:

- use Hermes dashboard-kit components where possible
- avoid copying reference screens pixel-for-pixel
- map into the six Hermes workspaces
- support loading, empty, error, warning, and critical states
- keep operator actions visually separate from read-only insight
- preserve data density without becoming unreadable

## Current Limitation

The MCP is configured through Codex OAuth, but individual Codex sessions may need a reload or fresh session before the Mobbin tools appear. Production dashboards must not depend on Mobbin at runtime.

