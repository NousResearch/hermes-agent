---
name: scaffold-mvp
license: MIT
version: 2.1.1
author: afonsoft
platforms: [linux, macos, windows]
description: "Scaffold a new .NET/Blazor/Angular MVP project."
metadata:
  hermes:
    tags: [Scaffolding, MVP, dotnet, Blazor, Angular]
  version: "2.1.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Scaffold MVP

## When to Use

- User asks or mentions this skill in English (e.g., "use /scaffold-mvp", "run scaffold-mvp").
- O usuário pede ou menciona esta skill em português (ex.: "use /scaffold-mvp", "execute scaffold-mvp").

## Trigger

This skill is activated **only** in brand-new repositories, right after `/write-specs` has established the shared language, domain, and an approved SPEC SDD, or when the user explicitly asks for a quick MVP bootstrap.

## Golden Rule (Non-Negotiable)

**It is strictly forbidden to build base UI components or infrastructure from scratch.**

Reuse shared components and mature libraries focused on prototyping speed. This is the highest priority and non-negotiable.

- **Blazor**: prefer MudBlazor, Radzen, Fluent UI Blazor, or Bootstrap Blazor.
- **Angular**: prefer Angular Material, PrimeNG, or NG-ZORRO.
- **.NET backend**: prefer ABP Framework, FastEndpoints, or minimal APIs with Carter.
- **Mobile/web hybrid**: prefer MAUI Blazor Hybrid, Ionic Angular, or similar proven kits.

## Stability and Technical Cadence (No Rushing)

MVP speed must not produce unstable code or structural shortcuts. Syntactic and architectural integrity is sovereign. The agent must follow:

1. **Incremental compilation check**: After installing any dependency or creating a base directory, run the local build or type check command (e.g., `dotnet build`, `ng build`, `npm run build`). Never accumulate changes without confirming the current build passes.
2. **Zero pseudo-code**: Escape comments such as `// ...` or `// rest of the code here` are forbidden in routes or scaffold files. Every created file must be self-contained and commercially functional.
3. **Safe dependency setup**: Pin exact library versions. Always run the explicit install command to ensure clean lockfile updates (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`).
4. **Bridge and contract building**: If the MVP depends on external services (database, cache, auth, queue, payment), provide locally usable stubs or mocks. Avoid unhandled crashes on the first startup.

## Workflow

### Phase 0 — Agent Harness

Before writing any application code, invoke `/create-agent-harness` to install the agent harness in the new repository.

After it completes, confirm the following files/directories exist:

- `CLAUDE.md` / `AGENTS.md`
- `.claude/CONTEXT.md`
- `.claude/MEMORY.md`
- `.claude/rules/`
- `.claude/skills/` and `.claude/hooks/`
- `docs/`
- `.git` with `main` branch

If the harness is already present, skip this phase and read `.claude/CONTEXT.md` instead.

### Phase 1 — Project Summary and Stack Proposal

Do not perform generic business interrogation; the predecessor skill already established the domain. Ask only for a brief project summary to refine the stack proposal.

1. **Read** `.claude/CONTEXT.md`, `docs/architecture/` (if any), and the approved `.specs/SPEC-*.md`.
2. **Ask the user for a one-line summary** in Portuguese:

```text
Resumo rapido do projeto: qual e o objetivo principal e quem sao os usuarios?

➡️ Meu palpite: [resumo baseado no SPEC lido]
```

3. Based on the domain, propose the most productive .NET/Blazor/Angular stack. Be consultative and opinionated in favor of speed.

### Phase 2 — Stack Decision

Use the decision tree below. Confirm the chosen stack with the user in Portuguese before proceeding.

| Profile | Default MVP stack | UI / component library |
| --- | --- | --- |
| Web dashboard / admin / SaaS (Blazor) | .NET 8 + Blazor Server or Blazor WebAssembly + EF Core | MudBlazor or Radzen or Fluent UI Blazor |
| Web dashboard / admin / SaaS (Angular) | .NET 8 Web API + Angular 17+ | PrimeNG or Angular Material or NG-ZORRO |
| API / backend only | .NET 8 minimal API or FastEndpoints or ABP | — |
| CLI / worker | .NET 8 console or worker service | — |
| Mobile / hybrid | .NET MAUI Blazor Hybrid | MudBlazor or Fluent UI |

**Mandatory validation:**

```text
Baseado no nosso contexto de dominio, proponho iniciar com [STACK_ESCOLHIDA] para maxima produtividade sem reinventar a roda.

Voce concorda com esta stack ou temos alguma restricao tecnica ainda nao mapeada?

➡️ Meu palpite: concordo com a proposta.
```

Wait for explicit user approval. If the user wants changes, adapt the bootstrap. If the user agrees, proceed.

### Phase 3 — Technical and Structural Execution

Proceed only after explicit user approval.

1. **Project skeleton** — create the solution/project structure:
   - .NET solution (`.sln`) and projects (`src/`, `tests/`)
   - `Directory.Build.props` or `Directory.Packages.props` for central package management
   - `appsettings.json` and `appsettings.Development.json`
   - `global.json` pinning the .NET SDK

2. **Base stack installation** — install the chosen UI kit and core libraries:
   - Blazor: `MudBlazor`, `Radzen.Blazor`, or `Microsoft.FluentUI.AspNetCore.Components`
   - Angular: `primeng`, `@angular/material`, or `ng-zorro-antd`
   - Shared .NET: `Serilog`, `FluentValidation`, `AutoMapper`/`Mapperly`, `EF Core` or `Dapper`

3. **Agile directory structure** focused on code reuse:
   - `src/Web/Components` — reusable UI components from the chosen kit
   - `src/Core/` — domain logic, DTOs, interfaces
   - `src/Infrastructure/` — EF Core, external services, stubs
   - `tests/UnitTests/` — first test project ready to run

4. **Documentation folders** — create the project decision trail:
   - `docs/architecture/`
   - `docs/architecture/AD-0001-initial-stack.md`
   - `docs/adr/` (for future ADRs)
   - `.specs/`
   - `docs/specs/`

5. **Generate a lean README** documenting:
   - local run commands (`dotnet run`, `ng serve`)
   - adopted stack and architecture
   - how to add features (reuse first, follow `.claude/CONTEXT.md`)

6. **Run the build/type check** after each significant step. Fix any error before moving on.

### Phase 4 — Initial Architecture Decision Record

Create `docs/architecture/AD-0001-initial-stack.md` with the following content:

```markdown
# AD-0001 — Initial MVP Stack

## Context
[One-paragraph summary from Phase 1.]

## Decision
[Chosen stack: .NET version, Blazor or Angular, UI kit, persistence, DI, logging.]

## Consequences
- Positive: fast prototyping, proven component library, stable build.
- Trade-off: limited custom UI, dependency on the chosen kit.

## Related SPEC
- [.specs/SPEC-{YYYYMMDD}-{slug}.md](../.specs/SPEC-{YYYYMMDD}-{slug}.md)
```

### Phase 5 — External Service Stubs

For every external dependency (database, cache, queue, auth, object storage, payment), create a local, runnable stub:

- Use Docker Compose or a local in-memory implementation where possible.
- Provide a `.env.example` with all required keys and fake local values.
- Add a health-check script or endpoint.
- Never leave the app crashing on startup when a service is missing.

## Return Criteria

Before handing back to the orchestrator:

- [ ] Agent harness is installed (`.claude/`, `CLAUDE.md`, `AGENTS.md`).
- [ ] `.claude/CONTEXT.md` is updated with the chosen stack under **Technical Details**.
- [ ] `.claude/MEMORY.md` is created with the project summary.
- [ ] `docs/architecture/AD-0001-initial-stack.md` is created.
- [ ] `.specs/` and `docs/specs/` directories exist.
- [ ] `README.md` exists with run commands and a short architecture note.
- [ ] The build / type check passes cleanly.
- [ ] Lockfiles are updated and committed.
- [ ] No pseudo-code, `TODO`, or `// ...` escape comments remain in scaffold files.
- [ ] External dependencies have local stubs or mocks.
- [ ] Initial `git` history exists with at least one commit on `main`.

Then return control to the orchestrator reporting that the ground is ready for feature development.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Custom UI base components | Use MudBlazor, Radzen, Fluent UI, PrimeNG, Angular Material, or NG-ZORRO. |
| Skipping build checks | Run `dotnet build` / `ng build` after every structural step. |
| Leaving `// ...` comments | Every file must be self-contained and functional. |
| No local stubs for external services | Add Docker / in-memory / env-example stubs. |
| Proceeding without user stack approval | Ask and wait for explicit Portuguese confirmation. |
| Missing agent harness or docs folders | Install `/create-agent-harness` before application scaffolding. |

## References

- `create-agent-harness` — for installing the agent harness in the new repo
- `write-specs` — for producing the SPEC SDD that precedes this skill
- `create-issues` — for turning Epics into GitHub Issues
