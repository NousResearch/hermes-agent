# Framework Comparison

Angular, React, and Blazor each have strengths. Choose or align based on the project context, not only the team's preference.

## Angular

**Best for:** large enterprise apps, long-lived codebases, teams that value conventions.

- Strong CLI and code generation.
- Component and module structure is enforced.
- Material, PrimeNG, and NG-ZORRO are common UI libraries.
- RxJS is pervasive; design around async data and reactive state.
- Dependency injection shapes how services and state are structured.

## React

**Best for:** flexible UI, design systems, content-heavy apps, teams that want control.

- Unopinionated; requires explicit design discipline.
- Component ecosystem is large (Radix, Headless UI, Shadcn, Chakra, MUI).
- Hooks and context make state management flexible but also easy to misuse.
- CSS-in-JS, Tailwind, and CSS Modules are all common.
- Strong for progressive enhancement and micro-frontends.

## Blazor

**Best for:** .NET shops, line-of-business apps, teams that want one language end-to-end.

- Component model similar to Angular.
- MudBlazor, Fluent UI Blazor, Radzen are common UI libraries.
- WebAssembly or Server mode affects loading and latency expectations.
- C# and Razor syntax. JavaScript interop is possible but avoid heavy reliance.

## Decision Matrix

| Concern | Angular | React | Blazor |
|---------|---------|-------|--------|
| enterprise conventions | high | medium | medium |
| design-system flexibility | medium | high | low-medium |
| component ecosystem | large | very large | growing |
| team language | TypeScript | TypeScript/JS | C# |
| pre-rendered state | no | possible (SSR) | yes (Server) |
| mobile-first | yes | yes | yes |

## Recommendation

When the framework is unknown, design the system first and then map to the framework. Do not let a framework's defaults drive the design.
