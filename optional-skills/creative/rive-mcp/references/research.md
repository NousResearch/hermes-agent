# Sources and evidence boundaries

Reviewed 2026-09-26. Prefer current official documentation and installed schemas;
a dated count or version below is not a permanent invariant. This is original
procedural guidance, not a mirror of vendor documentation.

## Authoritative sources

| Area | Source |
| --- | --- |
| Rive documentation index | https://rive.app/docs/llms.txt |
| Official local CLI | https://rive.app/docs/cli/overview.md |
| Installation, offline builds and signing | https://rive.app/docs/cli/getting-started.md |
| CLI commands and exit codes | https://rive.app/docs/cli/reference/commands.md |
| Asset discovery and push/pull | https://rive.app/docs/cli/reference/project-config.md |
| RML units and schema lookup | https://rive.app/docs/runtimes/advanced-topic/rml.md |
| Agent workflow | https://rive.app/docs/cli/agents.md |
| Official desktop MCP | https://rive.app/docs/editor/ai/mcp.md |
| Runtime export | https://rive.app/docs/editor/exporting/exporting-for-runtime.md |
| Editable backup | https://rive.app/docs/editor/exporting/exporting-for-backup.md |
| Pricing and plan terms | https://rive.app/pricing |
| Runtime feature matrix | https://rive.app/docs/feature-support.md |
| Web renderer selection | https://rive.app/docs/runtimes/web/canvas-vs-webgl.md |
| Runtime constructor/lifecycle | https://rive.app/docs/runtimes/web/rive-parameters.md |
| Typed bindings | https://rive.app/docs/runtimes/web/data-binding.md |
| Experimental Web GPU Canvas | https://rive.app/docs/runtimes/web/gpu-canvas.md |
| Experimental React GPU Canvas | https://rive.app/docs/runtimes/react/gpu-canvas.md |
| Experimental Web semantics | https://rive.app/docs/runtimes/web/semantics.md |
| Authored reduced motion | https://rive.app/docs/editor/accessibility/reduced-motion.md |
| Scripting and tests | https://rive.app/docs/scripting/getting-started.md |
| WGSL | https://rive.app/docs/scripting/wgsl-shaders.md |
| Hermes MCP | https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp |
| Hermes skills | https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills |

## Changed guidance

- **An official editor-free path now exists.** CLI/RML compiles local text projects,
  inspects them and renders screenshots. Do not confuse it with the custom
  `scripts/rive_mcp.py` adapter or the third-party RiveMCP product.
- **Export entitlement differs by route and output.** The 0.9.9 desktop MCP's
  `export_file` schema says runtime `.riv` works on every plan and editable `.rev`
  may be refused. The public editor runtime-export page still says paid plans;
  pricing advertises runtime export under Cadet. CLI docs separately allow local
  unsigned builds without login, require login for `--publish`/`--rev`, and tie
  watermark-free publishing to a linked file in a Cadet-or-higher workspace.
  Record this conflict rather than declaring universal free or paid export.
  Respect actual permissions; permitted runtime output is not editable source.
- **GPU Canvas is available experimentally on WebGL2**, not Canvas2D. Check the
  opt-in flag, per-canvas context ownership, and incompatible offscreen sharing.
- **MCP visual review now has `capture_artboard`.** Discovery is not execution
  evidence and its annotation is not the helper's safety policy.
- Native MCP discovery and supported reload are distinct from a fresh connection
  probe. Use Tool Slimmer expansion if installed and a required tool is hidden.

## Evidence boundaries

A macOS desktop 0.9.9/build5998 read-only probe discovered 40 tools at the
loopback endpoint and successfully called `session_info`. No document was open:
`activeFileId` was null. This establishes connection and schema discovery, **not**
editor mutations, capture, export entitlement, script execution, or source reopening.

The helper's automated tests exercise argument/schema/file guards and SDK
transport behavior against synthetic servers, not live editor acceptance.
The official-CLI reference records its separate local-source checks. Development
runtime-fixture evidence is not original artwork or native/game/device coverage.

## Third-party alternatives and lineage

The earlier proposal documented `paradoxsyn/rivemcp-releases`, a separately
licensed third-party headless product. It is not required by either official route
and has not been exercised for this contribution. Do not install it implicitly,
copy old free-export quotas as current facts, or use it to bypass an export denial.

The original proposal is preserved by commit ancestry and credited in `NOTICE.md`.
This package contributes no Hermes core tools, dependency changes, service,
self-updater, telemetry, private workbench data, or mandatory ProDyn design system.
