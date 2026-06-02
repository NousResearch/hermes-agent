---
name: vision-mcp
description: Reuse desktop GUI workflows with Vision-MCP maps.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, windows]
metadata:
  hermes:
    tags: [MCP, Desktop, GUI, Vision, Automation]
    homepage: https://github.com/Haruhiyuki/vision-mcp
    related_skills: [native-mcp, mcporter]
prerequisites:
  commands: [node, npx]
---

# Vision-MCP

Use Vision-MCP to operate visible desktop applications through reusable maps, workflows, screenshots, accessibility trees, OCR, and guarded GUI actions.

## When to Use

Use this skill when the user asks Hermes to:

- run a repeatable workflow in a macOS or Windows desktop application
- inspect an app window visually before deciding the next action
- recover when a saved GUI workflow drifts because labels, positions, or states changed
- turn a successful desktop procedure into a reusable Vision-MCP app map

Use a normal CLI, API, or web integration instead when the same task can be completed without controlling a real desktop session.

## Prerequisites

Install the optional MCP catalog entry and start a new Hermes session:

```bash
hermes mcp install vision-mcp
```

Check the Vision-MCP native helper and platform readiness:

```bash
npx -y @vision-mcp/cli@latest doctor
npx -y @vision-mcp/cli@latest install-helper
```

Copy starter app maps into the default apps root:

```bash
npx -y @vision-mcp/cli@latest init-apps
```

On macOS, grant Screen Recording and Accessibility permissions to the Vision-MCP helper or the terminal process that launches Hermes. On Windows, use PowerShell 5.1 or newer and run Hermes elevated when controlling elevated target applications.

## How to Run

After `hermes mcp install vision-mcp`, Hermes registers the server's tools with the `mcp_vision_mcp_` prefix. If the server is installed under a different MCP name, replace that prefix with `mcp_<server_name>_`, with hyphens and dots changed to underscores.

Useful starting tools:

- `mcp_vision_mcp_vision_map_list_apps`
- `mcp_vision_mcp_vision_map_list_workflows`
- `mcp_vision_mcp_vision_map_describe_workflow`
- `mcp_vision_mcp_vision_map_run_workflow`
- `mcp_vision_mcp_vision_map_snapshot`
- `mcp_vision_mcp_vision_map_detect_state`
- `mcp_vision_mcp_vision_map_repair_minimal`
- `mcp_vision_mcp_capsule_attach_window`
- `mcp_vision_mcp_capsule_validate_geometry`

## Quick Reference

| Goal | Preferred tool |
| --- | --- |
| Confirm setup | `mcp_vision_mcp_vision_map_list_apps` |
| Find reusable automation | `mcp_vision_mcp_vision_map_list_workflows` |
| Inspect workflow contract | `mcp_vision_mcp_vision_map_describe_workflow` |
| Execute saved workflow | `mcp_vision_mcp_vision_map_run_workflow` |
| Inspect current window state | `mcp_vision_mcp_vision_map_snapshot` |
| Identify current state | `mcp_vision_mcp_vision_map_detect_state` |
| Recover from drift | `mcp_vision_mcp_vision_map_repair_minimal` |
| Export debugging evidence | `mcp_vision_mcp_vision_map_export_trace` |

## Procedure

1. Confirm the MCP server is loaded with `mcp_vision_mcp_vision_map_list_apps`. If the tool is unavailable, ask the user to run `hermes mcp install vision-mcp` and restart Hermes.
2. Prefer saved workflows over raw GUI input. List apps, list workflows, inspect the chosen workflow, then run it with explicit inputs.
3. Establish or repair the visible window only when needed. Use capsule tools to attach, raise, migrate, or validate geometry before taking screenshots or running actions.
4. Use `mcp_vision_mcp_vision_map_snapshot` for unknown states, failed postconditions, or user-visible confirmation. Avoid repeated screenshots when workflow and state tools already provide enough context.
5. When a known control or postcondition fails, call `mcp_vision_mcp_vision_map_repair_minimal` before escalating to manual coordinates or fresh exploration.
6. For map-building or low-level control work, get explicit user consent before enabling or using tools such as `vision_map.click_at`, `vision_map.type_text`, `vision_map.press_key`, `vision_map.scroll`, `vision_map.init`, `vision_map.apply_patch`, `vision_map.add_control`, `vision_map.commit_state`, `vision_map.commit_workflow`, or `vision_map.harvest_session`.
7. Verify the result through the workflow's postcondition, `mcp_vision_mcp_vision_map_verify`, a final state check, or a user-visible screenshot when the task outcome depends on what is on screen.

## Pitfalls

- Vision-MCP controls real visible applications. Do not bypass logins, two-factor prompts, CAPTCHAs, consent dialogs, or destructive confirmations.
- The default Hermes catalog entry does not enable raw coordinate/input tools or map mutation tools. Use `hermes mcp configure vision-mcp` only when the user explicitly wants that capability.
- Window geometry matters. A hidden, minimized, off-screen, or partially covered target can make screenshots, OCR, and control locators unreliable.
- The first `npx` run may download the npm package, so setup and the first MCP probe can take longer than later sessions.
- Browser and Electron apps often need OCR or bounding-box locators even when accessibility data is incomplete.

## Verification

Run the catalog connection check:

```bash
hermes mcp test vision-mcp
```

Then start a new Hermes session and call `mcp_vision_mcp_vision_map_list_apps`. For platform diagnostics, run:

```bash
npx -y @vision-mcp/cli@latest doctor
```
