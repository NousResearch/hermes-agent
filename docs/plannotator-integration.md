# Plannotator integration requirements

This document describes what Hermes expects from a Plannotator installation and
how operators configure the native `plannotator_session` tool.

## Goal

Hermes integrates with Plannotator so users can:
- review code diffs in a browser
- annotate markdown or text artifacts visually
- submit comments/replacements back to Hermes from the browser UI
- keep the feedback loop inline in chat when using `inline_review` or `inline_annotate`

## What Hermes ships

Hermes ships:
- the native `plannotator_session` tool
- the generic `service_expose` tool
- support for inline chat updates through `send_message`

Hermes does not ship:
- the Plannotator binary itself
- a reverse proxy / tunnel / hosting backend
- a mandatory launcher daemon or router implementation

Operators provide those pieces through command templates.

## Requirements

### 1. Plannotator must be installed

Hermes needs a working Plannotator executable or wrapper command that can be
invoked by the configured launcher template.

At minimum, the operator must be able to launch:
- a review flow
- an annotate flow
- optionally a prepare flow and/or last-message flow

Hermes does not require a specific installation method. For example, the
launcher template may call:
- a `plannotator` binary on `PATH`
- a virtualenv wrapper
- a script inside a checked-out repo
- a custom bridge script

### 2. Hermes must know how to launch Plannotator

Configure one or more of these environment variables in the Hermes runtime:
- `HERMES_PLANNOTATOR_PREPARE_TEMPLATE`
- `HERMES_PLANNOTATOR_REVIEW_TEMPLATE`
- `HERMES_PLANNOTATOR_ANNOTATE_TEMPLATE`
- `HERMES_PLANNOTATOR_LAST_TEMPLATE`

You can also override the launcher per tool call with `command_template`.

## Launcher contract

The configured launcher command should print structured lines that Hermes can
parse:
- `URL=...`
- `PID=...` (recommended)
- `LOG=...` (recommended)
- `HOST=...` (recommended when host reuse matters)

Minimum useful contract:
- `URL=...`

Recommended contract for inline flows:
- `HOST=...`
- `URL=...`
- `PID=...`
- `LOG=...`

Important detail for `PID=...`:
- when the launcher daemonizes or spawns Plannotator in the background, `PID` should refer to a supervisor process, not only the immediate launcher shell
- that supervisor should remain alive until feedback has been submitted, final log output has been flushed, and the local web server has been stopped
- in other words, Hermes should be waiting on a PID whose lifetime matches the review session lifecycle, not a short-lived bootstrap process

## Requirements for inline flows

`inline_review` and `inline_annotate` work best when the launcher supports two
capabilities:

1. Prepare/reserve a host before the full session is launched
2. Launch a session pinned to a fixed host

Hermes uses that to:
1. reserve a URL
2. post the URL immediately in chat
3. launch Plannotator on the same host
4. wait for the submitted feedback

A launcher can support fixed-host reuse via an environment variable, flag, or
other operator-defined contract. Hermes passes `fixed_host` through the launcher
environment as `PLANNOTATOR_HOST`.

For remote/shared review flows, the launcher should also set:
- `PLANNOTATOR_REMOTE=1`
- `PLANNOTATOR_PORT=<fixed-port>`

That prevents local browser auto-open behavior and makes the forwarded/shared
port predictable.

## Exposure configuration

If Plannotator is not directly reachable, operators may expose it through:
- `reverse-proxy`
- `tailscale-serve`
- `tailscale-funnel`
- another custom command-template strategy

For browser UX, prefer a stable host per chat/session/workspace and distinguish
individual reviews by path, token, or session state when possible. Reusing the
same host helps preserve browser-side settings such as cookies.

The `service_expose` tool supports:
- `HERMES_SERVICE_EXPOSE_REVERSE_PROXY_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_SERVE_TEMPLATE`
- `HERMES_SERVICE_EXPOSE_TAILSCALE_FUNNEL_TEMPLATE`

## Example launcher templates

These are examples only. Operators should replace them with commands appropriate
for their environment.

Review:
```bash
export HERMES_PLANNOTATOR_REVIEW_TEMPLATE='plannotator-bridge review {review_target_arg}'
```

Annotate:
```bash
export HERMES_PLANNOTATOR_ANNOTATE_TEMPLATE='plannotator-bridge annotate {artifact_path}'
```

Prepare:
```bash
export HERMES_PLANNOTATOR_PREPARE_TEMPLATE='plannotator-bridge prepare'
```

## Operational expectations

A working setup should allow Hermes to:
- open a browser review URL for a diff or artifact
- return that URL to the user
- observe session completion from PID and/or log output
- summarize the returned review result

## Troubleshooting checklist

If the tool is configured but not working:
- verify the launcher command runs successfully outside Hermes
- verify the launcher prints `URL=...`
- verify `PID=` and `LOG=` are correct if synchronous waiting is expected
- verify the configured exposure backend actually publishes the requested host
- verify the launched process exits when feedback is submitted
