# Maintenance Action Policy Validator

Status: implemented as a non-executing safety primitive.

Hermes includes a small maintenance-action policy validator in
`tools/maintenance_actions.py`. It is a static classifier, not an actuator. Its
job is to decide whether a proposed named maintenance action is blocked,
eligible but waiting for current-user approval, or statically approved by
policy.

It deliberately does not execute commands, load live `~/.hermes/config.yaml`,
run preflight probes, run postcheck probes, write audit logs, contact remote
hosts, or integrate with the terminal tool.

## Safety contract

The validator follows a default-deny contract:

- absent policy blocks;
- disabled global policy blocks;
- malformed global enabled flags block;
- missing or malformed `actions` blocks;
- malformed action IDs block;
- unknown actions block;
- disabled actions block;
- malformed action enabled flags block;
- malformed action bodies block;
- missing or malformed `exact_argv` blocks;
- malformed requested argv blocks;
- exact argv mismatch blocks;
- shell strings and shell-wrapper forms such as `sh -c`, `bash -lc`,
  `/usr/bin/env bash -lc`, and `/usr/bin/env -S ...` block;
- unattended contexts such as cron, background, scheduler, and unattended runs
  block by default;
- malformed unattended policy values block;
- missing preflight metadata blocks;
- malformed preflight metadata blocks;
- missing postcheck metadata blocks;
- malformed postcheck metadata blocks;
- malformed approval-requirement policy values block;
- malformed current-user approval values block;
- current-user approval is required by default even after all static gates pass.

The exact-argv rule is intentionally strict. A maintenance action policy names
one action and one concrete argv list. Aliases, wrapper scripts, shell strings,
interpolation, or equivalent-looking commands are not considered equivalent.

## Relationship to hardline and Tirith guards

This validator does not replace existing command guards. Hardline shutdown,
reboot, and similar protections remain authoritative. Tirith and the dangerous
command approval system remain authoritative.

Future integration must preserve that layering: maintenance-action policy can
only narrow what is allowed. It must not become a bypass around existing guard
layers.

## Dry-run classifier

`classify_maintenance_action(...)` wraps `evaluate_maintenance_action(...)` and
returns a JSON-serializable dict with stable keys:

- `allowed`
- `eligible`
- `reason`
- `action_id`
- `command_id`
- `host_label`

This is meant for future UI or tool layers that need a dry-run classification
surface. It is still inert and does not execute anything.

## Future execution work

Any future execution path must be a separate, reviewed, test-driven slice. At a
minimum it needs fresh tests for:

1. current-user approval capture;
2. live read-only preflight verification;
3. command execution through a non-shell exact argv path;
4. post-action verification;
5. audit logging;
6. interaction with existing hardline/Tirith approval layers;
7. unattended/cron failure-closed behavior.

Do not combine validator changes with execution integration in the same patch.
That separation is intentional: first build the interlock, then review whether
an actuator is justified.
