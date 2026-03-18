---
name: autoresearch
description: Run bounded, manifest-driven research cycles against a workspace-defined project. Inspect AutoResearch manifests, validate families, execute research_cycle runs, inspect reports, and prepare publishable summaries through the bundled helper script.
version: 1.0.0
author: Nous Research
license: MIT
metadata:
  hermes:
    tags: [research, experimentation, benchmarking, evaluator, manifests, param-mutation, agent-patch]
    category: research
    requires_toolsets: [terminal, file]
---

# AutoResearch

This optional skill keeps AutoResearch out of the core toolset while preserving the bounded research workflow.

Use it when the user wants Hermes to work with a project that defines:

- `.hermes/autoresearch/project.yaml`
- `.hermes/autoresearch/families/*.yaml`

The skill ships with a helper script at:

```text
~/.hermes/skills/research/autoresearch/scripts/autoresearch.py
```

Run it with Python and parse the JSON it prints.

## When to Use

- the workspace already has AutoResearch manifests
- the user wants to inspect projects, validate manifests, or run a bounded research cycle
- the user wants to inspect previous runs or prepare a publishable summary
- the user wants to set up a new AutoResearch manifest and needs the required shape

## Safety Rules

Always inspect the manifests before running a research cycle. The evaluator and validation commands come from the workspace and will be executed directly.

Recommended sequence:

1. Read `.hermes/autoresearch/project.yaml` and the target family manifest.
2. Run `inspect-project` or `validate-project`.
3. Confirm the evaluator commands match the user's intent.
4. Only then run `research-cycle`.

For `agent_patch` families, confirm the editable files and marker bounds before running.

## Command Reference

Use these commands from the project root unless the user gives a different `--project-root`.

```bash
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py list-projects
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py inspect-project
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py validate-project
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py research-cycle --family-id threshold-sweep
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py status --run-id ar-...
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py list-runs
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py inspect-run --run-id ar-...
python ~/.hermes/skills/research/autoresearch/scripts/autoresearch.py publish-summary --run-id ar-...
```

Useful flags:

- `--project-root /path/to/project`
- `--population N`
- `--survivors N`
- `--seed N`
- `--model provider/model` for `agent_patch`
- `--target telegram` or another Hermes messaging target for `publish-summary`
- `--send` to send the prepared summary immediately

## Workflow

Follow this order unless the user asks for something narrower:

1. Inspect the manifests with `read_file`.
2. Run `inspect-project`.
3. Run `validate-project`.
4. Run `research-cycle`.
5. Run `inspect-run`.
6. Run `publish-summary` if the user wants a digest or delivery target.

Summarize the JSON results in plain language after each step instead of dumping raw output unless the user asks for the raw payload.

## Manifest Checklist

Project manifest:

- `project_id`
- `description`
- `default_cwd`
- evaluator commands plus `result_json`
- `report_output_dir`
- optional `publish_target`

Family manifest:

- `family_id`
- `thesis`
- `commands.validation` and `commands.evaluation`
- `mutation.mode` (`param_mutation` or `agent_patch`)
- `selection.primary_metric`
- `interesting_if`
- `anchors`
- `editable_files` for `agent_patch`
- `editable_markers` when edits must stay inside explicit bounds

## Notes

- `research-cycle` creates candidate workspaces under `.hermes/autoresearch/workspaces/...`
- interesting runs write reports under the configured report output directory
- `publish-summary --send` uses Hermes messaging if a target is configured
- if the workspace is missing manifests, help the user author them with the file tools before retrying
