#!/usr/bin/env python3
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "track-upstream.yml"
SYNC_TOKEN_EXPR = "${{ secrets.UPSTREAM_SYNC_TOKEN || github.token }}"


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise SystemExit(f"{WORKFLOW}: duplicate key {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_workflow():
    with WORKFLOW.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=UniqueKeyLoader)


def main() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow = load_workflow()
    steps = workflow["jobs"]["rebase-onto-upstream"]["steps"]

    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    if checkout["with"]["token"] != SYNC_TOKEN_EXPR:
        raise SystemExit("track-upstream checkout must use UPSTREAM_SYNC_TOKEN fallback")

    gh_token_envs = [
        step["env"]["GH_TOKEN"]
        for step in steps
        if isinstance(step.get("env"), dict) and "GH_TOKEN" in step["env"]
    ]
    if not gh_token_envs:
        raise SystemExit("track-upstream must pass GH_TOKEN to GitHub CLI steps")
    if any(token != SYNC_TOKEN_EXPR for token in gh_token_envs):
        raise SystemExit("track-upstream GH_TOKEN envs must use UPSTREAM_SYNC_TOKEN fallback")
    if 'git push origin "$BRANCH" 2>/tmp/push_err' not in workflow_text:
        raise SystemExit("track-upstream must capture conflict-branch push failures")
    if "Could not push conflict branch" not in workflow_text:
        raise SystemExit("track-upstream must explain conflict-branch push failures")


if __name__ == "__main__":
    main()
