#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-NousResearch/hermes-agent}"

labels=(
  "lane/librarian|Research and evidence-gathering work for agents|5319E7"
  "lane/architect|Planning/specification work before implementation|7057FF"
  "lane/coder|Implementation-ready work for coding agents|0052CC"
  "lane/reviewer|Review, verification, and acceptance work|0E8A16"
  "lane/guardian|Audit, release-gate, safety, and security review|B60205"
  "status/researching|Evidence gathering is in progress|D876E3"
  "status/needs-plan|Needs an implementation plan or acceptance criteria|FBCA04"
  "status/ready-for-coder|Ready for Hero7, Copilot, or another coding agent|0E8A16"
  "status/in-progress|Agent or human is actively working this task|1D76DB"
  "status/pr-open|A pull request is open and linked|5319E7"
  "status/reviewing|Awaiting or undergoing review|D4C5F9"
  "status/changes-requested|Reviewer requested changes before acceptance|D93F0B"
  "status/ready-for-approval|Ready for explicit human approval/merge decision|0E8A16"
  "status/done|Accepted and complete|0E8A16"
  "agent/librarian|Assigned or suitable for Librarian research agent|C5DEF5"
  "agent/hero8|Assigned or suitable for Hero8 architect/reviewer|BFDADC"
  "agent/hero7|Assigned or suitable for Hero7 coding agent|C2E0C6"
  "agent/guardiancoo|Assigned or suitable for GuardianCOO audit agent|F9D0C4"
  "agent/copilot|Suitable for GitHub Copilot coding agent|EDEDED"
  "approval/required|Requires explicit approval before merge, release, deploy, or risky write|B60205"
  "production-impacting|Could affect production behavior, deployment, user data, or billing|D93F0B"
  "safety-sensitive|Touches safety, privacy, vulnerable users, or clinical/education-sensitive scope|B60205"
  "risk/low|Low implementation or operational risk|0E8A16"
  "risk/medium|Moderate implementation or operational risk|FBCA04"
  "risk/high|High risk; needs careful review and approval|B60205"
  "type/research|Research-only task or evidence brief|C5DEF5"
)

for entry in "${labels[@]}"; do
  IFS='|' read -r name desc color <<< "$entry"
  if gh label view "$name" --repo "$REPO" >/dev/null 2>&1; then
    gh label edit "$name" --repo "$REPO" --description "$desc" --color "$color"
  else
    gh label create "$name" --repo "$REPO" --description "$desc" --color "$color"
  fi
done
