#!/usr/bin/env python3
"""Emit review_status JSON for the review-labels workflow.

Builds a JSON array with one entry::

    [
      {
        "source": "review-label-gate",
        "results": [
          {"kind": "action_required", "title": "...", "summary": "...",
           "how_to_fix": "..."},
          {"kind": "info", "title": "...", "summary": "..."}
        ]
      }
    ]

The ``source`` field is the workflow name that declared the status; the
assembler uses it to exclude the corresponding job from the synthesized
❌ Error list (the job already has its own status section).

The array can contain 0 to 3 results — one per lane that ran
(``ci_review``, ``mcp_catalog``, ``supply_chain``). When the ``ci-reviewed`` label is
present, the kind is ``info``; when missing, it's ``action_required``
with the verification checklist.
"""

from __future__ import annotations

import argparse
import json
import sys

# The source identifier used for error-synthesis exclusion. This must
# match (as a normalized substring) the job name as it appears in the
# GitHub Actions API. The ci.yml job key is ``review-labels`` with
# ``name: Review label gate``, and the reusable workflow's job is also
# ``name: Review label gate``, so the API shows the job as
# "Review label gate / Review label gate". Normalizing "review-label-gate"
# (lowercase, hyphens→spaces) gives "review label gate", which is a
# substring of "review label gate / review label gate".
SOURCE = "review-label-gate"


def build_results(
    ci_review: bool,
    mcp_catalog: bool,
    supply_chain: bool,
    label_present: bool,
) -> list[dict]:
    """Build the list of result objects for this source."""
    results: list[dict] = []

    if ci_review:
        if label_present:
            results.append({
                "kind": "info",
                "title": "CI-sensitive file review",
                "summary": "`ci-reviewed` label is present.",
            })
        else:
            results.append({
                "kind": "action_required",
                "title": "CI-sensitive file review",
                "summary": (
                    "This PR changes CI-sensitive files (eslint config, "
                    "workflow YAMLs, or composite actions). These influence "
                    "what the js-autofix job executes and pushes to main."
                ),
                "how_to_fix": (
                    "Add the `ci-reviewed` label after verifying:\n"
                    "- no new eslint rules with custom `fix` functions that write outside linted paths,\n"
                    "- no workflow changes that widen permissions or remove guards,\n"
                    "- no composite action changes that alter what gets executed."
                ),
            })

    if mcp_catalog:
        if label_present:
            results.append({
                "kind": "info",
                "title": "MCP catalog security review",
                "summary": "`ci-reviewed` label is present.",
            })
        else:
            results.append({
                "kind": "action_required",
                "title": "MCP catalog security review",
                "summary": (
                    "This PR changes the bundled MCP catalog or MCP catalog "
                    "installer code. MCP entries can define local commands "
                    "that users later install into `mcp_servers`, so this "
                    "needs explicit maintainer review before merge."
                ),
                "how_to_fix": (
                    "Add the `ci-reviewed` label after verifying:\n"
                    "- any new/changed `optional-mcps/**/manifest.yaml` command and args are expected,\n"
                    "- stdio transports do not use shell+egress/exfiltration payloads,\n"
                    "- git install refs are pinned and bootstrap commands are minimal,\n"
                    "- requested env vars/secrets match the upstream MCP's documented needs."
                ),
            })

    if supply_chain and not label_present:
        results.append({
            "kind": "action_required",
            "title": "Critical supply chain risk",
            "summary": "Critical supply chain risk patterns were detected in this PR.",
            "how_to_fix": (
                "Review the flagged code carefully. If it is intentional, add the "
                "`ci-reviewed` label to confirm maintainer review."
            ),
        })

    return results


def build_statuses(
    ci_review: bool,
    mcp_catalog: bool,
    supply_chain: bool,
    label_present: bool,
) -> list[dict]:
    """Build the full review_status array (one entry with a results list)."""
    results = build_results(ci_review, mcp_catalog, supply_chain, label_present)
    if not results:
        return []
    return [{"source": SOURCE, "results": results}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-review", action="store_true",
                        help="Whether CI-sensitive files changed.")
    parser.add_argument("--mcp-catalog", action="store_true",
                        help="Whether the MCP catalog / installer changed.")
    parser.add_argument("--supply-chain", action="store_true",
                        help="Whether the critical supply-chain scanner found a risk.")
    parser.add_argument("--label-present", action="store_true",
                        help="Whether the ci-reviewed label is present.")
    parser.add_argument("--output", default="-",
                        help="Output file ('-' for stdout, or a GITHUB_OUTPUT path).")
    args = parser.parse_args()

    statuses = build_statuses(
        args.ci_review, args.mcp_catalog, args.supply_chain, args.label_present
    )
    json_str = json.dumps(statuses)

    if args.output == "-":
        print(json_str)
    else:
        # GITHUB_OUTPUT format: key=value\n
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(f"review_status={json_str}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
