#!/usr/bin/env python3
"""Verify docker-publish.yml digest output wiring for cosign signing.

Checks:
1. merge job declares a 'digest' output
2. merge job has a 'Read manifest digest' step with id 'manifest_digest'
3. merge's digest output references steps.manifest_digest.outputs.digest
4. sign job exists and needs: merge
5. sign job reads needs.merge.outputs.digest
6. sign job has id-token: write permission for keyless OIDC signing
7. sign job uses cosign-installer action
8. sign job runs 'cosign sign' with the digest reference
"""

import sys
import yaml


def load_workflow(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check(wf, label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main():
    wf = load_workflow(".github/workflows/docker-publish.yml")
    jobs = wf["jobs"]
    all_pass = True

    # 1. merge job exists
    assert "merge" in jobs, "merge job missing"
    merge = jobs["merge"]

    # 2. merge declares digest output
    outputs = merge.get("outputs", {})
    all_pass &= check(wf, "merge declares 'digest' output", "digest" in outputs)

    # 3. digest output references manifest_digest step
    digest_expr = outputs.get("digest", "")
    all_pass &= check(
        wf,
        "digest output references manifest_digest step",
        "manifest_digest" in str(digest_expr),
    )

    # 4. merge has 'Read manifest digest' step
    steps = merge.get("steps", [])
    digest_step = next(
        (s for s in steps if s.get("id") == "manifest_digest"), None
    )
    all_pass &= check(wf, "merge has manifest_digest step", digest_step is not None)

    # 5. manifest_digest step uses imagetools inspect
    if digest_step:
        run = digest_step.get("run", "")
        all_pass &= check(
            wf,
            "manifest_digest step uses 'imagetools inspect'",
            "imagetools inspect" in run,
        )
        all_pass &= check(
            wf,
            "manifest_digest step writes to GITHUB_OUTPUT",
            "GITHUB_OUTPUT" in run,
        )

    # 6. sign job exists
    assert "sign" in jobs, "sign job missing"
    sign = jobs["sign"]

    # 7. sign job needs merge
    needs = sign.get("needs", [])
    all_pass &= check(wf, "sign needs merge", "merge" in needs)

    # 8. sign job has id-token: write
    perms = sign.get("permissions", {})
    all_pass &= check(
        wf, "sign has id-token: write", perms.get("id-token") == "write"
    )

    # 9. sign job installs cosign
    sign_steps = sign.get("steps", [])
    cosign_install = next(
        (s for s in sign_steps if "cosign-installer" in str(s.get("uses", ""))),
        None,
    )
    all_pass &= check(wf, "sign installs cosign", cosign_install is not None)

    # 10. sign step reads needs.merge.outputs.digest
    sign_step = next(
        (s for s in sign_steps if "cosign sign" in s.get("run", "")), None
    )
    all_pass &= check(wf, "sign has 'cosign sign' step", sign_step is not None)

    if sign_step:
        env = sign_step.get("env", {})
        digest_env = env.get("DIGEST", "")
        all_pass &= check(
            wf,
            "sign step DIGEST reads from merge output",
            "needs.merge.outputs.digest" in str(digest_env),
        )
        run = sign_step.get("run", "")
        all_pass &= check(
            wf,
            "sign step references DIGEST in run",
            "${DIGEST}" in run or "${image}" in run,
        )

    # Summary
    print()
    if all_pass:
        print("ALL CHECKS PASSED — digest output wiring is correct.")
        return 0
    else:
        print("SOME CHECKS FAILED — see above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
