"""Log what kind of bundle run this dispatch is.

The print is a log: it must not fail the build. Every release.py build (stable, canary, commit and
channel) claims a tag and runs Stable Release, so no release.py command recreates a direct dispatch
of this workflow, and no run of it writes a post-build receipt. A tagless commit dispatch is the
legacy mode.
"""
from __future__ import annotations

import json
from collections.abc import Mapping


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _flag(value: object) -> bool:
    return str(value).lower() == "true"


def bundle_env(raw: object) -> dict[str, str | None]:
    """The dispatcher's own bundle-env object, or empty when it is absent."""
    text = _text(raw).strip()
    if not text or text == "{}":
        return {}
    from scripts.releases.bundle_env import decode

    return decode(text)


def _all_jobs(raw: object) -> bool:
    """True when the dispatch builds every job group; a subset has no release.py replay.

    An unreadable value is reported, not raised: this log must not fail the
    build, and the admission step that follows refuses the input.
    """
    from scripts.releases.job_groups import selects_all

    try:
        return selects_all(raw if isinstance(raw, str) else None)
    except ValueError:
        return False


def describe(env: Mapping[str, str]) -> dict[str, object]:
    """Classify one workflow dispatch without reading git or the network."""
    tag = _text(env.get("TAG"))
    commit = _text(env.get("BUILD_COMMIT"))
    phase = _text(env.get("RELEASE_PHASE"))
    if commit:
        kind = "commit"
    elif phase == "candidate":
        kind = "stable-candidate"
    elif phase in {"publish", "promote"}:
        kind = "stable-" + phase
    elif "-canary." in tag:
        kind = "canary"
    elif tag:
        kind = "stable"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "repository": _text(env.get("GITHUB_REPOSITORY")),
        "ref": _text(env.get("GITHUB_REF")),
        "sha": _text(env.get("GITHUB_SHA")),
        "run_id": _text(env.get("GITHUB_RUN_ID")),
        "run_attempt": _text(env.get("GITHUB_RUN_ATTEMPT")),
        "actor": _text(env.get("GITHUB_ACTOR")),
        "tag": tag,
        "build_commit": commit,
        "release_phase": phase,
        "upload_release": _flag(env.get("UPLOAD_RELEASE")),
        "jobs": _text(env.get("JOBS")),
        "all_jobs": _all_jobs(env.get("JOBS")),
        "termux_upgrade_from_tag": _text(env.get("TERMUX_UPGRADE_FROM_TAG")),
        "bundle_env": bundle_env(env.get("BUNDLE_ENV_JSON")),
    }


def report(env: Mapping[str, str]) -> str:
    """One log block: the mode and the facts."""
    facts = describe(env)
    lines = [
        "::group::Bundle dispatch",
        "kind: " + str(facts["kind"]),
    ]
    if facts["kind"] == "commit":
        lines.append("release.py: a legacy tagless commit build; release.py --build-commit now claims "
                     "commit.<stamp>-v<version> and runs Stable Release")
    else:
        lines.append("release.py: this dispatch is not a release.py --build-commit run")
    lines.append("receipt: this dispatch writes no post-build receipt tag")
    lines.append("facts: " + json.dumps(facts, sort_keys=True))
    lines.append("::endgroup::")
    return "\n".join(lines)


def main() -> None:
    import os

    print(report(os.environ))


if __name__ == "__main__":
    main()
