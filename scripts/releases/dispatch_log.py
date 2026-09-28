"""Print the release.py command and receipt identity for this bundle run.

The print is a log. It must not fail the build, and it must not invent a
command the dispatcher does not send. A commit or channel run also gets a
post-build receipt tag. This step runs before that tag exists, so it prints
the tag shape and names the step that writes the real tag.
"""
from __future__ import annotations

import json
import shlex
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
    channel = _text(env.get("CHANNEL"))
    disposable = _text(env.get("DISPOSABLE_CHANNEL"))
    phase = _text(env.get("RELEASE_PHASE"))
    if channel or disposable:
        kind = "channel"
    elif commit:
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
        "channel": channel,
        "disposable_channel": disposable,
        "release_phase": phase,
        "upload_release": _flag(env.get("UPLOAD_RELEASE")),
        "jobs": _text(env.get("JOBS")),
        "all_jobs": _all_jobs(env.get("JOBS")),
        "termux_upgrade_from_tag": _text(env.get("TERMUX_UPGRADE_FROM_TAG")),
        "disposable_receivers": _flag(env.get("DISPOSABLE_RECEIVERS")),
        "disposable_run": _text(env.get("R2_DISPOSABLE_RUN")),
        "bundle_env": bundle_env(env.get("BUNDLE_ENV_JSON")),
    }


def release_command(env: Mapping[str, str]) -> list[str] | None:
    """The gh workflow command for a dispatch release.py can recreate.

    Stable and canary dispatches mint a tag before they start a workflow, so
    no release.py command recreates those runs. Return None for them.
    """
    facts = describe(env)
    repository = facts["repository"]
    branch = _text(env.get("DEFAULT_BRANCH"))
    if not repository or not branch or facts["kind"] not in {"commit", "channel"}:
        return None
    if facts["disposable_channel"] or facts["disposable_run"] or facts["disposable_receivers"]:
        return None
    if not facts["all_jobs"] or facts["termux_upgrade_from_tag"] or facts["upload_release"]:
        return None
    commit = facts["build_commit"]
    if not isinstance(commit, str) or not commit:
        return None
    baked = facts["bundle_env"]
    if not isinstance(baked, dict):
        return None
    if facts["kind"] == "channel":
        from scripts.releases.channel_build import dispatch_command

        return dispatch_command(str(facts["channel"]), commit, str(repository), branch, baked or None)
    from scripts.releases.commit_build import dispatch_command

    return dispatch_command(commit, str(repository), branch, baked or None)


def receipt_shape(facts: Mapping[str, object]) -> str | None:
    """The post-build receipt tag shape for a commit or channel run.

    The real tag needs the run's creation time, which this step does not have.
    Show the shape from receipt_tag so the log and the publisher agree.
    """
    kind = facts["kind"]
    run_id = facts["run_id"]
    if kind not in {"commit", "channel"} or not isinstance(run_id, str) or not run_id:
        return None
    if facts["disposable_run"]:
        return None
    from scripts.releases.commit_build import receipt_tag

    return receipt_tag(str(kind), "0.0.0", "2000-01-01T00:00:00Z", run_id).replace(
        "v0.0.0", "v<version>", 1).replace("20000101T000000Z", "<run-created-utc>", 1)


def report(env: Mapping[str, str]) -> str:
    """One log block: the mode, the replay command, the receipt, and the facts."""
    facts = describe(env)
    lines = [
        "::group::Bundle dispatch",
        "kind: " + str(facts["kind"]),
    ]
    command = release_command(env)
    if command is None:
        lines.append("release.py: this dispatch is not a release.py --build-commit run")
    else:
        lines.append("release.py: " + shlex.join(["python", "scripts/release.py", "--publish", "--remote", "<remote>", *command_flags(facts)]))
        lines.append("workflow: " + shlex.join(command))
    shape = receipt_shape(facts)
    if shape is None:
        lines.append("receipt: this dispatch writes no post-build receipt tag")
    else:
        lines.append("receipt: " + shape)
        lines.append("receipt writer: publish step, after the native jobs succeed")
    lines.append("facts: " + json.dumps(facts, sort_keys=True))
    lines.append("::endgroup::")
    return "\n".join(lines)


def command_flags(facts: Mapping[str, object]) -> list[str]:
    """The release.py flags that name this dispatch, in dispatcher order."""
    flags = ["--build-commit", str(facts["build_commit"])]
    if facts["kind"] == "channel":
        flags += ["--channel", str(facts["channel"])]
    baked = facts["bundle_env"]
    if isinstance(baked, dict):
        for name in sorted(baked):
            value = baked[name]
            if value is None:
                flags += ["--bundle-unset", name]
            else:
                flags += ["--bundle-env", name + "=" + value]
    return flags


def main() -> None:
    import os

    print(report(os.environ))


if __name__ == "__main__":
    main()
