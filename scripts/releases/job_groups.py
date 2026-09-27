"""The desktop-bundled-release `jobs` input: one parser for the job groups.

The workflow's admission step is the only place that reads the raw input; it
emits one `<group>=true|false` output per group and every job gates on its own
group output. Channel callers use `selects_all` where they refused
`TERMUX_ONLY=true` before, and the stable phase result uses `phase_jobs` to
judge only the selected groups.
"""
from __future__ import annotations

import os

JOB_GROUPS = ("darwin-arm64", "darwin-x64", "win32-arm64", "win32-x64",
              "win32-bundle", "linux-x64", "linux-arm64", "termux")
ALL_JOBS = ",".join(JOB_GROUPS)

# The workflow jobs each group runs, in the names desktop-bundled-release.yml
# declares. `phase_jobs` turns a selection into the list the stable phase
# result requires.
GROUP_JOBS = {
    "darwin-arm64": ("build-darwin-arm64",),
    "darwin-x64": ("build-darwin-x64",),
    "win32-arm64": ("build-win32-arm64",),
    "win32-x64": ("build-win32-x64",),
    "win32-bundle": ("assemble-win32-bundle",),
    "termux": ("termux-deb",),
}
# The native smoke each group runs after its build. A claim that skipped
# tests runs none of them.
GROUP_SMOKES = {
    "darwin-arm64": ("smoke-darwin-arm64",),
    "darwin-x64": ("smoke-darwin-x64",),
    "win32-arm64": ("smoke-win32-arm64",),
    "win32-x64": ("smoke-win32-x64",),
}


def parse_jobs(raw: str | None) -> dict[str, bool]:
    """Select job groups from a comma-separated list. ``None`` is the default.

    An unknown name, a duplicate, an empty entry and an explicitly empty list
    are refused; admission then fails before any build starts.
    """
    if raw is None:
        return {group: True for group in JOB_GROUPS}
    names = [name.strip() for name in raw.split(",")]
    if not any(names):
        raise ValueError(f"jobs must name at least one group: {ALL_JOBS}")
    selected = {group: False for group in JOB_GROUPS}
    for name in names:
        if not name:
            raise ValueError("jobs contains an empty group name")
        if name not in selected:
            raise ValueError(f"unknown job group: {name}")
        if selected[name]:
            raise ValueError(f"duplicate job group: {name}")
        selected[name] = True
    return selected


def selects_all(raw: str | None) -> bool:
    """True only when the input selects every group."""
    return all(parse_jobs(raw).values())


def phase_jobs(selected: dict[str, bool], phase: str, *, skip_tests: bool = False) -> list[str]:
    """Jobs the stable phase result judges: an unselected group never fails it.

    B4 moved candidate-manifest into stable-release.yml, so the desktop
    workflow no longer owns it and the phase result never names it. A claim
    that skipped tests runs no smoke, so the smokes are not judged.
    """
    required = ["validate"]
    if phase == "publish":
        return required + ["stable-publish", "stable-store"]
    if phase != "candidate":
        raise ValueError(f"Unknown release phase: {phase}")
    for group, jobs in GROUP_JOBS.items():
        if selected.get(group):
            required.extend(jobs)
            if not skip_tests:
                required.extend(GROUP_SMOKES.get(group, ()))
    return required


def main() -> None:
    """Emit one `<group>=true|false` line per group, and `all-jobs`, for `$GITHUB_OUTPUT`."""
    selected = parse_jobs(os.environ.get("JOBS"))
    for group in JOB_GROUPS:
        print(f"{group}={'true' if selected[group] else 'false'}")
    print(f"all-jobs={'true' if all(selected.values()) else 'false'}")


if __name__ == "__main__":
    main()
