"""Benchmark eager versus routed skill loading on deterministic catalogs.

Generates isolated SKILL.md catalogs, then exercises the production prompt
builder and trigger matcher. Makes no model, network, or user-profile calls.

Usage:
    .venv/bin/python scripts/benchmark_routed_skills.py
    .venv/bin/python scripts/benchmark_routed_skills.py --iterations 500
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

_DESCRIPTION = (
    "Deterministic benchmark skill for measuring catalog prompt size and routed "
    "trigger lookup without depending on a user's private skill library."
)


def _measure(action: Callable[[], object], iterations: int) -> list[float]:
    timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        action()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return timings


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _write_catalog(
    skills_root: Path,
    skill_count: int,
    category_count: int,
    trigger_fraction: float,
) -> int:
    shutil.rmtree(skills_root, ignore_errors=True)
    skills_root.mkdir(parents=True)
    trigger_count = min(skill_count, max(1, round(skill_count * trigger_fraction)))

    for index in range(skill_count):
        top = f"category-{index % category_count:02d}"
        category = f"{top}/nested-{index % 3:02d}" if index % 4 == 0 else top
        name = f"skill-{index:04d}"
        skill_dir = skills_root / category / name
        skill_dir.mkdir(parents=True)
        trigger = (
            f"triggers:\n  - route synthetic skill {index:04d}\n"
            if index < trigger_count
            else ""
        )
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {_DESCRIPTION}\n"
            f"{trigger}"
            "---\n\n"
            f"# {name}\n\nRun the deterministic benchmark procedure.\n",
            encoding="utf-8",
        )

    return trigger_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", nargs="+", type=int, default=[73, 267, 500])
    parser.add_argument("--categories", type=int, default=12)
    parser.add_argument(
        "--trigger-fractions", nargs="+", type=float, default=[0.03, 0.85]
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cold-iterations", type=int, default=5)
    args = parser.parse_args()

    if any(count < 1 for count in args.skills):
        parser.error("--skills values must be positive")
    if args.categories < 1:
        parser.error("--categories must be positive")
    if any(not 0 < fraction <= 1 for fraction in args.trigger_fractions):
        parser.error("--trigger-fractions values must be in (0, 1]")
    if args.iterations < 2 or args.cold_iterations < 1:
        parser.error("--iterations must be at least 2; cold iterations must be positive")

    with tempfile.TemporaryDirectory(prefix="hermes-routed-benchmark-") as temp:
        hermes_home = Path(temp)
        skills_root = hermes_home / "skills"
        os.environ["HERMES_HOME"] = str(hermes_home)

        # Import only after selecting an isolated profile. This keeps module-level
        # paths away from the user's real HERMES_HOME.
        import agent.skill_commands as skill_commands
        import tools.skills_tool as skills_tool
        from agent.prompt_builder import (
            build_skills_system_prompt,
            clear_skills_system_prompt_cache,
        )

        skills_tool.SKILLS_DIR = skills_root
        print(
            "skills triggers eager_B routed_B reduction "
            "cold_eager_ms cold_routed_ms scan_ms first_miss_ms warm_miss_ms quality"
        )

        for skill_count in args.skills:
            for trigger_fraction in args.trigger_fractions:
                trigger_count = _write_catalog(
                    skills_root,
                    skill_count,
                    min(args.categories, skill_count),
                    trigger_fraction,
                )
                skill_commands._skill_commands = {}
                skill_commands._skill_commands_platform = None
                skill_commands._compile_skill_trigger_pattern.cache_clear()
                clear_skills_system_prompt_cache(clear_snapshot=True)

                prompts = {}
                cold = {}
                for mode in ("eager", "routed"):
                    cold_times = []
                    for _ in range(args.cold_iterations):
                        clear_skills_system_prompt_cache(clear_snapshot=True)
                        started = time.perf_counter_ns()
                        prompts[mode] = build_skills_system_prompt(loading_mode=mode)
                        cold_times.append(
                            (time.perf_counter_ns() - started) / 1_000_000
                        )
                    cold[mode] = _median(cold_times)

                scan_ms = _median(
                    _measure(skill_commands.scan_skill_commands, args.cold_iterations)
                )
                skill_commands.scan_skill_commands()
                hit = "Please route synthetic skill 0000 for this request."
                miss = "This ordinary request has no deterministic route."

                skill_commands._compile_skill_trigger_pattern.cache_clear()
                first_miss_ms = _measure(
                    lambda: skill_commands.find_triggered_skill_command(miss), 1
                )[0]
                skill_commands.find_triggered_skill_command(miss)  # warm regex cache
                warm_miss_ms = _median(
                    _measure(
                        lambda: skill_commands.find_triggered_skill_command(miss),
                        args.iterations,
                    )
                )

                exact = skill_commands.find_triggered_skill_command(hit)
                casefolded = skill_commands.find_triggered_skill_command(
                    "ROUTE SYNTHETIC SKILL 0000 now."
                )
                boundary = skill_commands.find_triggered_skill_command(
                    "Please reroute synthetic skill 0000."
                )
                ordinary = skill_commands.find_triggered_skill_command(miss)
                passed = sum(
                    (exact is not None, casefolded is not None, boundary is None, ordinary is None)
                )

                eager_bytes = len(prompts["eager"].encode("utf-8"))
                routed_bytes = len(prompts["routed"].encode("utf-8"))
                reduction = 100 * (eager_bytes - routed_bytes) / eager_bytes
                print(
                    f"{skill_count:>6} {trigger_count:>8} "
                    f"{eager_bytes:>7} {routed_bytes:>8} {reduction:>8.2f}% "
                    f"{cold['eager']:>13.3f} {cold['routed']:>14.3f} "
                    f"{scan_ms:>7.3f} {first_miss_ms:>13.3f} "
                    f"{warm_miss_ms:>12.3f} {passed:>5}/4"
                )


if __name__ == "__main__":
    main()
