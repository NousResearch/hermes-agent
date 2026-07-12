#!/usr/bin/env python3
"""Compatibility entrypoint for the packaged writer-only canary planner.

All planning and staging contracts live in ``gateway.canonical_writer_planner``
so the exact same reviewed implementation runs from a sealed wheel under
``python -I``.  This source-tree module intentionally contains no alternate
preview, routing, approval, or deployment logic.
"""

from __future__ import annotations

from gateway.canonical_writer_planner import (
    ACTIVATION_PLAN_SCHEMA,
    ROOT_COLLECTOR_MODULE,
    ActivationDigests,
    ActivationPaths,
    ActivationPlan,
    ExternalNativeExecutableMapping,
    HostNumericIdentities,
    NativeObservationDigests,
    PreapprovedNativeExecutablePolicy,
    build_activation_plan,
    build_and_stage_final_activation_plan,
    build_and_stage_native_observation_plan,
    build_native_observation_plan,
    load_release_manifest,
    main,
)


__all__ = [
    "ACTIVATION_PLAN_SCHEMA",
    "ROOT_COLLECTOR_MODULE",
    "ActivationDigests",
    "ActivationPaths",
    "ActivationPlan",
    "ExternalNativeExecutableMapping",
    "HostNumericIdentities",
    "NativeObservationDigests",
    "PreapprovedNativeExecutablePolicy",
    "build_activation_plan",
    "build_and_stage_final_activation_plan",
    "build_and_stage_native_observation_plan",
    "build_native_observation_plan",
    "load_release_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
