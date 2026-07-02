#!/usr/bin/env python3
"""Isolated adoption smoke for the fork-vendored LCM context engine.

This probe drives the real ``plugins.context_engine.lcm.LCMEngine`` in-process
against a throwaway profile directory. It never installs the plugin, never uses a
live profile database, and stubs summarization deterministically for offline
reproducibility.

Smoke dimensions:
  1. load + identity
  2. normal chat/tool ingestion
  3. threshold compaction
  4. lcm_grep / lcm_describe / lcm_expand byte-exact recall
  5. bad-id loud error
  6. reset semantics
  7. fail-open summarization fallback

Exit code 0 = all checks passed; non-zero = at least one check failed or the
isolation guard refused the requested profile path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
FORK_PLUGIN_DIR = WORKTREE_ROOT / "plugins" / "context_engine" / "lcm"
_ALLOWED_PROFILE_MARKERS = {"staging", "tmp", "temp"}


@dataclass(frozen=True)
class Check:
    """One smoke assertion with observed evidence."""

    name: str
    ok: bool
    evidence: str

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.name} — {self.evidence}"


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _hermes_roots() -> list[Path]:
    roots: list[Path] = []
    for parent in (WORKTREE_ROOT, *WORKTREE_ROOT.parents):
        if parent.name == ".hermes":
            roots.append(parent)
            break
    roots.append(Path.home() / ".hermes")
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        home = Path(env_home).expanduser()
        if home.name == ".hermes":
            roots.append(home)
        for parent in home.parents:
            if parent.name == ".hermes":
                roots.append(parent)
                break
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(Path(key))
    return deduped


def _looks_temp_or_staging(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    if _path_is_relative_to(resolved, WORKTREE_ROOT / "staging"):
        return True
    components = {part.lower() for part in resolved.parts}
    if components & _ALLOWED_PROFILE_MARKERS:
        return True
    return any(part.lower().startswith(("tmp", "temp", "staging")) for part in resolved.parts)


def _live_roots() -> list[Path]:
    roots: list[Path] = []
    for hermes_root in _hermes_roots():
        roots.extend([hermes_root / "plugins", hermes_root / "profiles"])
    return roots


def _live_path_refusal(path: Path) -> str | None:
    resolved = path.expanduser().resolve()
    for root in _live_roots():
        if _path_is_relative_to(resolved, root) or resolved == root.expanduser().resolve():
            if _looks_temp_or_staging(resolved):
                return None
            return f"{resolved} is under live Hermes path {root.expanduser().resolve()}"
    return None


def _guard_isolated_profile(profile_dir: Path) -> None:
    refusal = _live_path_refusal(profile_dir)
    if refusal:
        raise RuntimeError(f"REFUSING live profile/plugin path: {refusal}")


def _ensure_plugin_importable() -> None:
    if str(WORKTREE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKTREE_ROOT))


def _new_engine(LCMEngine, LCMConfig, profile_dir: Path, name: str, *, leaf_chunk_tokens: int = 1):
    db_dir = profile_dir / "lcm-smoke"
    db_dir.mkdir(parents=True, exist_ok=True)
    cfg = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=leaf_chunk_tokens,
        database_path=str(db_dir / f"{name}.db"),
        # This probe pins a tiny fixed tail to force compaction on a toy
        # corpus; the token-budgeted tail would (correctly) widen past the
        # whole fixture, so pin the legacy fixed-count regime.
        fresh_tail_token_budget_enabled=False,
    )
    engine = LCMEngine(config=cfg, hermes_home=str(profile_dir))
    engine.context_length = 200_000
    engine.threshold_tokens = int(200_000 * cfg.context_threshold)
    return engine, cfg


def _convo_with_secret(secret: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": f"Remember the deploy code is {secret} for prod."},
        {"role": "assistant", "content": f"Noted {secret}."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
        {"role": "user", "content": "What was the deploy code?"},
    ]


def _shutdown(engine: Any) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def run_smoke(plugin_dir: Path, profile_dir: Path) -> list[Check]:
    """Run the seven isolated smoke checks against the fork LCM engine."""
    _guard_isolated_profile(profile_dir)
    _ensure_plugin_importable()

    from agent.context_engine import ContextEngine
    from plugins.context_engine.lcm.config import LCMConfig
    from plugins.context_engine.lcm.engine import LCMEngine
    import plugins.context_engine.lcm.engine as lcm_engine
    import plugins.context_engine.lcm.escalation as lcm_escalation

    checks: list[Check] = []
    profile_dir.mkdir(parents=True, exist_ok=True)

    # 1. load + identity
    e, _ = _new_engine(LCMEngine, LCMConfig, profile_dir, "identity")
    try:
        e.on_session_start("identity-s", platform="cli", context_length=200_000)
        is_engine = issubclass(LCMEngine, ContextEngine)
        plugin_yaml = plugin_dir / "plugin.yaml"
        checks.append(Check(
            "load+identity",
            is_engine and e.name == "lcm" and plugin_yaml.exists(),
            f"ContextEngine subclass={is_engine}, engine.name={e.name!r}, "
            f"plugin_yaml={plugin_yaml.exists()}",
        ))

        # 2. normal chat/tool ingestion
        e._ingest_messages([
            {"role": "user", "content": "hello, normal turn"},
            {"role": "assistant", "content": "hi"},
        ])
        status = json.loads(e.handle_tool_call("lcm_status", {}))
        describe = json.loads(e.handle_tool_call("lcm_describe", {}))
        normal_ok = (
            bool(status.get("session_id"))
            and describe.get("store_message_count", 0) >= 2
        )
        checks.append(Check(
            "normal-chat/tool-ingestion",
            normal_ok,
            f"lcm_status.session_id={status.get('session_id')!r}, "
            f"lcm_describe.store_message_count={describe.get('store_message_count')}",
        ))
    finally:
        _shutdown(e)

    # 3 + 4 + 5. deterministic compaction, recall, bad-id error
    original_summary = lcm_engine.summarize_with_escalation
    lcm_engine.summarize_with_escalation = lambda **kw: (
        "SUMMARY: earlier turns covered the deploy code and arithmetic", 1
    )
    e2, _ = _new_engine(LCMEngine, LCMConfig, profile_dir, "compact")
    try:
        e2.on_session_start("compact-s", platform="cli", context_length=200_000)
        secret = "DEPLOY-CODE-7F3A"
        convo = _convo_with_secret(secret)
        fires = e2.should_compress(e2.threshold_tokens) and not e2.should_compress(1000)
        active = e2.compress(list(convo))
        compacted = (
            e2._last_compression_status == "compacted"
            and e2.compression_count == 1
            and len(active) < len(convo)
        )
        has_summary = any("SUMMARY:" in (msg.get("content") or "") for msg in active)
        checks.append(Check(
            "threshold-compaction",
            fires and compacted and has_summary,
            f"should_compress(threshold)=True/should_compress(1000)=False={fires}; "
            f"status={e2._last_compression_status}, count={e2.compression_count}, "
            f"active {len(active)}<orig {len(convo)}; DAG-summary-in-active={has_summary}",
        ))

        fact_out_of_active = not any(secret in (msg.get("content") or "") for msg in active)
        grep = json.loads(e2.handle_tool_call("lcm_grep", {"query": secret}))
        describe_after = json.loads(e2.handle_tool_call("lcm_describe", {}))
        describe_ok = isinstance(describe_after, dict) and "error" not in describe_after
        results = grep.get("results") or []
        chosen = next(
            (r for r in results if secret in (r.get("snippet") or r.get("content") or "")),
            results[0] if results else {},
        )
        store_id = chosen.get("store_id")
        expand = json.loads(e2.handle_tool_call("lcm_expand", {"store_id": store_id})) if store_id else {}
        byte_exact = secret in (expand.get("content") or "")
        recall_ok = (
            fact_out_of_active
            and grep.get("total_results", 0) >= 1
            and describe_ok
            and store_id is not None
            and byte_exact
        )
        describe_keys = ",".join(sorted(str(key) for key in describe_after.keys())[:6])
        checks.append(Check(
            "grep/describe/expand-byte-exact-recall",
            recall_ok,
            f"fact_out_of_active={fact_out_of_active}; grep total_results={grep.get('total_results', 0)}; "
            f"describe_ok={describe_ok} keys={describe_keys}; selected store_id={store_id}; "
            f"expand.content recovers raw {secret!r} byte-exact={byte_exact}",
        ))

        expand_bad = json.loads(e2.handle_tool_call("lcm_expand", {"store_id": 999_999}))
        checks.append(Check(
            "bad-id-loud-error",
            "error" in expand_bad and "999999" in json.dumps(expand_bad),
            f"lcm_expand(bad id) -> {json.dumps(expand_bad)[:160]}",
        ))
    finally:
        lcm_engine.summarize_with_escalation = original_summary
        _shutdown(e2)

    # 6. reset semantics
    lcm_engine.summarize_with_escalation = lambda **kw: (
        "SUMMARY: reset smoke retained raw facts", 1
    )
    e3, _ = _new_engine(LCMEngine, LCMConfig, profile_dir, "reset")
    try:
        e3.on_session_start("reset-s", platform="cli", context_length=200_000)
        reset_convo = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fact ALPHA-secret-token"},
            {"role": "assistant", "content": "ok alpha"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        e3.compress(list(reset_convo))
        count_before = e3.compression_count
        grep_before = json.loads(
            e3.handle_tool_call("lcm_grep", {"query": "ALPHA-secret-token"})
        ).get("total_results", 0)
        e3.on_session_reset()
        count_after = e3.compression_count
        grep_after_all = json.loads(e3.handle_tool_call(
            "lcm_grep", {"query": "ALPHA-secret-token", "session_scope": "all"}
        )).get("total_results", 0)
        reset_ok = count_before >= 1 and count_after == 0 and grep_after_all >= 1
        checks.append(Check(
            "reset-semantics",
            reset_ok,
            f"compression_count {count_before}->{count_after} after on_session_reset; "
            f"grep-before={grep_before}; lossless store still answers grep after reset "
            f"(all-scope)={grep_after_all}",
        ))
    finally:
        lcm_engine.summarize_with_escalation = original_summary
        _shutdown(e3)

    # 7. fail-open summarization fallback
    original_chain = lcm_escalation._invoke_summary_llm_chain
    lcm_escalation._invoke_summary_llm_chain = lambda *a, **k: None
    e4, _ = _new_engine(LCMEngine, LCMConfig, profile_dir, "failopen")
    try:
        e4.on_session_start("fail-s", platform="cli", context_length=200_000)
        fail_convo = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"fact number {i} with filler text to summarize"}
            for i in range(8)
        ]
        crashed = False
        crash_detail = ""
        try:
            fail_active = e4.compress(list(fail_convo))
        except Exception as exc:  # noqa: BLE001 - evidence belongs in report
            crashed = True
            fail_active = None
            crash_detail = repr(exc)
        fail_grep = 0
        if not crashed:
            fail_grep = json.loads(
                e4.handle_tool_call("lcm_grep", {"query": "fact number 0"})
            ).get("total_results", 0)
        fail_ok = (not crashed) and fail_active is not None and len(fail_active) >= 1 and fail_grep >= 1
        checks.append(Check(
            "fail-open",
            fail_ok,
            (
                f"summarizer LLM unavailable -> no crash={not crashed}, "
                f"status={e4._last_compression_status}, "
                f"active_len={len(fail_active) if fail_active else None}, "
                f"raw still grep-recoverable={fail_grep}"
            ) if not crashed else f"CRASHED (fail-closed) -> {crash_detail}",
        ))
    finally:
        lcm_escalation._invoke_summary_llm_chain = original_chain
        _shutdown(e4)

    return checks


def _plugin_identity(plugin_dir: Path) -> dict[str, str]:
    info = {"vendored_path": str(plugin_dir)}
    yaml_path = plugin_dir / "plugin.yaml"
    if yaml_path.exists():
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                info["plugin_name"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("version:"):
                info["plugin_version"] = line.split(":", 1)[1].strip().strip('"')
    provenance_path = plugin_dir / "VENDORED_FROM.txt"
    if provenance_path.exists():
        provenance = " ".join(
            line.strip() for line in provenance_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        info["provenance"] = provenance[:600]
    return info


def write_report(out_path: Path, checks: Iterable[Check], identity: dict[str, str], profile_dir: Path) -> None:
    checks = list(checks)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    passed = sum(1 for check in checks if check.ok)
    total = len(checks)
    verdict = "GO (isolated smoke clean)" if passed == total else "BLOCKED (smoke failure)"

    lines: list[str] = [
        "# hermes-lcm Adoption Smoke — Isolated Profile",
        "",
        f"**Generated:** {ts}",
        f"**Probe:** `scripts/probe_hermes_lcm_isolated.py --profile-dir {profile_dir} --out {out_path}`",
        f"**Verdict:** **{verdict}** — {passed}/{total} checks passed",
        "",
        "## Plugin under test",
        "",
    ]
    for key, value in identity.items():
        lines.append(f"- **{key}:** `{value}`")
    lines.extend([
        "",
        "## Isolation guarantees",
        "",
        f"- Engine loaded only from the fork-vendored plugin at `{FORK_PLUGIN_DIR}`.",
        f"- Profile/storage root was `{profile_dir}` and passed the live-path refusal guard.",
        "- No install script was run; no writes are made to live `~/.hermes/plugins` or profile directories.",
        "- Each check uses a throwaway SQLite DB under the supplied temp/staging profile directory.",
        "- Summarization is deterministic/stubbed for offline reproducibility.",
        "",
        "## Smoke results",
        "",
        "| # | Check | Result | Evidence |",
        "|---|-------|--------|----------|",
    ])
    for index, check in enumerate(checks, 1):
        evidence = check.evidence.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {index} | {check.name} | {'PASS' if check.ok else 'FAIL'} | {evidence} |")
    lines.extend([
        "",
        "## Raw check log",
        "",
        "```",
    ])
    lines.extend(check.line() for check in checks)
    lines.extend([
        "```",
        "",
        "## Notes for the reviewer",
        "",
        "- This isolated smoke proves in-process load, compaction, byte-exact recall, reset behavior, and fail-open fallback.",
        "- It does not enable LCM in any live Hermes profile and does not prove a live model chooses retrieval tools unaided.",
        "- Public redistribution remains gated by the upstream licensing posture recorded in `plugins/context_engine/lcm/VENDORED_FROM.txt`.",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        required=True,
        help="Throwaway temp/staging profile directory used for isolated smoke storage.",
    )
    parser.add_argument("--out", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)

    profile_dir = Path(args.profile_dir).expanduser()
    if not profile_dir.is_absolute():
        profile_dir = WORKTREE_ROOT / profile_dir
    profile_dir = profile_dir.resolve()

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = WORKTREE_ROOT / out_path
    out_path = out_path.resolve()

    try:
        _guard_isolated_profile(profile_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if not FORK_PLUGIN_DIR.is_dir():
        print(f"Fork-vendored LCM plugin not found at {FORK_PLUGIN_DIR}", file=sys.stderr)
        return 2

    old_hermes_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(profile_dir)
    try:
        checks = run_smoke(FORK_PLUGIN_DIR, profile_dir)
    finally:
        if old_hermes_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_hermes_home

    write_report(out_path, checks, _plugin_identity(FORK_PLUGIN_DIR), profile_dir)
    for check in checks:
        print(check.line())
    passed = sum(1 for check in checks if check.ok)
    print(f"\n{passed}/{len(checks)} checks passed. Report: {out_path}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
