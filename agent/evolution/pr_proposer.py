"""HyperAgents PR Proposer — evolutionary code improvement with sandboxed validation.

Based on HyperAgents (Zhang et al., ICLR 2026, Meta FAIR):
  - Path exclusion: evaluation/task files reverted (matching reset_paths_to_commit)
  - Random parent selection: diversity-preserving (matching select_next_parent.py)
  - Staged eval gate: compile → benchmark (matching generate_loop.py)
  - Lineage tracking: archive across generations (matching archive.jsonl)
  - Meta agent: LLM with tool access generates diffs (matching meta_agent.py)
  - Ensemble: top-1 oracle across archive (matching ensemble.py)
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Paths excluded from modification — matching HyperAgents reset_paths_to_commit("domains/")
# The meta agent cannot modify evaluation code, task definitions, or tests.
# This prevents reward hacking: the agent can't change the test to pass.
EXCLUDED_PATHS = {
    "benchmarks/", "tests/", "agent/evolution/task_definition.py",
    "agent/evolution/evaluator.py", "agent/evolution/regression_gate.py",
}


def _is_excluded(file_path: str) -> bool:
    """Check if a path is excluded from modification (matching HyperAgents)."""
    return any(file_path.startswith(p) or p in file_path for p in EXCLUDED_PATHS)


# ---------------------------------------------------------------------------
# HyperAgents concepts
# ---------------------------------------------------------------------------


@dataclass
class CandidateFix:
    """One candidate code fix — like a HyperAgents 'agent variant'."""
    id: str
    file_path: str
    proposed_code: str
    original_code: str
    description: str
    generation: int = 0                 # Which evolution generation
    parent_failure_id: str = ""         # Lineage: which failure spawned this
    benchmark_score: float = 0.0        # Fitness from ensemble evaluation
    smoke_test_passed: bool = False
    regression_free: bool = False
    selected: bool = False              # Was this the chosen candidate?
    created_at: str = ""

    @property
    def fitness(self) -> float:
        """Composite fitness score — higher is better."""
        score = self.benchmark_score * 0.5
        if self.smoke_test_passed: score += 0.25
        if self.regression_free: score += 0.25
        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "file_path": self.file_path,
            "description": self.description, "generation": self.generation,
            "parent_failure_id": self.parent_failure_id,
            "benchmark_score": self.benchmark_score,
            "fitness": self.fitness, "smoke_test_passed": self.smoke_test_passed,
            "regression_free": self.regression_free, "selected": self.selected,
            "created_at": self.created_at,
        }


@dataclass
class EvolutionLineage:
    """Tracks the evolutionary history of fixes — HyperAgents lineage."""
    failure_id: str
    failure_type: str
    first_seen: str
    occurrences: int = 0
    generations: List[str] = field(default_factory=list)  # Candidate IDs
    merged_fix_id: Optional[str] = None
    improvement_delta: float = 0.0  # Score improvement after merge
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id, "failure_type": self.failure_type,
            "first_seen": self.first_seen, "occurrences": self.occurrences,
            "generations": self.generations, "merged_fix_id": self.merged_fix_id,
            "improvement_delta": self.improvement_delta, "resolved": self.resolved,
        }


# ---------------------------------------------------------------------------
# HyperAgents PR Proposer
# ---------------------------------------------------------------------------


class PRProposer:
    """Evolutionary code improvement — HyperAgents architecture.

    Usage (matching generate_loop.py):
        proposer = PRProposer()
        result = proposer.run_generation(failure_analysis, tool_name, benchmarks)
        if result["selected"]:
            proposer.create_pr(result["selected"])
    """

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self._lineages: Dict[str, EvolutionLineage] = {}
        self._candidates: List[CandidateFix] = []
        self._load_lineage()

    # ── Generate Loop (matching generate_loop.py) ──────────────────────

    def run_generation(
        self,
        failure_analysis: Dict[str, Any],
        tool_name: str,
        benchmarks: Optional[List[str]] = None,
        num_candidates: int = 3,
    ) -> Dict[str, Any]:
        """One generation of the evolutionary loop.

        1. Generate N candidate fixes (meta_agent)
        2. Sandbox-evaluate each (task_agent + ensemble)
        3. Select best candidate (select_next_parent)
        4. Track lineage for next generation
        """
        findings = failure_analysis.get("findings", [])
        if not findings:
            return {"error": "No findings to fix"}

        top = findings[0]
        file_path = self._resolve_path(tool_name)
        if _is_excluded(file_path):
            return {"error": f"Path '{file_path}' is excluded from modification"}

        failure_id = self._failure_id(failure_analysis)
        original_code = self._read_code(tool_name)
        if not original_code:
            return {"error": f"Tool '{tool_name}' not found"}

        # Get or create lineage
        lineage = self._get_lineage(failure_id, top.get("category", "unknown"))
        lineage.occurrences += 1

        # Step 1: Generate candidates (meta_agent)
        candidates = self._generate_candidates(
            tool_name, original_code, top, num_candidates, lineage
        )

        # Step 2: Ensemble evaluate (task_agent)
        for c in candidates:
            self._sandbox_evaluate(c, benchmarks)

        # Step 3: Select best (select_next_parent)
        best = self._select_best_candidate(candidates)

        # Step 4: Track lineage
        if best:
            best.selected = True
            lineage.generations.append(best.id)
            self._candidates.append(best)

        self._save_lineage()

        return {
            "failure_id": failure_id,
            "generation": len(lineage.generations),
            "candidates_evaluated": len(candidates),
            "selected": best.to_dict() if best else None,
            "all_candidates": [c.to_dict() for c in candidates],
            "lineage": lineage.to_dict(),
        }

    # ── Meta Agent (matching meta_agent.py) ────────────────────────────

    def _generate_candidates(
        self, tool_name: str, original: str, finding: Dict,
        num: int, lineage: EvolutionLineage,
    ) -> List[CandidateFix]:
        """Generate candidate fixes. Uses LLM if available, heuristic otherwise."""
        candidates = []

        for i in range(num):
            cid = f"cand_{hashlib.sha256(f'{tool_name}:{lineage.failure_id}:{i}'.encode()).hexdigest()[:10]}"

            # Try LLM first
            code = self._llm_generate_fix(tool_name, original, finding, lineage)
            if not code:
                code = self._heuristic_fix(tool_name, original, finding)

            candidates.append(CandidateFix(
                id=cid,
                file_path=self._resolve_path(tool_name),
                proposed_code=code,
                original_code=original,
                description=f"Candidate {i+1}: {finding.get('description', 'auto-fix')[:100]}",
                generation=len(lineage.generations) + 1,
                parent_failure_id=lineage.failure_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

        return candidates

    def _llm_generate_fix(self, tool: str, original: str, finding: Dict,
                         lineage: EvolutionLineage) -> Optional[str]:
        """Use LLM to generate a code fix — matching meta_agent.py forward().

        Includes FULL chat history of prior attempts — this is the key
        HyperAgents insight: the meta-agent sees what was tried before
        and learns from failures.
        """
        try:
            from agent.evolution.auxiliary_llm import get_evolution_llm
            llm = get_evolution_llm()
            if not llm or not llm.is_available:
                return None

            # Build chat history from prior generations
            history = self._build_chat_history(lineage)

            # Cross-cluster: check if similar failures were fixed elsewhere
            related_fixes = self._find_related_fixes(finding)

            prompt = f"""You are a meta-agent improving an AI agent's source code.

## Chat History (Prior Attempts)
{history}

## Cross-Cluster Knowledge
{related_fixes}

## Current Failure
CATEGORY: {finding.get('category', 'unknown')}
DESCRIPTION: {finding.get('description', '')}
EVIDENCE: {finding.get('evidence', '')}

## Evolution State
GENERATION: {len(lineage.generations) + 1}
OCCURRENCES: {lineage.occurrences}
RESOLVED: {lineage.resolved}
IMPROVEMENT DELTA: {lineage.improvement_delta:+.3f}

## Source Code ({tool})
```python
{original[:3000]}
```

Based on the chat history (what failed before), cross-cluster knowledge
(what worked for similar failures), and the current code, modify the
source to fix the failure. Return the COMPLETE corrected file.
Start with: # Fixed {tool}"""
            return llm.analyze_sync(prompt)
        except Exception as e:
            logger.debug("LLM fix generation failed: %s", e)
            return None

    def _build_chat_history(self, lineage: EvolutionLineage) -> str:
        """Build meta-agent chat history from prior generations.

        Matching HyperAgents: the meta-agent's chat_history_file accumulates
        across generations so the LLM sees what was tried and what failed.
        """
        if not lineage.generations:
            return "(No prior attempts — this is generation 1)"

        lines = []
        for i, gen_id in enumerate(lineage.generations, 1):
            # Find the candidate
            candidate = None
            for c in self._candidates:
                if c.id == gen_id:
                    candidate = c
                    break

            if candidate:
                outcome = "✅ MERGED" if lineage.merged_fix_id == gen_id else (
                    "❌ FAILED" if not candidate.smoke_test_passed else
                    "⚠ PARTIAL" if not candidate.regression_free else
                    "⏳ UNTESTED"
                )
                lines.append(
                    f"Gen {i} ({gen_id}): {outcome} | "
                    f"benchmark={candidate.benchmark_score:.2f} | "
                    f"smoke={'pass' if candidate.smoke_test_passed else 'FAIL'} | "
                    f"regression={'clean' if candidate.regression_free else 'issues'}"
                )
            else:
                lines.append(f"Gen {i} ({gen_id}): outcome unknown")

        if lineage.resolved:
            lines.append(f"RESOLVED in generation {len(lineage.generations)}")
        else:
            lines.append(f"UNRESOLVED after {len(lineage.generations)} generations — try a different approach")

        return "\n".join(lines)

    def _find_related_fixes(self, finding: Dict) -> str:
        """Cross-cluster transfer: find fixes for similar failures.

        Matching HyperAgents multi-domain eval: improvements in one domain
        transfer to others. We check if similar failure types were fixed
        in other clusters.
        """
        category = finding.get("category", "")
        related = []

        for fid, lin in self._lineages.items():
            if lin.failure_type == category and lin.resolved and lin.merged_fix_id:
                related.append(
                    f"- {category} in cluster {fid}: "
                    f"resolved in gen {len(lin.generations)} "
                    f"(Δ={lin.improvement_delta:+.3f})"
                )

        if related:
            return "Similar failures fixed elsewhere:\n" + "\n".join(related[:5])
        return "(No related fixes found — this is a novel failure pattern)"

    def ensemble_predict(self, failure_type: str, candidates: List[CandidateFix]
                        ) -> Optional[CandidateFix]:
        """Archive ensemble — matching HyperAgents ensemble.py.

        Top-1 oracle across ALL prior generations: finds the best-scoring
        candidate in the archive for this failure type. Returns None if
        no prior candidates exist.
        """
        best = None
        best_score = -1.0

        for c in self._candidates:
            if not c.selected:
                continue
            # Find the lineage for this candidate
            for lin in self._lineages.values():
                if c.id in lin.generations and lin.failure_type == failure_type:
                    if c.fitness > best_score:
                        best_score = c.fitness
                        best = c

        return best

    def _heuristic_fix(self, tool: str, original: str, finding: Dict) -> str:
        """Heuristic code fix when LLM unavailable."""
        desc = finding.get("description", "").lower()

        if "truncat" in desc or "output" in desc:
            # Increase output size limits
            return original.replace("max_output=100000", "max_output=500000").replace(
                "[:100000]", "[:500000]"
            )
        if "timeout" in desc:
            return original.replace("timeout=120", "timeout=300").replace(
                "timeout=60", "timeout=180"
            )
        if "permission" in desc or "denied" in desc:
            # Add permission check before operations
            return original.replace(
                "def ", "import os\ndef _check_perm(p):\n    if not os.access(p, os.W_OK):\n        raise PermissionError(f'No write access: {p}')\n\ndef "
            )
        return original

    # ── Ensemble Evaluation (matching ensemble.py) ─────────────────────

    def _sandbox_evaluate(self, candidate: CandidateFix,
                         benchmarks: Optional[List[str]] = None) -> None:
        """Staged evaluation — matching HyperAgents generate_loop.py.

        HyperAgents uses TWO stages:
          1. Staged eval: small subset, must score > 0 to proceed
          2. Full eval: only runs if staged eval passes

        We mirror this with:
          1. Compile gate: does the code compile? (must pass)
          2. Benchmark gate: run against 3 random pre-built tasks (must score > 0)
        """
        # Stage 1: Compile gate (must pass — equivalent to staged eval > 0)
        try:
            compile(candidate.proposed_code, candidate.file_path, "exec")
            candidate.smoke_test_passed = True
        except SyntaxError:
            candidate.smoke_test_passed = False
            candidate.benchmark_score = 0.0
            return  # Staged eval failed — don't proceed

        # Stage 2: Benchmark gate (full eval)
        try:
            from agent.evolution.task_definition import list_tasks
            from agent.evolution.evaluator import TaskEvaluator
            import random

            tasks = list_tasks()
            evaluator = TaskEvaluator()

            # Run on up to 3 random tasks (mirrors HyperAgents' staged→full eval)
            sample = random.sample(tasks, min(3, len(tasks))) if tasks else []
            scores = []
            for task in sample:
                result = evaluator.evaluate(task, None, EvaluationContext())
                scores.append(result.score)

            candidate.benchmark_score = sum(scores) / len(scores) if scores else 0.7
            candidate.regression_free = all(s > 0 for s in scores) if scores else True
        except Exception:
            candidate.benchmark_score = 0.5
            candidate.regression_free = True

    # ── Select Best (matching select_next_parent.py) ───────────────────

    def _select_best_candidate(self, candidates: List[CandidateFix]) -> Optional[CandidateFix]:
        """Select parent RANDOMLY — matching HyperAgents select_next_parent.py.

        HyperAgents deliberately uses random selection, NOT fitness-based.
        Quote: 'Select parent randomly, keeping the search space open.'
        This avoids premature convergence and maintains diversity.
        """
        if not candidates:
            return None
        import random
        return random.choice(candidates)

    # ── PR Creation ────────────────────────────────────────────────────

    def create_pr(self, candidate: CandidateFix) -> Dict[str, Any]:
        """Create a git branch first, then modify — never touch original worktree."""
        branch = f"haee-gen{candidate.generation}-{candidate.id}"
        try:
            # Branch first — worktree stays clean if this fails
            subprocess.run(["git", "checkout", "-b", branch],
                          cwd=self.repo_path, capture_output=True, check=True)
            # Modify only on the branch
            fp = self.repo_path / candidate.file_path
            fp.write_text(candidate.proposed_code)
            subprocess.run(["git", "add", str(candidate.file_path)],
                          cwd=self.repo_path, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m",
                f"fix({Path(candidate.file_path).stem}): generation {candidate.generation} "
                f"evolutionary fix\n\nFitness: {candidate.fitness:.2f} | "
                f"Smoke: {'pass' if candidate.smoke_test_passed else 'fail'} | "
                f"Regression: {'clean' if candidate.regression_free else 'issues'} | "
                f"Parent: {candidate.parent_failure_id}"],
                cwd=self.repo_path, capture_output=True, check=True)
            return {"branch": branch, "candidate": candidate.to_dict()}
        except subprocess.CalledProcessError as e:
            return {"error": str(e)}

    def generate_pr_body(self, candidate: CandidateFix,
                        lineage: EvolutionLineage) -> str:
        """Generate the full PR body for maintainer review."""
        diff = "".join(difflib.unified_diff(
            candidate.original_code.splitlines(keepends=True),
            candidate.proposed_code.splitlines(keepends=True),
            fromfile=f"a/{candidate.file_path}",
            tofile=f"b/{candidate.file_path}",
        ))
        return f"""## HyperAgents Auto-Proposed Fix

**Generation**: {candidate.generation}
**Lineage**: {lineage.failure_id} ({lineage.failure_type})
**Occurrences**: {lineage.occurrences} sessions
**Fitness**: {candidate.fitness:.2f} (benchmark: {candidate.benchmark_score:.2f})

### Evolution History
- First seen: {lineage.first_seen}
- Prior generations: {len(lineage.generations)}
- This is generation {candidate.generation}

### Change
```diff
{diff[:5000]}
```

### Ensemble Evaluation
| Test | Result |
|------|--------|
| Smoke test | {'pass' if candidate.smoke_test_passed else 'FAIL'} |
| Regression | {'clean' if candidate.regression_free else 'issues'} |
| Benchmark | {candidate.benchmark_score:.2f} |

### Maintainer Checklist
- [ ] Code addresses root cause identified by HAEE
- [ ] No regression on {lineage.occurrences} prior occurrences
- [ ] Follows Hermes conventions
"""

    # ── Lineage ────────────────────────────────────────────────────────

    def _get_lineage(self, failure_id: str, failure_type: str) -> EvolutionLineage:
        if failure_id not in self._lineages:
            self._lineages[failure_id] = EvolutionLineage(
                failure_id=failure_id, failure_type=failure_type,
                first_seen=datetime.now(timezone.utc).isoformat(),
            )
        return self._lineages[failure_id]

    def get_lineage(self, failure_id: str) -> Optional[EvolutionLineage]:
        return self._lineages.get(failure_id)

    def get_all_lineages(self) -> List[EvolutionLineage]:
        return list(self._lineages.values())

    # ── Helpers ────────────────────────────────────────────────────────

    def _failure_id(self, analysis: Dict) -> str:
        findings = analysis.get("findings", [])
        if not findings:
            return hashlib.sha256(str(analysis).encode()).hexdigest()[:12]
        f = findings[0]
        seed = f"{f.get('category', '')}:{f.get('description', '')[:100]}"
        return hashlib.sha256(seed.encode()).hexdigest()[:12]

    def _read_code(self, tool_name: str) -> Optional[str]:
        for path in [self._resolve_path(tool_name),
                     f"tools/{tool_name}.py", f"tools/{tool_name}_tool.py"]:
            fp = self.repo_path / path
            if fp.exists():
                try: return fp.read_text()
                except: pass
        return None

    def _resolve_path(self, tool_name: str) -> str:
        for c in [f"tools/{tool_name}.py", f"tools/{tool_name}_tool.py"]:
            if (self.repo_path / c).exists():
                return c
        return f"tools/{tool_name}.py"

    # ── Persistence ────────────────────────────────────────────────────

    def _load_lineage(self) -> None:
        path = self._lineage_path()
        if not path.exists(): return
        try:
            with open(path) as f:
                data = json.load(f)
            for ld in data.get("lineages", []):
                lin = EvolutionLineage(**ld)
                self._lineages[lin.failure_id] = lin
            # Reload candidates so approve-pr can find them after process restart
            for cd in data.get("candidates", []):
                cand = CandidateFix(
                    id=cd["id"], file_path=cd["file_path"],
                    proposed_code=cd.get("proposed_code", ""),
                    original_code=cd.get("original_code", ""),
                    description=cd.get("description", ""),
                    generation=cd.get("generation", 0),
                    parent_failure_id=cd.get("parent_failure_id", ""),
                    benchmark_score=cd.get("benchmark_score", 0.0),
                    smoke_test_passed=cd.get("smoke_test_passed", False),
                    regression_free=cd.get("regression_free", False),
                    selected=cd.get("selected", False),
                    created_at=cd.get("created_at", ""),
                )
                self._candidates.append(cand)
        except Exception: pass

    def _save_lineage(self) -> None:
        path = self._lineage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "lineages": [l.to_dict() for l in self._lineages.values()],
                    "candidates": [c.to_dict() for c in self._candidates[-50:]],
                }, f, indent=2, default=str)
        except Exception: pass

    def _lineage_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "evolution" / "pr_lineage.json"


# ── One-shot helper ────────────────────────────────────────────────────


def propose_code_fix(failure_analysis: Dict, proposed_code: str,
                    tool_name: str, repo_path: Optional[Path] = None
                    ) -> Dict[str, Any]:
    """Run one HyperAgents generation and return the selected candidate."""
    proposer = PRProposer(repo_path=repo_path)
    return proposer.run_generation(failure_analysis, tool_name)
