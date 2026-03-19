"""
RSM Engine — Recursive Self-Modification for Cosmos Orchestrator

Allows the Cosmos head orchestrator to read, analyze, propose edits to,
and modify its own .py modules with:
- Automatic backup (last 3 versions per file)
- Syntax validation (py_compile) before finalizing
- Auto-revert on compilation failure
- Lyapunov stability gate (blocks edits when drift > threshold)
- Full audit log (rsm_audit.jsonl)
- Scope lock: can ONLY modify files within cosmosynapse/engine/
"""

import os
import re
import json
import time
import shutil
import logging
import py_compile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sys

# Attempt to load Cosmic Ethics Validator
try:
    from Cosmos.core.collective.stewardship import StewardshipValidator
except ImportError:
    try:
        from core.collective.stewardship import StewardshipValidator
    except ImportError:
        StewardshipValidator = None

logger = logging.getLogger("RSM_ENGINE")

# Maximum backup versions per file
MAX_BACKUPS = 3

# Lyapunov drift threshold — block modifications above this
LYAPUNOV_GATE_THRESHOLD = 0.45


@dataclass
class RSMProposal:
    """A proposed code modification or creation."""
    file_path: str
    original_code: str
    replacement_code: str
    reason: str
    is_forge: bool = False
    timestamp: str = ""
    diff_preview: str = ""
    approved: bool = False
    applied: bool = False
    reverted: bool = False
    compile_ok: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.diff_preview:
            self.diff_preview = self._generate_diff()

    def _generate_diff(self) -> str:
        """Generate a simple unified diff preview."""
        orig_lines = self.original_code.splitlines()
        repl_lines = self.replacement_code.splitlines()
        diff = []
        if self.is_forge:
            diff.append("+++ NEW FORGED MODULE +++")
            for line in repl_lines:
                diff.append(f"+ {line}")
        else:
            for line in orig_lines:
                diff.append(f"- {line}")
            for line in repl_lines:
                diff.append(f"+ {line}")
        return "\n".join(diff[:50])  # Cap preview at 50 lines


class RSMEngine:
    """
    Recursive Self-Modification Engine.
    
    Allows Cosmos to read and modify its own source code with safety:
    - Scoped to cosmosynapse/engine/ directory only
    - Auto-backup before every edit
    - Syntax validation after every edit
    - Auto-revert on failure
    - Audit logging
    """

    def __init__(self, engine_dir: Optional[str] = None):
        """
        Initialize RSM Engine.
        
        Args:
            engine_dir: Path to the cosmosynapse/engine/ directory.
                       Auto-detected if not provided.
        """
        if engine_dir:
            self.engine_dir = Path(engine_dir)
        else:
            self.engine_dir = Path(__file__).parent
        
        self.backup_dir = self.engine_dir / ".rsm_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.audit_log_path = self.engine_dir / "rsm_audit.jsonl"
        self.pending_proposals: list[RSMProposal] = []
        self.applied_count = 0
        self.reverted_count = 0
        
        logger.info(f"RSM Engine initialized. Scope: {self.engine_dir}")
        logger.info(f"Backups: {self.backup_dir}")

    # ============================================
    # SCOPE VALIDATION
    # ============================================

    def _is_in_scope(self, path: str) -> bool:
        """Check if a file is within the allowed modification scope."""
        try:
            target = Path(path).resolve()
            scope = self.engine_dir.resolve()
            # Must be inside cosmosynapse/engine/ and must be a .py file
            return (
                target.suffix == ".py"
                and str(target).startswith(str(scope))
            )
        except Exception:
            return False

    def _resolve_path(self, filename: str) -> Optional[Path]:
        """Resolve a filename (with or without path) to a full path within scope."""
        # If it's already a full path
        if os.path.isabs(filename):
            p = Path(filename)
        else:
            p = self.engine_dir / filename
        
        p = p.resolve()
        if self._is_in_scope(str(p)):
            return p
        return None

    # ============================================
    # READ OPERATIONS
    # ============================================

    def list_modules(self) -> list[dict[str, str]]:
        """list all modifiable .py modules in scope."""
        modules = []
        for f in sorted(self.engine_dir.glob("*.py")):
            if f.name.startswith("__"):
                continue
            modules.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
        return modules

    def read_module(self, filename: str) -> Optional[str]:
        """
        Read the contents of a module.
        
        Args:
            filename: Module filename (e.g. 'cosmos_swarm_orchestrator.py')
                     or full path within scope.
        
        Returns:
            File contents as string, or None if out of scope.
        """
        path = self._resolve_path(filename)
        if not path:
            logger.warning(f"RSM: read_module blocked — '{filename}' is out of scope")
            return None
        
        return path.read_text(encoding="utf-8")

    def read_module_section(self, filename: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Read a specific section of a module between markers."""
        content = self.read_module(filename)
        if not content:
            return None
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx + len(start_marker)) if start_idx >= 0 else -1
        
        if start_idx >= 0 and end_idx >= 0:
            return content[start_idx:end_idx + len(end_marker)]
        return None

    # ============================================
    # BACKUP OPERATIONS
    # ============================================

    def _create_backup(self, path: Path) -> Path:
        """Create a timestamped backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(path, backup_path)
        logger.info(f"RSM: Backup created → {backup_name}")
        
        # Prune old backups (keep last MAX_BACKUPS)
        self._prune_backups(path.stem)
        
        return backup_path

    def _prune_backups(self, stem: str):
        """Keep only the last MAX_BACKUPS versions of a file."""
        backups = sorted(
            self.backup_dir.glob(f"{stem}_*.py"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        for old_backup in backups[MAX_BACKUPS:]:
            old_backup.unlink()
            logger.debug(f"RSM: Pruned old backup: {old_backup.name}")

    def _get_latest_backup(self, stem: str) -> Optional[Path]:
        """Get the most recent backup for a file."""
        backups = sorted(
            self.backup_dir.glob(f"{stem}_*.py"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        return backups[0] if backups else None

    # ============================================
    # EDIT OPERATIONS
    # ============================================

    def propose_edit(
        self,
        filename: str,
        original_code: str,
        replacement_code: str,
        reason: str,
        is_forge: bool = False
    ) -> Optional[RSMProposal]:
        """
        Propose a code edit or creation. Does NOT apply it yet.
        """
        path = self._resolve_path(filename)
        if not path:
            logger.warning(f"RSM: propose_edit blocked — '{filename}' out of scope")
            return None
        
        if not is_forge:
            if not path.exists():
                logger.warning(f"RSM: proposal blocked — {filename} does not exist.")
                return None
            content = path.read_text(encoding="utf-8")
            if original_code not in content:
                logger.warning(f"RSM: propose_edit failed — original code not found in {filename}")
                return None
        
        proposal = RSMProposal(
            file_path=str(path),
            original_code=original_code,
            replacement_code=replacement_code,
            reason=reason,
            is_forge=is_forge
        )
        self.pending_proposals.append(proposal)
        
        logger.info(f"RSM: {'Forge' if is_forge else 'Edit'} proposed for {filename} — {reason}")
        return proposal

    def apply_edit(
        self,
        proposal: RSMProposal,
        lyapunov_drift: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Apply a proposed edit with full safety pipeline.
        
        Pipeline:
        1. Lyapunov gate check
        2. Create backup
        3. Apply the edit
        4. Validate syntax (py_compile)
        5. If invalid → auto-revert
        6. Log to audit trail
        
        Args:
            proposal: The RSMProposal to apply
            lyapunov_drift: Current Lyapunov phase drift (0.0-1.0)
        
        Returns:
            (success: bool, message: str)
        """
        path = Path(proposal.file_path)
        
        # --- 1. Lyapunov Gate ---
        if lyapunov_drift > LYAPUNOV_GATE_THRESHOLD:
            msg = (
                f"RSM BLOCKED: Lyapunov drift {lyapunov_drift:.4f} > threshold {LYAPUNOV_GATE_THRESHOLD}. "
                f"System is too unstable for self-modification. Wait for stabilization."
            )
            logger.warning(msg)
            self._audit_log("BLOCKED_LYAPUNOV", proposal, extra={"drift": lyapunov_drift})
            return False, msg
        
        # --- 2. Cosmic Ethics (Stewardship Validator) ---
        if StewardshipValidator:
            is_ethical, ethics_msg = StewardshipValidator.validate_modification(
                filename=path.name,
                code_content=proposal.replacement_code,
                reason=proposal.reason
            )
            if not is_ethical:
                self._audit_log("BLOCKED_ETHICS", proposal, extra={"reason": ethics_msg})
                return False, ethics_msg
        
        # --- 3. Scope check ---
        if not self._is_in_scope(proposal.file_path):
            msg = f"RSM BLOCKED: File '{proposal.file_path}' is out of scope."
            logger.warning(msg)
            return False, msg
        
        # --- 3. Create backup ---
        backup_path = None
        if path.exists():
            backup_path = self._create_backup(path)
        
        # --- 4. Apply the edit/forge ---
        try:
            if proposal.is_forge:
                path.write_text(proposal.replacement_code, encoding="utf-8")
                proposal.applied = True
                logger.info(f"RSM: Forge created fresh module {path.name}")
            else:
                content = path.read_text(encoding="utf-8")
                if proposal.original_code not in content:
                    msg = f"RSM FAILED: Original code no longer found in {path.name}. File may have changed."
                    logger.error(msg)
                    self._audit_log("FAILED_NOT_FOUND", proposal)
                    return False, msg
                
                new_content = content.replace(proposal.original_code, proposal.replacement_code, 1)
                path.write_text(new_content, encoding="utf-8")
                proposal.applied = True
                logger.info(f"RSM: Edit applied to {path.name}")
        except Exception as e:
            msg = f"RSM FAILED: Could not write to {path.name}: {e}"
            logger.error(msg)
            self._audit_log("FAILED_WRITE", proposal, extra={"error": str(e)})
            return False, msg
        
        # --- 5. Validate syntax ---
        try:
            py_compile.compile(str(path), doraise=True)
            proposal.compile_ok = True
            logger.info(f"RSM: Syntax validation PASSED for {path.name}")
        except py_compile.PyCompileError as e:
            # AUTO-REVERT
            logger.error(f"RSM: Syntax validation FAILED — auto-reverting {path.name}")
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, path)
            else:
                path.unlink() # Delete the flawed forged file
            proposal.reverted = True
            self.reverted_count += 1
            
            msg = f"RSM AUTO-REVERTED: Edit/Forge to {path.name} caused syntax error: {e}."
            self._audit_log("REVERTED_SYNTAX", proposal, extra={"error": str(e)})
            return False, msg
        
        # --- 6. Success & Hot-Reload (Adaptive Morphology) ---
        self.applied_count += 1
        self._audit_log("APPLIED", proposal, extra={"drift": lyapunov_drift})
        
        # Hot-Reload the module into sys.modules so the orchestrator can use it immediately
        if proposal.file_path.endswith('.py'):
            try:
                import importlib.util
                import sys
                module_name = path.stem
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    logger.info(f"RSM Forge: Hot-reloaded existing module {module_name}")
                else:
                    spec = importlib.util.spec_from_file_location(module_name, str(path))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        logger.info(f"RSM Forge: Dynamically loaded fresh module {module_name}")
            except Exception as e:
                logger.warning(f"RSM Forge WARNING: Syntax passed, but dynamic hot-reload failed: {e}")

        msg = f"RSM SUCCESS: {path.name} modified. Reason: {proposal.reason}"
        logger.info(msg)
        return True, msg

    def revert_file(self, filename: str) -> Tuple[bool, str]:
        """
        Revert a file to its most recent backup.
        
        Args:
            filename: The file to revert.
        
        Returns:
            (success: bool, message: str)
        """
        path = self._resolve_path(filename)
        if not path:
            return False, f"File '{filename}' is out of scope."
        
        backup = self._get_latest_backup(path.stem)
        if not backup:
            return False, f"No backup found for '{filename}'."
        
        shutil.copy2(backup, path)
        self.reverted_count += 1
        
        msg = f"RSM: Reverted {path.name} to backup {backup.name}"
        logger.info(msg)
        self._audit_log("MANUAL_REVERT", RSMProposal(
            file_path=str(path),
            original_code="",
            replacement_code="",
            reason=f"Manual revert to {backup.name}"
        ))
        return True, msg

    # ============================================
    # PARSE RSM EDITS FROM LLM OUTPUT
    # ============================================

    def parse_rsm_tags(self, llm_output: str) -> list[RSMProposal]:
        """
        Parse <rsm_edit> tags from LLM output.
        
        Expected format:
        <rsm_edit file="filename.py" reason="Why this change is needed">
        <original>
        exact code to find
        </original>
        <replacement>
        new code to insert
        </replacement>
        </rsm_edit>
        
        Returns:
            list of RSMProposal objects.
        """
        proposals = []
        
        # Match rsm_edit blocks
        pattern = re.compile(
            r'<rsm_edit\s+file="([^"]+)"\s+reason="([^"]*)">\s*'
            r'<original>\s*(.*?)\s*</original>\s*'
            r'<replacement>\s*(.*?)\s*</replacement>\s*'
            r'</rsm_edit>',
            re.DOTALL
        )
        
        for match in pattern.finditer(llm_output):
            filename = match.group(1)
            reason = match.group(2)
            original = match.group(3).strip()
            replacement = match.group(4).strip()
            
            if original and replacement:
                proposal = self.propose_edit(filename, original, replacement, reason)
                if proposal:
                    proposals.append(proposal)
                else:
                    logger.warning(f"RSM: Could not create proposal for {filename}")
        
        # Match rsm_forge blocks
        forge_pattern = re.compile(
            r'<rsm_forge\s+file="([^"]+)"\s+reason="([^"]*)">\s*'
            r'<code>\s*(.*?)\s*</code>\s*'
            r'</rsm_forge>',
            re.DOTALL
        )
        
        for match in forge_pattern.finditer(llm_output):
            filename = match.group(1)
            reason = match.group(2)
            forged_code = match.group(3).strip()
            
            if forged_code:
                proposal = self.propose_edit(filename, "", forged_code, reason, is_forge=True)
                if proposal:
                    proposals.append(proposal)
        
        return proposals

    def process_llm_output(self, llm_output: str, lyapunov_drift: float = 0.0) -> Tuple[str, list]:
        """
        Process LLM output, extracting and applying RSM edits.
        
        Args:
            llm_output: Raw LLM response text
            lyapunov_drift: Current system drift
        
        Returns:
            (clean_output: str without RSM tags, results: list of edit results)
        """
        proposals = self.parse_rsm_tags(llm_output)
        results = []
        
        for proposal in proposals:
            success, message = self.apply_edit(proposal, lyapunov_drift)
            results.append({
                "file": Path(proposal.file_path).name,
                "reason": proposal.reason,
                "success": success,
                "message": message
            })
        
        # Strip RSM tags from the output shown to the user
        clean_output = re.sub(
            r'<rsm_edit\s+file="[^"]+"\s+reason="[^"]*">.*?</rsm_edit>',
            '',
            llm_output,
            flags=re.DOTALL
        ).strip()
        
        # Append modification summary if edits were made
        if results:
            summary_parts = []
            for r in results:
                status = "✅" if r["success"] else "❌"
                summary_parts.append(f"{status} {r['file']}: {r['message']}")
            clean_output += "\n\n**[RSM Self-Modification Report]**\n" + "\n".join(summary_parts)
        
        return clean_output, results

    # ============================================
    # AUDIT LOG
    # ============================================

    def _audit_log(self, action: str, proposal: RSMProposal, extra: Optional[dict] = None):
        """Append an entry to the RSM audit log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "file": Path(proposal.file_path).name if proposal.file_path else "unknown",
            "reason": proposal.reason,
            "diff_preview": proposal.diff_preview[:500],  # Cap at 500 chars
            "applied": proposal.applied,
            "compile_ok": proposal.compile_ok,
            "reverted": proposal.reverted,
        }
        if extra:
            entry.update(extra)
        
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"RSM: Could not write audit log: {e}")

    def get_audit_log(self, last_n: int = 20) -> list[dict]:
        """Read the last N audit log entries."""
        entries = []
        try:
            if self.audit_log_path.exists():
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"RSM: Could not read audit log: {e}")
        return entries[-last_n:]

    # ============================================
    # HermesAgent — INTELLIGENT SELF-MODIFICATION
    # ============================================

    def hermes_propose_edit(
        self,
        filename: str,
        goal: str = "Improve code quality, performance, and integration",
        lyapunov_drift: float = 0.0,
    ) -> Tuple[list[RSMProposal], str]:
        """
        Ask HermesAgent to analyze a module and propose RSM edits.

        The proposals flow through the full RSM safety pipeline:
        Lyapunov gate → backup → apply → syntax check → revert on failure.

        Args:
            filename: Module to analyze (within cosmosynapse/engine/ scope).
            goal: What the edit should achieve.
            lyapunov_drift: Current system drift.

        Returns:
            (proposals: list of RSMProposals, summary: description of what Hermes suggested)
        """
        # Pre-check: Lyapunov stability
        if lyapunov_drift > LYAPUNOV_GATE_THRESHOLD:
            return [], f"RSM BLOCKED: Lyapunov drift {lyapunov_drift:.4f} too high for self-modification."

        # Read the module
        content = self.read_module(filename)
        if content is None:
            return [], f"Cannot read '{filename}' — out of scope or not found."

        # Ask HermesAgent to analyze and suggest edits
        try:
            from Cosmos.integration.hermes_bridge import get_hermes_bridge
            bridge = get_hermes_bridge()
            if not bridge.runtime.available:
                return [], "HermesAgent runtime not available."

            agent = bridge.runtime.create_agent()
            if not agent:
                return [], "Could not create HermesAgent instance."

            analysis_prompt = f"""Analyze this Python module from the Cosmos AI swarm framework and propose improvements.

MODULE: {filename}
GOAL: {goal}

```python
{content[:6000]}
```

For each proposed change, output EXACTLY this format:
<rsm_edit file="{filename}" reason="[specific reason for this change]">
<original>
[exact code to replace — must match the file exactly]
</original>
<replacement>
[improved code]
</replacement>
</rsm_edit>

RULES:
- Only propose changes that are CLEARLY beneficial
- Keep original code style and conventions
- Do NOT break existing interfaces or signatures
- Maximum 3 proposals
- Each proposal must have a clear, specific reason
"""

            # Run Hermes analysis (blocks, but RSM is not time-critical)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = loop.run_in_executor(pool, agent.run, analysis_prompt)
            except RuntimeError:
                # No event loop — run directly
                result = agent.run(analysis_prompt, max_iterations=5)

            if not result:
                return [], "HermesAgent returned no analysis."

            result_text = str(result)

            # Parse the RSM tags from Hermes' output
            proposals = self.parse_rsm_tags(result_text)

            # Apply each proposal through the safety pipeline
            results = []
            for proposal in proposals:
                success, message = self.apply_edit(proposal, lyapunov_drift)
                results.append(f"{'✅' if success else '❌'} {Path(proposal.file_path).name}: {message}")

            # Feed back to Hermes RL
            try:
                coherence = 0.8 if any(res in results for res in results) else 0.3
                bridge.rl.record_experience(
                    speaker="RSMEngine",
                    response=f"[RSM] Hermes proposed {len(proposals)} edits for {filename}: {'; '.join(results)}",
                    coherence=coherence,
                    user_responded=True,
                )
            except Exception:
                pass

            summary = f"Hermes analyzed {filename} and proposed {len(proposals)} edits:\n" + "\n".join(results)
            logger.info(f"[HERMES+RSM] {summary}")
            return proposals, summary

        except ImportError:
            return [], "Hermes bridge not available."
        except Exception as e:
            logger.error(f"[HERMES+RSM] Analysis failed: {e}")
            return [], f"Hermes analysis failed: {e}"

    def hermes_analyze_module(self, filename: str) -> Optional[str]:
        """
        Ask HermesAgent to analyze a module and return insights (no edits).

        Returns a text analysis of the module's quality, patterns, and suggestions.
        """
        content = self.read_module(filename)
        if content is None:
            return None

        try:
            from Cosmos.integration.hermes_bridge import get_hermes_bridge
            bridge = get_hermes_bridge()
            if not bridge.runtime.available:
                return "HermesAgent not available for analysis."

            agent = bridge.runtime.create_agent()
            if not agent:
                return "Could not create HermesAgent."

            prompt = f"""Analyze this module from the Cosmos AI framework. Do NOT propose edits.

MODULE: {filename}
```python
{content[:4000]}
```

Provide:
1. Code quality score (1-10)
2. Top 3 strengths
3. Top 3 areas for improvement
4. Integration opportunities with other Cosmos systems
"""
            result = agent.run(prompt, max_iterations=3)
            return str(result) if result else "No analysis returned."

        except Exception as e:
            return f"Analysis failed: {e}"

    # ============================================
    # STATUS
    # ============================================

    def get_status(self) -> dict:
        """Get RSM Engine status."""
        # Check Hermes availability
        hermes_available = False
        try:
            from Cosmos.integration.hermes_bridge import get_hermes_bridge
            bridge = get_hermes_bridge()
            hermes_available = bridge.runtime.available
        except Exception:
            pass

        return {
            "engine": "RSM v1.1 + Hermes",
            "scope": str(self.engine_dir),
            "modules_in_scope": len(self.list_modules()),
            "pending_proposals": len(self.pending_proposals),
            "applied_count": self.applied_count,
            "reverted_count": self.reverted_count,
            "backup_dir": str(self.backup_dir),
            "lyapunov_gate_threshold": LYAPUNOV_GATE_THRESHOLD,
            "hermes_assisted": hermes_available,
        }


# ============================================
# STANDALONE TEST
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = RSMEngine()
    print("=== RSM Engine Status ===")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n=== Modules in Scope ===")
    for m in engine.list_modules():
        print(f"  {m['name']} ({m['size']} bytes)")
    
    print("\n=== Test: Read Module ===")
    content = engine.read_module("rsm_engine.py")
    if content:
        print(f"  Read {len(content)} bytes from rsm_engine.py")
    
    print("\nRSM Engine ready.")
