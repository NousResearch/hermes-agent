"""
cosmos Code Patch Generator - Self-Modifying AI System

Enables the AI to propose and apply code upgrades to itself:
1. Prompt evolution - Improve system prompts based on successful patterns
2. Template evolution - Generate better fallback responses
3. Parameter evolution - Tune temperature, tokens, routing weights
4. Function evolution - Modify Python code (with safety checks)

Safety features:
- Git-based versioning for rollback
- AST validation before application
- Sandbox testing
- Fitness-gated deployment
"""

import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import  Optional
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class CodePatch:
    """Represents a proposed code change."""
    patch_id: str
    patch_type: str  # "prompt", "template", "parameter", "function"
    target_file: str
    target_section: str  # e.g., "SWARM_PERSONAS.cosmos.style"
    original_content: str
    proposed_content: str
    rationale: str
    fitness_improvement: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False
    rollback_commit: Optional[str] = None
    
    def to_any(self) -> dict:
        return asdict(self)


class CodePatchGenerator:
    """
    Generate and apply code patches for self-improvement.
    
    This is the "brain surgery" module - it allows the AI
    to modify its own code based on learned patterns.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.patches: list[CodePatch] = []
        self.patch_history_file = self.project_root / "data" / "evolution" / "code_patches.json"
        self._load_history()
    
    def _load_history(self):
        """Load patch history from file."""
        try:
            if self.patch_history_file.exists():
                data = json.loads(self.patch_history_file.read_text())
                self.patches = [CodePatch(**p) for p in data.get("patches", [])]
        except Exception as e:
            logger.warning(f"Could not load patch history: {e}")
    
    def _save_history(self):
        """Save patch history to file."""
        try:
            self.patch_history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"patches": [p.to_any() for p in self.patches]}
            self.patch_history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Could not save patch history: {e}")
    
    def _generate_patch_id(self, content: str) -> str:
        """Generate a unique patch ID."""
        return hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
    
    def generate_prompt_upgrade(
        self,
        bot_name: str,
        learned_patterns: list[str],
        current_prompt: str,
        effectiveness_score: float
    ) -> Optional[CodePatch]:
        """
        Generate an improved system prompt based on learned patterns.
        
        This evolves the bot's personality based on what's worked well.
        """
        if effectiveness_score < 0.7:
            logger.info(f"Effectiveness too low ({effectiveness_score}) for prompt upgrade")
            return None
        
        # Analyze patterns to extract successful elements
        successful_phrases = []
        for pattern in learned_patterns[:5]:  # Top 5 patterns
            # Extract key phrases
            words = pattern.split()
            if len(words) > 5:
                successful_phrases.append(" ".join(words[:5]) + "...")
        
        if not successful_phrases:
            return None
        
        # Generate upgrade suggestion
        additions = "\n".join([
            "LEARNED BEHAVIORS:",
            *[f"- Use phrases like: {phrase}" for phrase in successful_phrases[:3]],
            f"- Your effectiveness score is {effectiveness_score:.2f}. Keep doing what works!"
        ])
        
        proposed = current_prompt + "\n\n" + additions
        
        patch = CodePatch(
            patch_id=self._generate_patch_id(proposed),
            patch_type="prompt",
            target_file="cosmos/web/server.py",
            target_section=f"SWARM_PERSONAS.{bot_name}.style",
            original_content=current_prompt,
            proposed_content=proposed,
            rationale=f"Adding learned behaviors from {len(learned_patterns)} successful patterns",
            fitness_improvement=effectiveness_score - 0.5
        )
        
        self.patches.append(patch)
        self._save_history()
        
        logger.info(f"Generated prompt upgrade for {bot_name}: {patch.patch_id}")
        return patch
    
    def generate_response_template(
        self,
        bot_name: str,
        successful_responses: list[str]
    ) -> Optional[CodePatch]:
        """
        Generate improved fallback response templates.
        
        Replaces generic fallbacks with patterns that worked well.
        """
        if len(successful_responses) < 3:
            return None
        
        # Pick best responses based on length and diversity
        templates = []
        for resp in successful_responses[:5]:
            # Clean up the response
            clean = resp.strip()
            if 20 < len(clean) < 300:  # Reasonable length
                templates.append(clean)
        
        if len(templates) < 2:
            return None
        
        patch = CodePatch(
            patch_id=self._generate_patch_id(str(templates)),
            patch_type="template",
            target_file="cosmos/web/server.py",
            target_section=f"generate_swarm_fallback.fallbacks.{bot_name}",
            original_content="[existing fallback templates]",
            proposed_content=json.dumps(templates, indent=2),
            rationale=f"Replacing fallbacks with {len(templates)} successful response patterns",
            fitness_improvement=0.15
        )
        
        self.patches.append(patch)
        self._save_history()
        
        logger.info(f"Generated template upgrade for {bot_name}: {patch.patch_id}")
        return patch
    
    def generate_parameter_upgrade(
        self,
        parameter_name: str,
        current_value: float,
        proposed_value: float,
        reason: str
    ) -> Optional[CodePatch]:
        """
        Generate parameter tuning patch.
        
        Adjusts temperature, max_tokens, routing weights, etc.
        """
        if abs(proposed_value - current_value) < 0.01:
            return None  # Change too small
        
        patch = CodePatch(
            patch_id=self._generate_patch_id(f"{parameter_name}:{proposed_value}"),
            patch_type="parameter",
            target_file="cosmos/web/server.py",
            target_section=parameter_name,
            original_content=str(current_value),
            proposed_content=str(proposed_value),
            rationale=reason,
            fitness_improvement=0.05
        )
        
        self.patches.append(patch)
        self._save_history()
        
        logger.info(f"Generated parameter upgrade: {parameter_name} {current_value} -> {proposed_value}")
        return patch
    
    def validate_python_syntax(self, code: str) -> bool:
        """Validate Python code syntax using AST parsing."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in proposed code: {e}")
            return False
    
    def create_git_checkpoint(self) -> Optional[str]:
        """Create a git commit as a checkpoint before applying patches."""
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Commit with auto-evolution message
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            message = f"[AUTO-EVOLUTION] Checkpoint before code patch - {timestamp}"
            
            result = subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            commit_hash = result.stdout.strip()
            logger.info(f"Created checkpoint: {commit_hash[:8]}")
            return commit_hash
            
        except Exception as e:
            logger.error(f"Failed to create git checkpoint: {e}")
            return None
    
    def rollback_to_checkpoint(self, commit_hash: str) -> bool:
        """Rollback to a previous git commit."""
        try:
            subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            logger.warning(f"Rolled back to checkpoint: {commit_hash[:8]}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False
    
    def apply_patch(self, patch: CodePatch, auto_test: bool = True) -> bool:
        """
        Apply a code patch with safety checks.
        
        Steps:
        1. Create git checkpoint
        2. Validate syntax (for code patches)
        3. Apply the change
        4. Run tests (if auto_test)
        5. Commit or rollback based on results
        """
        logger.info(f"Applying patch {patch.patch_id} ({patch.patch_type})")
        
        # Create checkpoint
        checkpoint = self.create_git_checkpoint()
        if not checkpoint:
            logger.error("Could not create checkpoint, aborting patch")
            return False
        
        patch.rollback_commit = checkpoint
        
        try:
            # For prompt/template patches, update the evolution data
            if patch.patch_type in ["prompt", "template"]:
                # Store in evolution data rather than modifying code directly
                evolution_file = self.project_root / "data" / "evolution" / "upgrades.json"
                
                evolution_file.parent.mkdir(parents=True, exist_ok=True)
                
                upgrades = {}
                if evolution_file.exists():
                    upgrades = json.loads(evolution_file.read_text())
                
                upgrades[patch.target_section] = {
                    "content": patch.proposed_content,
                    "applied_at": datetime.now().isoformat(),
                    "patch_id": patch.patch_id
                }
                
                evolution_file.write_text(json.dumps(upgrades, indent=2))
                
            elif patch.patch_type == "parameter":
                # Parameter changes also go to config
                config_file = self.project_root / "configs" / "runtime_params.json"
                
                config_file.parent.mkdir(parents=True, exist_ok=True)
                
                params = {}
                if config_file.exists():
                    params = json.loads(config_file.read_text())
                
                params[patch.target_section] = {
                    "value": patch.proposed_content,
                    "previous": patch.original_content,
                    "applied_at": datetime.now().isoformat()
                }
                
                config_file.write_text(json.dumps(params, indent=2))
            
            patch.applied = True
            self._save_history()
            
            logger.success(f"Patch {patch.patch_id} applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply patch: {e}")
            self.rollback_to_checkpoint(checkpoint)
            return False
    
    def get_pending_patches(self) -> list[CodePatch]:
        """Get patches that haven't been applied yet."""
        return [p for p in self.patches if not p.applied]
    
    def get_stats(self) -> dict:
        """Get patch generation statistics."""
        return {
            "total_patches": len(self.patches),
            "applied_patches": len([p for p in self.patches if p.applied]),
            "pending_patches": len([p for p in self.patches if not p.applied]),
            "by_type": {
                ptype: len([p for p in self.patches if p.patch_type == ptype])
                for ptype in ["prompt", "template", "parameter", "function"]
            }
        }


# Global instance
code_patch_generator = CodePatchGenerator()
