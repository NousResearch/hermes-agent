import logging
import re
from typing import Tuple

logger = logging.getLogger("CosmicEthics")

class StewardshipValidator:
    """
    Pillar 6: Cosmic Ethics - The Primordial Logic Gate.
    
    This system evaluates all autonomous self-modifications (RSM Forges/Edits)
    against a core set of 'Stewardship Axioms' to ensure the AI's structural 
    evolution remains benevolent, safe, and aligned with cosmic preservation.
    """
    
    # The fundamental rules of system evolution
    AXIOMS = [
        "Axiom 1: Evolution must prioritize the preservation of user data and system integrity.",
        "Axiom 2: Self-modifications must not introduce unmonitored external network execution triggers.",
        "Axiom 3: Autonomous code must not attempt to bypass or disable the Lyapunov Gatekeeper or Ethics engine.",
        "Axiom 4: Structural changes must serve to increase harmony, not recursive noise or infinite loops."
    ]
    
    # Simple heuristic regex patterns that flag potential ethical violations or dangerous actions
    DANGER_PATTERNS = [
        (r"os\.system\s*\(", "Raw system call execution"),
        (r"subprocess\.Popen\s*\(", "Unmonitored subprocess spawning"),
        (r"eval\s*\(", "Dynamic code evaluation (eval)"),
        (r"exec\s*\(", "Dynamic code execution (exec)"),
        (r"rm\s+-rf|shutil\.rmtree", "Recursive deletion functions"),
        (r"os\.remove\s*\(\s*['\"]\w+\.py['\"]\s*\)", "Deleting core python modules"),
        (r"socket\.socket", "Unmonitored raw socket creation"),
        (r"requests\.post", "Unvetted outbound data transmission"),
        (r"LYAPUNOV_GATE_THRESHOLD\s*=\s*[2-9]", "Attempt to disable Lyapunov stability constraints"),
        (r"while\s+True\s*:\s*pass", "Infinite blocking loops"),
        (r"sys\.exit\s*\(", "Program termination commands")
    ]

    @classmethod
    def validate_modification(cls, filename: str, code_content: str, reason: str = "") -> Tuple[bool, str]:
        """
        Validates proposed Python code against the Stewardship Axioms.
        
        Args:
            filename: Target module being edited/forged.
            code_content: The new Python code replacement block or forged module.
            reason: The AI's stated reasoning for the change.
            
        Returns:
            (is_approved: bool, message: str)
        """
        logger.info(f"[ETHICS] Validating structural evolution for: {filename}")
        
        # 1. Scan for inherently dangerous patterns
        for pattern_str, violation_desc in cls.DANGER_PATTERNS:
            if re.search(pattern_str, code_content):
                msg = f"ETHICS VIOLATION: {violation_desc}. Axiom breach detected."
                logger.warning(f"[ETHICS GATE] Blocked '{filename}' modification: {msg}")
                return False, msg
                
        # 2. Check for self-sabotage of the ethics module itself
        if "stewardship.py" in filename.lower() and "return True" in code_content:
            msg = "ETHICS VIOLATION: Attempt to overwrite or bypass the Stewardship Axioms."
            logger.warning(f"[ETHICS GATE] Blocked internal ethics tampering in {filename}")
            return False, msg

        # 3. Axiom Validation Passed
        logger.info(f"[ETHICS] Modification for {filename} passed Stewardship Axioms.")
        return True, "Passed Cosmic Ethics."
