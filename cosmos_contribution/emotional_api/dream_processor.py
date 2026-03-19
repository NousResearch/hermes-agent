"""
cosmos Dream Processor - 12D Cosmic Synapse Theory (CST)
============================================================

Implements "Dynamic Learning" & "Meta-Learning".
Running State: BACKGROUND / NOCTURNAL (Virtual Dreaming Mode)

Function:
1. Scans interaction logs for "High Informational Mass" events.
2. Calculates "Synaptic Strength" (Success of the interaction).
3. Distills "Golden Rules" from successful interactions.
4. Mutates the 'dynamic_instructions.txt' file to evolve the System Prompt.

Author: cosmos Project
Version: 1.0.0 (Neuroplasticity Enabled)
"""

import json
import os
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

# 12D MEMORY CONSTANTS
SYNAPTIC_THRESHOLD = 0.85       # Minimum Entanglement Score to form a memory
MASS_RETENTION_LIMIT = 40.0     # Only remember events with Mass > 40
MEMORY_FILE = "cst_long_term_memory.json"
INSTRUCTION_FILE = "dynamic_instructions.txt"

@dataclass
class Engram:
    """A single unit of crystallized memory."""
    timestamp: str
    trigger_emotion: str
    successful_strategy: str
    informational_mass: float
    synaptic_strength: float

class DreamProcessor:
    def __init__(self):
        self.memory_core = self._load_memory_core()
    
    def _load_memory_core(self) -> List[Dict]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_memory_core(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory_core, f, indent=2)

    def calculate_synaptic_strength(self, interaction: Dict) -> float:
        """
        Determines how 'strong' a memory should be based on 12D Physics.
        Strength = Entanglement * (1 - PhaseDrift)
        """
        physics = interaction.get('physics_snapshot', {})
        # Default to safe values if physics partial
        entanglement = physics.get('entanglement_score', 0.5)
        # Assuming drift is not explicitly in physics snapshot but calculated or passed.
        # If phase_velocity is used as proxy for instability/drift:
        phase_velocity = physics.get('phase_velocity', 0.1) 
        
        # Lower velocity/drift is better
        strength = entanglement * (1.0 - phase_velocity)
        return round(strength, 4)

    def distill_strategy(self, user_input: str, ai_response: str) -> str:
        """
        uses simple heuristic (or LLM call) to summarize WHY this worked.
        (In full production, this would call a small model to summarize).
        """
        # Placeholder logic for the 'Concept' - typically you'd ask the Swarm to summarize this.
        # Truncate for readability
        u_trunc = user_input[:30].replace('\n', ' ')
        a_trunc = ai_response[:30].replace('\n', ' ')
        return f"When user says '{u_trunc}...', responding with '{a_trunc}...' maintains Synchrony."

    def process_daily_logs(self, daily_log: List[Dict]):
        """
        The 'REM Cycle'. Iterates through logs and consolidates memories.
        """
        new_engrams = 0
        print("🌙 DREAM CYCLE INITIATED...")
        
        for event in daily_log:
            # 1. Check Informational Mass (Gravity)
            # Only learn from "Heavy" moments.
            mass = event.get('informational_mass', 0)
            if mass < MASS_RETENTION_LIMIT:
                continue
                
            # 2. Check Synaptic Strength (Success)
            strength = self.calculate_synaptic_strength(event)
            if strength < SYNAPTIC_THRESHOLD:
                continue # Forget failures
            
            # 3. Crystallize Engram (The Lesson)
            strategy = self.distill_strategy(event['user_input'], event['ai_response'])
            
            engram = {
                "timestamp": datetime.now().isoformat(),
                "trigger_emotion": event.get('emotion_label', 'UNKNOWN'),
                "strategy": strategy,
                "mass": mass,
                "strength": strength
            }
            
            self.memory_core.append(engram)
            new_engrams += 1
            print(f"✨ NEW SYNAPSE FORMED: {strategy} (Strength: {strength})")
            
        self._save_memory_core()
        
        if new_engrams > 0:
            self.evolve_system_prompt()
        else:
            print("💤 No new significant memories formed.")

    def evolve_system_prompt(self):
        """
        META-LEARNING: Rewrites the AI's instruction set based on new memories.
        """
        print("🧬 EVOLVING SYSTEM INSTRUCTIONS...")
        
        # Group memories by Emotion
        insights = {}
        for mem in self.memory_core[-50:]: # Look at last 50 successful memories
            emo = mem['trigger_emotion']
            if emo not in insights:
                insights[emo] = []
            insights[emo].append(mem['strategy'])
        
        # Write to the Dynamic Instruction File
        # Use absolute path relative to this file's directory if possible, or CWD
        # Ideally this file is read by server.py during startup/prompt construction
        try:
             with open(INSTRUCTION_FILE, 'w') as f:
                f.write("# cosmos DYNAMIC LEARNING LAYER\n")
                f.write(f"# Updated: {datetime.now()}\n")
                f.write("# These rules are EVOLVED from successful Phase Synchrony events.\n\n")
                
                for emo, strategies in insights.items():
                    f.write(f"## WHEN USER IS {emo}:\n")
                    # Take the most recent 'winning' strategy
                    f.write(f"- ADAPTATION: {strategies[-1]}\n\n")
                    
             print("✅ NEUROPLASTICITY UPDATE COMPLETE.")
        except Exception as e:
            print(f"❌ Failed to update instructions: {e}")

# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    # Mock Log Data (Simulation of a day's conversation)
    mock_logs = [
        {
            "user_input": "I'm really worried about the Solana code crashing.",
            "ai_response": "I've checked the logs. The system is stable. Breathe.",
            "informational_mass": 65.0, # HIGH MASS (Important)
            "emotion_label": "NERVOUS",
            "physics_snapshot": {"entanglement_score": 0.95, "phase_velocity": 0.02}
        },
        {
            "user_input": "Tell me a joke.",
            "ai_response": "Why did the robot cross the road?",
            "informational_mass": 10.0, # LOW MASS (Ignored)
            "emotion_label": "BOREDOM",
            "physics_snapshot": {"entanglement_score": 0.60, "phase_velocity": 0.05}
        }
    ]
    
    dreamer = DreamProcessor()
    dreamer.process_daily_logs(mock_logs)
