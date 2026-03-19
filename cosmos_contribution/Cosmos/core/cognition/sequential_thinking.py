"""
cosmos Sequential Thinking - Systematic Reasoning Tool.

"Thinking about thinking is the only way to ensure we're actually thinking."

A tool for the agent to break down complex problems into a sequence of steps,
verifying each one before moving forward.
"""


from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ThoughtStep:
    step_number: int
    content: str
    verification: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class SequentialThinking:
    def __init__(self):
        self.current_chain: list[ThoughtStep] = []

    def start_new_chain(self):
        self.current_chain = []

    def add_step(self, content: str, verification: str = "") -> ThoughtStep:
        step = ThoughtStep(
            step_number=len(self.current_chain) + 1,
            content=content,
            verification=verification
        )
        self.current_chain.append(step)
        return step

    def get_summary(self) -> str:
        if not self.current_chain:
            return "No reasoning chain in progress."
        
        summary = ["### Sequential Reasoning Chain:"]
        for step in self.current_chain:
            summary.append(f"{step.step_number}. **{step.content}**")
            if step.verification:
                summary.append(f"   *Verification:* {step.verification}")
        
        return "\n".join(summary)

sequential_thinker = SequentialThinking()
