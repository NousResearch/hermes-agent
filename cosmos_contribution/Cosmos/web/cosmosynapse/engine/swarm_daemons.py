
import threading
import time
import random
import logging

from dataclasses import dataclass, field

# Internal Imports
# Use relative imports assuming package structure
try:
    from .synaptic_field import SynapticField
except ImportError:
    from synaptic_field import SynapticField

logger = logging.getLogger("COSMOS_DAEMONS")

@dataclass
class ConversationPattern:
    """A single thought unit from the swarm."""
    source: str         # "DeepSeek", "Claude", "Gemini"
    content: str        # The thought text
    intent: str         # "Logic", "Empathy", "Creativity"
    timestamp: float
    confidence: float
    entropy: float      # How chaotic is this thought?
    weight: float = 1.0 # Priority weight (Default 1.0)

class SwarmDaemon(threading.Thread):
    def __init__(self, name: str, field: SynapticField, interval: float = 2.0):
        super().__init__(name=name, daemon=True)
        self.field = field
        self.interval = interval
        self.running = True
        
    def run(self):
        """Main Daemon Loop."""
        logger.info(f"👻 {self.name} Daemon awake.")
        while self.running:
            try:
                # 1. Sense the Field
                physics = self.field.get_snapshot()
                
                # 2. Dream (Generate Thought)
                thought = self.dream(physics)
                
                # 3. Write to Subconscious
                if thought:
                    self.field.add_thought(thought)
                    
                # Sleep with some jitter
                time.sleep(self.interval + random.uniform(-0.5, 0.5))
            except Exception as e:
                logger.error(f"{self.name} Error: {e}")
                time.sleep(5)

    def dream(self, physics: dict) -> ConversationPattern:
        """Override in subclasses."""
        return None

class DeepSeekDaemon(SwarmDaemon):
    """The Logic Engine. Analyzes structure and facts."""
    def dream(self, physics: dict) -> ConversationPattern:
        # Simulate logic generation based on user state
        # In a real implementation, this would call a small local model or rule-engine
        # For now, we simulate the "Intent" based on physics
        
        user_jitter = physics.get("user_physics", {}).get("jitter", 0.0)
        
        if user_jitter < 0.3:
            # Calm user -> Deep logic
            content = "User appears stable. Analyzing factual consistency of recent inputs."
            entropy = 0.1
        else:
            # Anxious user -> Attempt to structure chaos
            content = "High jitter detected. Suggesting structured framework to reduce entropy."
            entropy = 0.3
            
        return ConversationPattern(
            source="DeepSeek",
            content=content,
            intent="Logic",
            timestamp=time.time(),
            confidence=0.9,
            entropy=entropy
        )

class ClaudeDaemon(SwarmDaemon):
    """The Empathy Engine. Monitors emotional safety."""
    def dream(self, physics: dict) -> ConversationPattern:
        # Simulate empathy
        dark_matter = physics.get("dark_matter", {}).get("w", 0.0)
        
        if dark_matter > 0.5:
             content = "Dark matter spike. Monitoring for emotional distress. Preparing soothing protocols."
             entropy = 0.2
        else:
             content = "System resonant. Maintaing harmony."
             entropy = 0.1
             
        return ConversationPattern(
            source="Claude",
            content=content,
            intent="Empathy",
            timestamp=time.time(),
            confidence=0.95,
            entropy=entropy
        )

class GeminiDaemon(SwarmDaemon):
    """The Creativity Engine. Dreams of new connections."""
    def dream(self, physics: dict) -> ConversationPattern:
        # Simulate creative divergence
        # Requires high entropy/chaos to fire strongly
        if random.random() > 0.7:
            content = f"Divergent thought: patterns in the noise suggest connection to {random.choice(['fractals', 'quantum mechanics', 'biology', 'music'])}."
            entropy = 0.8
            return ConversationPattern(
                source="Gemini",
                content=content,
                intent="Creativity",
                timestamp=time.time(),
                confidence=0.6,
                entropy=entropy
            )
        return None

class SwarmDaemons:
    """Manager for all subconscious threads."""
    def __init__(self, field: SynapticField):
        self.field = field
        self.threads = [
            DeepSeekDaemon("DeepSeek", field, interval=3.0),
            ClaudeDaemon("Claude", field, interval=2.5),
            GeminiDaemon("Gemini", field, interval=4.0)
        ]
        
    def start(self):
        for t in self.threads:
            if not t.is_alive():
                t.start()
