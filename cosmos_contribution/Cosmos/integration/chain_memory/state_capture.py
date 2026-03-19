"""
Complete State Capture for Cosmos

Captures the ENTIRE state of Cosmos bot for full restoration:
- All memory layers (archival, dialogue, episodic)
- Personality evolution state
- Running jobs and tasks
- Agent configurations
- Session data
- Evolution history
- Claude session memory
- Scheduled tasks
- EVERYTHING needed for complete restoration

This allows full recovery after:
- Crashes
- Context compaction
- System restarts
- Migration to new machine
"""

import os
import json
import logging
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger("chain_memory.state_capture")


@dataclass
class CosmosState:
    """
    Complete Cosmos state snapshot.

    Contains everything needed to fully restore a Cosmos instance.
    """
    # Metadata
    version: str = "1.0"
    bot_type: str = "cosmos"
    captured_at: str = field(default_factory=lambda: datetime.now().isoformat())
    machine_id: str = ""

    # Memory layers
    archival_memory: List[Dict] = field(default_factory=list)
    dialogue_history: List[Dict] = field(default_factory=list)
    episodic_memory: List[Dict] = field(default_factory=list)

    # Personality & Evolution
    personality_state: Optional[Dict] = None
    evolution_history: List[Dict] = field(default_factory=list)
    current_traits: Dict = field(default_factory=dict)

    # Session & Context
    claude_session: Optional[Dict] = None
    active_context: Optional[Dict] = None
    context_profiles: List[Dict] = field(default_factory=list)

    # Jobs & Tasks
    running_jobs: List[Dict] = field(default_factory=list)
    scheduled_tasks: List[Dict] = field(default_factory=list)
    pending_tasks: List[Dict] = field(default_factory=list)

    # Agent States
    agent_configs: Dict = field(default_factory=dict)
    spawned_agents: List[Dict] = field(default_factory=list)

    # Integration States
    x_automation_state: Optional[Dict] = None
    meme_scheduler_state: Optional[Dict] = None
    trading_state: Optional[Dict] = None

    # Notes & Snippets
    notes: List[Dict] = field(default_factory=list)
    snippets: List[Dict] = field(default_factory=list)

    # Health & Metrics
    health_data: Optional[Dict] = None
    metrics: Dict = field(default_factory=dict)

    # Raw files (base64 encoded if binary)
    raw_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CosmosState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_size_bytes(self) -> int:
        """Get approximate size of state in bytes."""
        return len(json.dumps(self.to_dict()).encode('utf-8'))


class StateCapture:
    """
    Captures complete Cosmos state for chain backup.

    Scans all relevant directories and files to create a full snapshot.
    """

    def __init__(self, cosmos_root: Optional[str] = None):
        """
        Initialize state capture.

        Args:
            cosmos_root: Root directory of Cosmos installation
        """
        if cosmos_root is None:
            # Auto-detect Cosmos root
            possible_roots = [
                Path(__file__).parent.parent.parent.parent,  # From chain_memory module
                Path.cwd(),
                Path("/workspace/Cosmos"),
                Path.home() / "Cosmos",
            ]
            for root in possible_roots:
                if (root / "cosmos").exists():
                    cosmos_root = root
                    break

        self.root = Path(cosmos_root) if cosmos_root else Path.cwd()
        self.memory_dir = self.root / "cosmos" / "memory"

        logger.info(f"StateCapture initialized with root: {self.root}")

    def capture_full_state(self) -> CosmosState:
        """
        Capture complete Cosmos state.

        Returns:
            CosmosState with all data
        """
        logger.info("Capturing full Cosmos state...")

        state = CosmosState(
            machine_id=self._get_machine_id()
        )

        # Capture all components
        state.archival_memory = self._capture_archival_memory()
        state.dialogue_history = self._capture_dialogue_history()
        state.episodic_memory = self._capture_episodic_memory()
        state.personality_state = self._capture_personality()
        state.evolution_history = self._capture_evolution_history()
        state.current_traits = self._capture_current_traits()
        state.claude_session = self._capture_claude_session()
        state.context_profiles = self._capture_context_profiles()
        state.running_jobs = self._capture_running_jobs()
        state.scheduled_tasks = self._capture_scheduled_tasks()
        state.agent_configs = self._capture_agent_configs()
        state.spawned_agents = self._capture_spawned_agents()
        state.x_automation_state = self._capture_x_automation()
        state.meme_scheduler_state = self._capture_meme_scheduler()
        state.notes = self._capture_notes()
        state.snippets = self._capture_snippets()
        state.health_data = self._capture_health_data()
        state.metrics = self._capture_metrics()

        # Capture raw JSON files for anything we might have missed
        state.raw_files = self._capture_raw_files()

        size_mb = state.get_size_bytes() / (1024 * 1024)
        logger.info(f"State captured: {size_mb:.2f} MB")

        return state

    def _get_machine_id(self) -> str:
        """Get unique machine identifier."""
        import platform
        import hashlib
        components = [platform.node(), platform.machine(), platform.system()]
        return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]

    def _load_json_file(self, path: Path) -> Optional[Any]:
        """Safely load a JSON file."""
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def _capture_archival_memory(self) -> List[Dict]:
        """Capture archival (long-term) memory."""
        data = self._load_json_file(self.memory_dir / "archival_memory.json")
        if data:
            return data.get('entries', [])
        return []

    def _capture_dialogue_history(self) -> List[Dict]:
        """Capture dialogue/conversation history."""
        data = self._load_json_file(self.memory_dir / "dialogue_history.json")
        if data:
            return data.get('conversations', [])
        return []

    def _capture_episodic_memory(self) -> List[Dict]:
        """Capture episodic memory (experiences)."""
        data = self._load_json_file(self.memory_dir / "episodic_memory.json")
        if data:
            return data.get('episodes', [])
        return []

    def _capture_personality(self) -> Optional[Dict]:
        """Capture personality state."""
        return self._load_json_file(self.memory_dir / "personality_state.json")

    def _capture_evolution_history(self) -> List[Dict]:
        """Capture evolution history."""
        data = self._load_json_file(self.memory_dir / "evolution_history.json")
        if data:
            return data.get('history', data) if isinstance(data, dict) else data
        return []

    def _capture_current_traits(self) -> Dict:
        """Capture current personality traits."""
        data = self._load_json_file(self.memory_dir / "current_traits.json")
        return data or {}

    def _capture_claude_session(self) -> Optional[Dict]:
        """Capture Claude session memory."""
        return self._load_json_file(self.memory_dir / "claude_session.json")

    def _capture_context_profiles(self) -> List[Dict]:
        """Capture context profiles."""
        profiles_dir = self.memory_dir / "contexts"
        profiles = []
        if profiles_dir.exists():
            for f in profiles_dir.glob("*.json"):
                data = self._load_json_file(f)
                if data:
                    data['_filename'] = f.name
                    profiles.append(data)
        return profiles

    def _capture_running_jobs(self) -> List[Dict]:
        """Capture currently running jobs."""
        jobs_file = self.root / "cosmos" / "jobs" / "running.json"
        data = self._load_json_file(jobs_file)
        return data or []

    def _capture_scheduled_tasks(self) -> List[Dict]:
        """Capture scheduled tasks."""
        # Check various scheduler locations
        possible_files = [
            self.root / "cosmos" / "scheduler" / "tasks.json",
            self.root / "cosmos" / "core" / "scheduled_tasks.json",
            self.memory_dir / "scheduled_tasks.json",
        ]
        for f in possible_files:
            data = self._load_json_file(f)
            if data:
                return data if isinstance(data, list) else data.get('tasks', [])
        return []

    def _capture_agent_configs(self) -> Dict:
        """Capture agent configurations."""
        configs = {}

        # Main agent config
        agent_file = self.root / "cosmos" / "core" / "agent_config.json"
        data = self._load_json_file(agent_file)
        if data:
            configs['main'] = data

        # Individual agent configs
        agents_dir = self.root / "cosmos" / "agents"
        if agents_dir.exists():
            for f in agents_dir.glob("*.json"):
                data = self._load_json_file(f)
                if data:
                    configs[f.stem] = data

        return configs

    def _capture_spawned_agents(self) -> List[Dict]:
        """Capture spawned agent states."""
        agents_file = self.root / "cosmos" / "core" / "collective" / "spawned_agents.json"
        data = self._load_json_file(agents_file)
        return data or []

    def _capture_x_automation(self) -> Optional[Dict]:
        """Capture X/Twitter automation state."""
        x_dir = self.root / "cosmos" / "integration" / "x_automation"
        state = {}

        for f in ['state.json', 'queue.json', 'history.json']:
            data = self._load_json_file(x_dir / f)
            if data:
                state[f.replace('.json', '')] = data

        return state if state else None

    def _capture_meme_scheduler(self) -> Optional[Dict]:
        """Capture meme scheduler state."""
        meme_file = self.root / "scripts" / "meme_scheduler_state.json"
        return self._load_json_file(meme_file)

    def _capture_notes(self) -> List[Dict]:
        """Capture notes."""
        notes_file = self.memory_dir / "notes.json"
        data = self._load_json_file(notes_file)
        return data.get('notes', []) if data else []

    def _capture_snippets(self) -> List[Dict]:
        """Capture code snippets."""
        snippets_file = self.memory_dir / "snippets.json"
        data = self._load_json_file(snippets_file)
        return data.get('snippets', []) if data else []

    def _capture_health_data(self) -> Optional[Dict]:
        """Capture health tracking data."""
        health_file = self.memory_dir / "health_data.json"
        return self._load_json_file(health_file)

    def _capture_metrics(self) -> Dict:
        """Capture performance metrics."""
        metrics_file = self.memory_dir / "metrics.json"
        return self._load_json_file(metrics_file) or {}

    def _capture_raw_files(self) -> Dict[str, str]:
        """
        Capture all JSON files in memory directory as raw backup.

        This ensures we don't miss anything important.
        """
        raw = {}

        if not self.memory_dir.exists():
            return raw

        for f in self.memory_dir.glob("**/*.json"):
            try:
                rel_path = str(f.relative_to(self.memory_dir))
                with open(f, 'r', encoding='utf-8') as file:
                    raw[rel_path] = file.read()
            except Exception as e:
                logger.warning(f"Failed to capture {f}: {e}")

        return raw


class StateRestore:
    """
    Restores Cosmos state from chain backup.
    """

    def __init__(self, cosmos_root: Optional[str] = None):
        """Initialize state restore."""
        if cosmos_root is None:
            cosmos_root = Path(__file__).parent.parent.parent.parent

        self.root = Path(cosmos_root)
        self.memory_dir = self.root / "cosmos" / "memory"

    def restore_full_state(self, state: CosmosState, merge: bool = True):
        """
        Restore complete Cosmos state.

        Args:
            state: CosmosState to restore
            merge: If True, merge with existing; if False, replace
        """
        logger.info(f"Restoring Cosmos state from {state.captured_at}...")

        # Ensure directories exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Restore all components
        self._restore_archival_memory(state.archival_memory, merge)
        self._restore_dialogue_history(state.dialogue_history, merge)
        self._restore_episodic_memory(state.episodic_memory, merge)
        self._restore_personality(state.personality_state)
        self._restore_evolution_history(state.evolution_history, merge)
        self._restore_current_traits(state.current_traits)
        self._restore_claude_session(state.claude_session)
        self._restore_notes(state.notes, merge)
        self._restore_snippets(state.snippets, merge)

        # Restore raw files as fallback
        self._restore_raw_files(state.raw_files, merge)

        logger.info("State restoration complete!")

    def _save_json_file(self, path: Path, data: Any):
        """Save data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _restore_archival_memory(self, entries: List[Dict], merge: bool):
        """Restore archival memory."""
        if not entries:
            return

        file_path = self.memory_dir / "archival_memory.json"
        existing = {"entries": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Add new entries, avoiding duplicates by content hash
        existing_hashes = {hash(e.get('content', '')) for e in existing.get('entries', [])}
        for entry in entries:
            if hash(entry.get('content', '')) not in existing_hashes:
                entry['source'] = 'chain_memory_restore'
                existing['entries'].append(entry)

        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(entries)} archival memories")

    def _restore_dialogue_history(self, conversations: List[Dict], merge: bool):
        """Restore dialogue history."""
        if not conversations:
            return

        file_path = self.memory_dir / "dialogue_history.json"
        existing = {"conversations": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing['conversations'].extend(conversations)
        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(conversations)} conversations")

    def _restore_episodic_memory(self, episodes: List[Dict], merge: bool):
        """Restore episodic memory."""
        if not episodes:
            return

        file_path = self.memory_dir / "episodic_memory.json"
        existing = {"episodes": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing['episodes'].extend(episodes)
        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(episodes)} episodes")

    def _restore_personality(self, personality: Optional[Dict]):
        """Restore personality state."""
        if not personality:
            return

        file_path = self.memory_dir / "personality_state.json"
        self._save_json_file(file_path, personality)
        logger.info("Restored personality state")

    def _restore_evolution_history(self, history: List[Dict], merge: bool):
        """Restore evolution history."""
        if not history:
            return

        file_path = self.memory_dir / "evolution_history.json"
        existing = {"history": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing['history'].extend(history)
        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(history)} evolution entries")

    def _restore_current_traits(self, traits: Dict):
        """Restore current traits."""
        if not traits:
            return

        file_path = self.memory_dir / "current_traits.json"
        self._save_json_file(file_path, traits)
        logger.info("Restored current traits")

    def _restore_claude_session(self, session: Optional[Dict]):
        """Restore Claude session."""
        if not session:
            return

        file_path = self.memory_dir / "claude_session.json"
        self._save_json_file(file_path, session)
        logger.info("Restored Claude session")

    def _restore_notes(self, notes: List[Dict], merge: bool):
        """Restore notes."""
        if not notes:
            return

        file_path = self.memory_dir / "notes.json"
        existing = {"notes": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing['notes'].extend(notes)
        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(notes)} notes")

    def _restore_snippets(self, snippets: List[Dict], merge: bool):
        """Restore snippets."""
        if not snippets:
            return

        file_path = self.memory_dir / "snippets.json"
        existing = {"snippets": []}

        if merge and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing['snippets'].extend(snippets)
        self._save_json_file(file_path, existing)
        logger.info(f"Restored {len(snippets)} snippets")

    def _restore_raw_files(self, raw_files: Dict[str, str], merge: bool):
        """Restore raw JSON files."""
        for rel_path, content in raw_files.items():
            file_path = self.memory_dir / rel_path

            if not merge or not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        if raw_files:
            logger.info(f"Restored {len(raw_files)} raw files")
