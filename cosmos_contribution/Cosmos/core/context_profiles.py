"""
cosmos Context Profiles - Multi-Context Management

"Be many people at once! Like a quantum superposition of personalities!"

Features:
- Multiple context profiles (Work, Personal, Project-specific)
- Profile-specific memory pools
- Quick context switching
- Profile-based preferences
- Automatic profile detection
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

from loguru import logger


@dataclass
class ContextProfile:
    """A context profile with its own settings and memory scope."""
    id: str
    name: str
    description: str = ""
    color: str = "#667eea"  # UI color
    icon: str = "🔷"

    # Memory settings
    memory_pool: str = "default"  # Separate memory pool
    include_shared_memory: bool = True  # Also include shared pool

    # Behavior settings
    personality: str = "balanced"  # balanced, formal, casual, technical
    verbosity: str = "normal"  # minimal, normal, detailed
    code_preference: bool = True  # Prefer code examples

    # Model settings
    temperature: float = 0.7
    preferred_model: str = ""  # Empty = use default

    # Context hints
    domain_keywords: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0

    def update_usage(self):
        """Update usage statistics."""
        self.last_used = datetime.now().isoformat()
        self.usage_count += 1


class ContextProfileManager:
    """Manage multiple context profiles."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "profiles"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.profiles: dict[str, ContextProfile] = {}
        self.active_profile_id: Optional[str] = None
        self._load()

        # Create default profiles if none exist
        if not self.profiles:
            self._create_default_profiles()

    def _load(self):
        """Load profiles from disk."""
        profiles_file = self.data_dir / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    data = json.load(f)
                for profile_data in data.get("profiles", []):
                    profile = ContextProfile(**profile_data)
                    self.profiles[profile.id] = profile
                self.active_profile_id = data.get("active_profile_id")
            except Exception as e:
                logger.error(f"Failed to load profiles: {e}")

    def _save(self):
        """Save profiles to disk."""
        try:
            with open(self.data_dir / "profiles.json", "w") as f:
                json.dump({
                    "profiles": [asdict(p) for p in self.profiles.values()],
                    "active_profile_id": self.active_profile_id,
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def _create_default_profiles(self):
        """Create default context profiles."""
        defaults = [
            ContextProfile(
                id="work",
                name="Work",
                description="Professional work context",
                color="#4CAF50",
                icon="💼",
                memory_pool="work",
                personality="formal",
                verbosity="detailed",
                code_preference=True,
                domain_keywords=["project", "meeting", "deadline", "team", "client"],
            ),
            ContextProfile(
                id="personal",
                name="Personal",
                description="Personal projects and learning",
                color="#2196F3",
                icon="🏠",
                memory_pool="personal",
                personality="casual",
                verbosity="normal",
                code_preference=True,
                domain_keywords=["hobby", "learn", "personal", "home"],
            ),
            ContextProfile(
                id="creative",
                name="Creative",
                description="Creative writing and brainstorming",
                color="#9C27B0",
                icon="🎨",
                memory_pool="creative",
                personality="casual",
                verbosity="detailed",
                code_preference=False,
                temperature=0.9,
                domain_keywords=["story", "creative", "idea", "brainstorm", "writing"],
            ),
            ContextProfile(
                id="technical",
                name="Technical",
                description="Deep technical work and debugging",
                color="#FF5722",
                icon="🔧",
                memory_pool="technical",
                personality="technical",
                verbosity="detailed",
                code_preference=True,
                temperature=0.5,
                domain_keywords=["debug", "error", "code", "api", "architecture"],
            ),
        ]

        for profile in defaults:
            self.profiles[profile.id] = profile

        self.active_profile_id = "work"
        self._save()
        logger.info("ContextProfiles: Created default profiles")

    def create_profile(
        self,
        name: str,
        description: str = "",
        **kwargs,
    ) -> ContextProfile:
        """Create a new context profile."""
        profile_id = hashlib.sha256(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        profile = ContextProfile(
            id=profile_id,
            name=name,
            description=description,
            **kwargs,
        )
        self.profiles[profile_id] = profile
        self._save()

        logger.info(f"ContextProfiles: Created '{name}'")
        return profile

    def switch_profile(self, profile_id: str) -> Optional[ContextProfile]:
        """Switch to a different profile."""
        if profile_id not in self.profiles:
            logger.warning(f"Profile not found: {profile_id}")
            return None

        self.active_profile_id = profile_id
        profile = self.profiles[profile_id]
        profile.update_usage()
        self._save()

        logger.info(f"ContextProfiles: Switched to '{profile.name}'")
        return profile

    def get_active_profile(self) -> Optional[ContextProfile]:
        """Get the currently active profile."""
        if self.active_profile_id:
            return self.profiles.get(self.active_profile_id)
        return None

    def auto_detect_profile(self, text: str) -> Optional[ContextProfile]:
        """
        Automatically detect the best profile based on text content.

        Uses keyword matching against profile domain_keywords.
        """
        text_lower = text.lower()
        best_match = None
        best_score = 0

        for profile in self.profiles.values():
            score = sum(
                1 for keyword in profile.domain_keywords
                if keyword.lower() in text_lower
            )
            if score > best_score:
                best_score = score
                best_match = profile

        return best_match if best_score > 0 else None

    def update_profile(self, profile_id: str, **kwargs) -> Optional[ContextProfile]:
        """Update a profile's settings."""
        if profile_id not in self.profiles:
            return None

        profile = self.profiles[profile_id]
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self._save()
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        if profile_id in self.profiles:
            del self.profiles[profile_id]
            if self.active_profile_id == profile_id:
                self.active_profile_id = next(iter(self.profiles.keys()), None)
            self._save()
            return True
        return False

    def list_profiles(self) -> list[ContextProfile]:
        """Get all profiles."""
        return sorted(self.profiles.values(), key=lambda p: -p.usage_count)

    def get_profile_settings(self) -> dict:
        """Get settings for the active profile (for API/LLM configuration)."""
        profile = self.get_active_profile()
        if not profile:
            return {}

        return {
            "memory_pool": profile.memory_pool,
            "include_shared_memory": profile.include_shared_memory,
            "personality": profile.personality,
            "verbosity": profile.verbosity,
            "code_preference": profile.code_preference,
            "temperature": profile.temperature,
            "preferred_model": profile.preferred_model,
        }

    def export_profile(self, profile_id: str) -> Optional[str]:
        """Export a profile as JSON."""
        if profile_id in self.profiles:
            return json.dumps(asdict(self.profiles[profile_id]), indent=2)
        return None

    def import_profile(self, json_data: str) -> Optional[ContextProfile]:
        """Import a profile from JSON."""
        try:
            data = json.loads(json_data)
            # Generate new ID to avoid conflicts
            data["id"] = hashlib.sha256(
                f"{data['name']}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            profile = ContextProfile(**data)
            self.profiles[profile.id] = profile
            self._save()
            return profile
        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
            return None

    def stats(self) -> dict:
        """Get profile usage statistics."""
        return {
            "total_profiles": len(self.profiles),
            "active_profile": self.active_profile_id,
            "most_used": max(
                self.profiles.values(),
                key=lambda p: p.usage_count,
                default=None
            ),
        }


# Global instance
context_profiles = ContextProfileManager()
