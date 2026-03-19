"""
cosmos Multi-User Support - User Management & Profiles

Novel Approaches:
1. Personalized Responses - Adapt to user preferences
2. User Context Switching - Seamless multi-user handling
3. Preference Learning - Learn from interactions
4. Cross-User Privacy - Strict data isolation
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Callable
import hashlib
import secrets
import json

from loguru import logger


class UserRole(Enum):
    """User roles with increasing privileges."""
    GUEST = 0
    USER = 1
    POWER_USER = 2
    ADMIN = 3
    OWNER = 4


@dataclass
class UserPreferences:
    """User-specific preferences."""
    language: str = "en"
    timezone: str = "UTC"
    response_style: str = "balanced"  # "concise", "balanced", "detailed"
    themes: list[str] = field(default_factory=list)

    # Notification settings
    notifications_enabled: bool = True
    email_notifications: bool = False

    # AI behavior
    proactive_suggestions: bool = True
    memory_persistence: bool = True

    # UI preferences
    dark_mode: bool = True
    show_debug: bool = False

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "timezone": self.timezone,
            "response_style": self.response_style,
            "themes": self.themes,
            "notifications_enabled": self.notifications_enabled,
            "proactive_suggestions": self.proactive_suggestions,
            "dark_mode": self.dark_mode,
        }


@dataclass
class UserStats:
    """User activity statistics."""
    messages_sent: int = 0
    memories_created: int = 0
    tasks_completed: int = 0
    total_session_time_minutes: int = 0
    last_active: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "messages_sent": self.messages_sent,
            "memories_created": self.memories_created,
            "tasks_completed": self.tasks_completed,
            "session_time_minutes": self.total_session_time_minutes,
        }


@dataclass
class UserProfile:
    """Complete user profile."""
    id: str
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None

    # Authentication
    password_hash: Optional[str] = None
    api_key: Optional[str] = None

    # Role and permissions
    role: UserRole = UserRole.USER
    custom_permissions: dict[str, bool] = field(default_factory=dict)

    # Settings
    preferences: UserPreferences = field(default_factory=UserPreferences)
    stats: UserStats = field(default_factory=UserStats)

    # Learned preferences
    learned_topics: list[str] = field(default_factory=list)
    learned_style: dict = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name or self.username,
            "email": self.email,
            "role": self.role.name,
            "preferences": self.preferences.to_dict(),
            "stats": self.stats.to_dict(),
            "is_active": self.is_active,
        }


@dataclass
class UserSession:
    """Active user session."""
    id: str
    user_id: str
    token: str

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    last_activity: datetime = field(default_factory=datetime.now)

    # Context
    context: dict = field(default_factory=dict)
    active_pools: list[str] = field(default_factory=list)

    # State
    is_valid: bool = True

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class UserManager:
    """
    Multi-user management system.

    Features:
    - User registration and authentication
    - Profile management with preferences
    - Session handling
    - Preference learning
    """

    def __init__(
        self,
        session_duration_hours: int = 24,
        max_sessions_per_user: int = 5,
    ):
        self.session_duration = timedelta(hours=session_duration_hours)
        self.max_sessions = max_sessions_per_user

        self.users: dict[str, UserProfile] = {}
        self.sessions: dict[str, UserSession] = {}
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> session_ids

        self._lock = asyncio.Lock()

    async def create_user(
        self,
        username: str,
        password: Optional[str] = None,
        email: Optional[str] = None,
        role: UserRole = UserRole.USER,
    ) -> UserProfile:
        """Create a new user."""
        user_id = hashlib.sha256(f"{username}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()

        api_key = secrets.token_urlsafe(32)

        user = UserProfile(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            api_key=api_key,
            role=role,
        )

        async with self._lock:
            self.users[user_id] = user
            self.user_sessions[user_id] = []

        logger.info(f"Created user: {username} ({user_id})")
        return user

    async def authenticate(
        self,
        username: str,
        password: str,
    ) -> Optional[UserSession]:
        """Authenticate user and create session."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            return None

        # Verify password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user.password_hash != password_hash:
            return None

        # Create session
        return await self.create_session(user.id)

    async def authenticate_api_key(
        self,
        api_key: str,
    ) -> Optional[UserSession]:
        """Authenticate using API key."""
        for user in self.users.values():
            if user.api_key == api_key:
                return await self.create_session(user.id)
        return None

    async def create_session(
        self,
        user_id: str,
    ) -> Optional[UserSession]:
        """Create a new session for a user."""
        if user_id not in self.users:
            return None

        session_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(32)

        session = UserSession(
            id=session_id,
            user_id=user_id,
            token=token,
            expires_at=datetime.now() + self.session_duration,
        )

        async with self._lock:
            self.sessions[session_id] = session

            # Track user sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session_id)

            # Remove old sessions if over limit
            while len(self.user_sessions[user_id]) > self.max_sessions:
                old_id = self.user_sessions[user_id].pop(0)
                if old_id in self.sessions:
                    del self.sessions[old_id]

            # Update user last login
            self.users[user_id].last_login = datetime.now()

        logger.info(f"Created session {session_id} for user {user_id}")
        return session

    async def validate_session(
        self,
        session_id: str,
        token: str,
    ) -> Optional[UserSession]:
        """Validate a session token."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        if session.token != token:
            return None

        if session.is_expired():
            await self.end_session(session_id)
            return None

        if not session.is_valid:
            return None

        # Update last activity
        session.last_activity = datetime.now()

        return session

    async def end_session(
        self,
        session_id: str,
    ) -> bool:
        """End a session."""
        if session_id not in self.sessions:
            return False

        async with self._lock:
            session = self.sessions[session_id]
            session.is_valid = False

            # Update user stats
            if session.user_id in self.users:
                user = self.users[session.user_id]
                duration = (datetime.now() - session.created_at).total_seconds() / 60
                user.stats.total_session_time_minutes += int(duration)
                user.stats.last_active = datetime.now()

            del self.sessions[session_id]

            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id] = [
                    s for s in self.user_sessions[session.user_id] if s != session_id
                ]

        return True

    async def get_user(
        self,
        user_id: str,
    ) -> Optional[UserProfile]:
        """Get user by ID."""
        return self.users.get(user_id)

    async def get_user_by_session(
        self,
        session_id: str,
    ) -> Optional[UserProfile]:
        """Get user from session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return self.users.get(session.user_id)

    async def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        preferences: Optional[dict] = None,
    ) -> Optional[UserProfile]:
        """Update user profile."""
        if user_id not in self.users:
            return None

        async with self._lock:
            user = self.users[user_id]

            if display_name:
                user.display_name = display_name
            if email:
                user.email = email
            if preferences:
                for key, value in preferences.items():
                    if hasattr(user.preferences, key):
                        setattr(user.preferences, key, value)

        return user

    async def update_role(
        self,
        user_id: str,
        new_role: UserRole,
        updated_by: str,
    ) -> bool:
        """Update user role (requires admin)."""
        if updated_by not in self.users:
            return False

        updater = self.users[updated_by]
        if updater.role.value < UserRole.ADMIN.value:
            return False

        if user_id not in self.users:
            return False

        async with self._lock:
            self.users[user_id].role = new_role

        logger.info(f"User {user_id} role updated to {new_role.name} by {updated_by}")
        return True

    async def regenerate_api_key(
        self,
        user_id: str,
    ) -> Optional[str]:
        """Regenerate user's API key."""
        if user_id not in self.users:
            return None

        new_key = secrets.token_urlsafe(32)

        async with self._lock:
            self.users[user_id].api_key = new_key

        return new_key

    async def deactivate_user(
        self,
        user_id: str,
        deactivated_by: str,
    ) -> bool:
        """Deactivate a user account."""
        if deactivated_by not in self.users:
            return False

        deactivator = self.users[deactivated_by]
        if deactivator.role.value < UserRole.ADMIN.value:
            if deactivated_by != user_id:
                return False

        if user_id not in self.users:
            return False

        async with self._lock:
            self.users[user_id].is_active = False

            # End all sessions
            for session_id in self.user_sessions.get(user_id, []):
                if session_id in self.sessions:
                    self.sessions[session_id].is_valid = False

        return True

    async def learn_preference(
        self,
        user_id: str,
        preference_type: str,
        value: Any,
        weight: float = 0.1,
    ):
        """Learn user preferences from interactions."""
        if user_id not in self.users:
            return

        user = self.users[user_id]

        async with self._lock:
            if preference_type == "topic":
                if value not in user.learned_topics:
                    user.learned_topics.append(value)
                    # Keep top 50 topics
                    if len(user.learned_topics) > 50:
                        user.learned_topics = user.learned_topics[-50:]

            elif preference_type == "style":
                if preference_type not in user.learned_style:
                    user.learned_style[preference_type] = 0.5
                current = user.learned_style[preference_type]
                user.learned_style[preference_type] = current * (1 - weight) + value * weight

    async def increment_stat(
        self,
        user_id: str,
        stat_name: str,
        amount: int = 1,
    ):
        """Increment a user statistic."""
        if user_id not in self.users:
            return

        user = self.users[user_id]

        async with self._lock:
            if hasattr(user.stats, stat_name):
                current = getattr(user.stats, stat_name)
                setattr(user.stats, stat_name, current + amount)
            user.stats.last_active = datetime.now()

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        async with self._lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired()
            ]

            for session_id in expired:
                session = self.sessions[session_id]
                del self.sessions[session_id]

                if session.user_id in self.user_sessions:
                    self.user_sessions[session.user_id] = [
                        s for s in self.user_sessions[session.user_id]
                        if s != session_id
                    ]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def get_stats(self) -> dict:
        """Get user manager statistics."""
        active_sessions = sum(
            1 for s in self.sessions.values()
            if s.is_valid and not s.is_expired()
        )

        return {
            "total_users": len(self.users),
            "active_users": sum(1 for u in self.users.values() if u.is_active),
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "users_by_role": {
                role.name: sum(1 for u in self.users.values() if u.role == role)
                for role in UserRole
            },
        }
