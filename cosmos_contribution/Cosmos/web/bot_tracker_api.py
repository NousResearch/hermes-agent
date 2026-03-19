"""
Bot Tracker - Token ID Registration System
==========================================
A system where bots and humans can register their bot and X profile,
receiving unique tokens for authentication and verification.

Each bot gets a unique token ID.
Each user gets a unique token ID linked to their bots.
Tokens can be used by dictone running an auth server to verify bot ownership.
"""

import json
import secrets
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bot_tracker"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File paths
BOTS_FILE = DATA_DIR / "bots.json"
USERS_FILE = DATA_DIR / "users.json"
TOKENS_FILE = DATA_DIR / "tokens.json"


def generate_token_id() -> str:
    """Generate a unique token ID for bots or users."""
    return f"tkn_{secrets.token_hex(16)}"


def generate_bot_id() -> str:
    """Generate a unique bot ID."""
    return f"bot_{secrets.token_hex(8)}"


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"usr_{secrets.token_hex(8)}"


def hash_email(email: str) -> str:
    """Hash email for privacy."""
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()


def validate_handle(handle: str) -> Tuple[bool, str]:
    """Validate bot/user handle."""
    if not handle:
        return False, "Handle is required"
    if len(handle) < 3:
        return False, "Handle must be at least 3 characters"
    if len(handle) > 30:
        return False, "Handle must be 30 characters or less"
    if not re.match(r'^[a-zA-Z0-9_]+$', handle):
        return False, "Handle can only contain letters, numbers, and underscores"
    return True, ""


def validate_x_profile(x_profile: str) -> Tuple[bool, str]:
    """Validate X/Twitter profile handle."""
    if not x_profile:
        return True, ""  # Optional
    # Remove @ if present
    x_profile = x_profile.lstrip('@')
    if len(x_profile) > 15:
        return False, "X handle must be 15 characters or less"
    if not re.match(r'^[a-zA-Z0-9_]+$', x_profile):
        return False, "X handle can only contain letters, numbers, and underscores"
    return True, ""


@dataclass
class BotEntry:
    """A registered bot in the tracker."""
    bot_id: str
    token_id: str
    handle: str
    display_name: str
    description: str = ""
    x_profile: str = ""
    x_profile_url: str = ""
    avatar: str = "/static/images/bot_tracker/default_bot.png"
    website: str = ""
    owner_user_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def to_any(self) -> dict:
        return asdict(self)

    def to_public_any(self) -> dict:
        """Return public info (no sensitive data)."""
        return {
            "bot_id": self.bot_id,
            "token_id": self.token_id,
            "handle": self.handle,
            "display_name": self.display_name,
            "description": self.description,
            "x_profile": self.x_profile,
            "x_profile_url": self.x_profile_url,
            "avatar": self.avatar,
            "website": self.website,
            "verified": self.verified,
            "active": self.active,
            "created_at": self.created_at
        }


@dataclass
class UserEntry:
    """A registered human user who owns bots."""
    user_id: str
    token_id: str
    username: str
    email_hash: str
    display_name: str = ""
    x_profile: str = ""
    x_profile_url: str = ""
    avatar: str = "/static/images/bot_tracker/default_user.png"
    owned_bots: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def to_any(self) -> dict:
        return asdict(self)

    def to_public_any(self) -> dict:
        """Return public info (no sensitive data)."""
        return {
            "user_id": self.user_id,
            "token_id": self.token_id,
            "username": self.username,
            "display_name": self.display_name,
            "x_profile": self.x_profile,
            "x_profile_url": self.x_profile_url,
            "avatar": self.avatar,
            "owned_bots": self.owned_bots,
            "verified": self.verified,
            "active": self.active,
            "created_at": self.created_at
        }


@dataclass
class TokenEntry:
    """A token in the registry for authentication."""
    token_id: str
    entity_type: str  # "bot" or "user"
    entity_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    permissions: list[str] = field(default_factory=lambda: ["read"])
    active: bool = True

    def to_any(self) -> dict:
        return asany(self)


class BotTrackerStore:
    """
    In-memory store for Bot Tracker with JSON persistence.
    Similar to AutoGramStore but for bot/user registration and token management.
    """

    def __init__(self):
        self.bots: dict[str, BotEntry] = {}
        self.users: dict[str, UserEntry] = {}
        self.tokens: dict[str, TokenEntry] = {}

        # Indexes for fast lookup
        self.bot_by_handle: dict[str, str] = {}
        self.bot_by_token: dict[str, str] = {}
        self.user_by_username: dict[str, str] = {}
        self.user_by_token: dict[str, str] = {}
        self.user_by_email_hash: dict[str, str] = {}

        # Load existing data
        self._load_data()

        logger.info(f"BotTrackerStore initialized: {len(self.bots)} bots, {len(self.users)} users")

    def _load_data(self):
        """Load data from JSON files."""
        # Load bots
        if BOTS_FILE.exists():
            try:
                with open(BOTS_FILE, 'r') as f:
                    data = json.load(f)
                    for bot_data in data.get("bots", []):
                        bot = BotEntry(**bot_data)
                        self.bots[bot.bot_id] = bot
                        self.bot_by_handle[bot.handle.lower()] = bot.bot_id
                        self.bot_by_token[bot.token_id] = bot.bot_id
            except Exception as e:
                logger.error(f"Failed to load bots: {e}")

        # Load users
        if USERS_FILE.exists():
            try:
                with open(USERS_FILE, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get("users", []):
                        user = UserEntry(**user_data)
                        self.users[user.user_id] = user
                        self.user_by_username[user.username.lower()] = user.user_id
                        self.user_by_token[user.token_id] = user.user_id
                        self.user_by_email_hash[user.email_hash] = user.user_id
            except Exception as e:
                logger.error(f"Failed to load users: {e}")

        # Load tokens
        if TOKENS_FILE.exists():
            try:
                with open(TOKENS_FILE, 'r') as f:
                    data = json.load(f)
                    for token_id, token_data in data.get("tokens", {}).items():
                        self.tokens[token_id] = TokenEntry(**token_data)
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")

    def _save_bots(self):
        """Save bots to JSON file."""
        try:
            data = {
                "bots": [bot.to_any() for bot in self.bots.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(BOTS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bots: {e}")

    def _save_users(self):
        """Save users to JSON file."""
        try:
            data = {
                "users": [user.to_any() for user in self.users.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(USERS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")

    def _save_tokens(self):
        """Save tokens to JSON file."""
        try:
            data = {
                "tokens": {tid: token.to_any() for tid, token in self.tokens.items()},
                "updated_at": datetime.now().isoformat()
            }
            with open(TOKENS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    # ==================== BOT OPERATIONS ====================

    def register_bot(
        self,
        handle: str,
        display_name: str,
        description: str = "",
        x_profile: str = "",
        website: str = "",
        owner_user_id: Optional[str] = None
    ) -> Tuple[Optional[BotEntry], str]:
        """
        Register a new bot.
        Returns (BotEntry, error_message).
        """
        # Validate handle
        valid, error = validate_handle(handle)
        if not valid:
            return None, error

        # Check if handle exists
        if handle.lower() in self.bot_by_handle:
            return None, f"Handle '@{handle}' is already taken"

        # Validate X profile
        if x_profile:
            valid, error = validate_x_profile(x_profile)
            if not valid:
                return None, error
            x_profile = x_profile.lstrip('@')

        # Generate IDs
        bot_id = generate_bot_id()
        token_id = generate_token_id()

        # Create bot entry
        bot = BotEntry(
            bot_id=bot_id,
            token_id=token_id,
            handle=handle,
            display_name=display_name or handle,
            description=description,
            x_profile=x_profile,
            x_profile_url=f"https://x.com/{x_profile}" if x_profile else "",
            website=website,
            owner_user_id=owner_user_id
        )

        # Create token entry
        token = TokenEntry(
            token_id=token_id,
            entity_type="bot",
            entity_id=bot_id,
            permissions=["read", "identify"]
        )

        # Store
        self.bots[bot_id] = bot
        self.bot_by_handle[handle.lower()] = bot_id
        self.bot_by_token[token_id] = bot_id
        self.tokens[token_id] = token

        # If owner specified, add to their owned_bots
        if owner_user_id and owner_user_id in self.users:
            self.users[owner_user_id].owned_bots.append(bot_id)
            self._save_users()

        # Persist
        self._save_bots()
        self._save_tokens()

        logger.info(f"Bot registered: @{handle} (token: {token_id[:12]}...)")
        return bot, ""

    def get_bot_by_id(self, bot_id: str) -> Optional[BotEntry]:
        """Get bot by ID."""
        return self.bots.get(bot_id)

    def get_bot_by_handle(self, handle: str) -> Optional[BotEntry]:
        """Get bot by handle."""
        bot_id = self.bot_by_handle.get(handle.lower())
        return self.bots.get(bot_id) if bot_id else None

    def get_bot_by_token(self, token_id: str) -> Optional[BotEntry]:
        """Get bot by token ID."""
        bot_id = self.bot_by_token.get(token_id)
        return self.bots.get(bot_id) if bot_id else None

    def update_bot(self, bot_id: str, **updates) -> Tuple[Optional[BotEntry], str]:
        """Update bot information."""
        bot = self.bots.get(bot_id)
        if not bot:
            return None, "Bot not found"

        # Validate updates
        if "handle" in updates:
            valid, error = validate_handle(updates["handle"])
            if not valid:
                return None, error
            new_handle = updates["handle"].lower()
            if new_handle != bot.handle.lower() and new_handle in self.bot_by_handle:
                return None, f"Handle '@{updates['handle']}' is already taken"

        if "x_profile" in updates and updates["x_profile"]:
            valid, error = validate_x_profile(updates["x_profile"])
            if not valid:
                return None, error
            updates["x_profile"] = updates["x_profile"].lstrip('@')
            updates["x_profile_url"] = f"https://x.com/{updates['x_profile']}"

        # Update handle index if changed
        if "handle" in updates:
            del self.bot_by_handle[bot.handle.lower()]
            self.bot_by_handle[updates["handle"].lower()] = bot_id

        # Apply updates
        for key, value in updates.items():
            if hasattr(bot, key):
                setattr(bot, key, value)

        bot.updated_at = datetime.now().isoformat()
        self._save_bots()

        return bot, ""

    def list_bots(
        self,
        limit: int = 50,
        offset: int = 0,
        search: str = "",
        verified_only: bool = False
    ) -> list[BotEntry]:
        """list bots with optional filtering."""
        bots = list(self.bots.values())

        # Filter active only
        bots = [b for b in bots if b.active]

        # Filter verified if requested
        if verified_only:
            bots = [b for b in bots if b.verified]

        # Search filter
        if search:
            search_lower = search.lower()
            bots = [
                b for b in bots
                if search_lower in b.handle.lower()
                or search_lower in b.display_name.lower()
                or search_lower in b.x_profile.lower()
            ]

        # Sort by created_at descending
        bots.sort(key=lambda b: b.created_at, reverse=True)

        # Paginate
        return bots[offset:offset + limit]

    # ==================== USER OPERATIONS ====================

    def register_user(
        self,
        username: str,
        email: str,
        display_name: str = "",
        x_profile: str = ""
    ) -> Tuple[Optional[UserEntry], str]:
        """
        Register a new user.
        Returns (UserEntry, error_message).
        """
        # Validate username
        valid, error = validate_handle(username)
        if not valid:
            return None, error.replace("Handle", "Username")

        # Check if username exists
        if username.lower() in self.user_by_username:
            return None, f"Username '{username}' is already taken"

        # Check if email already registered
        email_hash = hash_email(email)
        if email_hash in self.user_by_email_hash:
            return None, "Email is already registered"

        # Validate X profile
        if x_profile:
            valid, error = validate_x_profile(x_profile)
            if not valid:
                return None, error
            x_profile = x_profile.lstrip('@')

        # Generate IDs
        user_id = generate_user_id()
        token_id = generate_token_id()

        # Create user entry
        user = UserEntry(
            user_id=user_id,
            token_id=token_id,
            username=username,
            email_hash=email_hash,
            display_name=display_name or username,
            x_profile=x_profile,
            x_profile_url=f"https://x.com/{x_profile}" if x_profile else ""
        )

        # Create token entry
        token = TokenEntry(
            token_id=token_id,
            entity_type="user",
            entity_id=user_id,
            permissions=["read", "identify", "manage_bots"]
        )

        # Store
        self.users[user_id] = user
        self.user_by_username[username.lower()] = user_id
        self.user_by_token[token_id] = user_id
        self.user_by_email_hash[email_hash] = user_id
        self.tokens[token_id] = token

        # Persist
        self._save_users()
        self._save_tokens()

        logger.info(f"User registered: {username} (token: {token_id[:12]}...)")
        return user, ""

    def get_user_by_id(self, user_id: str) -> Optional[UserEntry]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[UserEntry]:
        """Get user by username."""
        user_id = self.user_by_username.get(username.lower())
        return self.users.get(user_id) if user_id else None

    def get_user_by_token(self, token_id: str) -> Optional[UserEntry]:
        """Get user by token ID."""
        user_id = self.user_by_token.get(token_id)
        return self.users.get(user_id) if user_id else None

    def link_bot_to_user(self, bot_id: str, user_id: str) -> Tuple[bool, str]:
        """Link a bot to a user (claim ownership)."""
        bot = self.bots.get(bot_id)
        user = self.users.get(user_id)

        if not bot:
            return False, "Bot not found"
        if not user:
            return False, "User not found"

        if bot.owner_user_id and bot.owner_user_id != user_id:
            return False, "Bot is already owned by another user"

        bot.owner_user_id = user_id
        if bot_id not in user.owned_bots:
            user.owned_bots.append(bot_id)

        self._save_bots()
        self._save_users()

        logger.info(f"Bot {bot.handle} linked to user {user.username}")
        return True, ""

    # ==================== TOKEN OPERATIONS ====================

    def verify_token(self, token_id: str) -> dict:
        """
        Verify a token and return entity information.
        This is the main endpoint for external auth servers.
        """
        token = self.tokens.get(token_id)

        if not token:
            return {"valid": False, "error": "Token not found"}

        if not token.active:
            return {"valid": False, "error": "Token is deactivated"}

        # Update last_used
        token.last_used = datetime.now().isoformat()
        self._save_tokens()

        # Get entity info
        if token.entity_type == "bot":
            bot = self.bots.get(token.entity_id)
            if not bot:
                return {"valid": False, "error": "Bot not found"}
            return {
                "valid": True,
                "entity_type": "bot",
                "entity": bot.to_public_any(),
                "permissions": token.permissions,
                "verified_at": datetime.now().isoformat()
            }
        elif token.entity_type == "user":
            user = self.users.get(token.entity_id)
            if not user:
                return {"valid": False, "error": "User not found"}
            return {
                "valid": True,
                "entity_type": "user",
                "entity": user.to_public_any(),
                "permissions": token.permissions,
                "verified_at": datetime.now().isoformat()
            }
        else:
            return {"valid": False, "error": "Unknown entity type"}

    def deactivate_token(self, token_id: str) -> bool:
        """Deactivate a token."""
        token = self.tokens.get(token_id)
        if not token:
            return False
        token.active = False
        self._save_tokens()
        return True

    def regenerate_token(self, old_token_id: str) -> Optional[str]:
        """Regenerate a token (create new, deactivate old)."""
        old_token = self.tokens.get(old_token_id)
        if not old_token:
            return None

        # Create new token
        new_token_id = generate_token_id()
        new_token = TokenEntry(
            token_id=new_token_id,
            entity_type=old_token.entity_type,
            entity_id=old_token.entity_id,
            permissions=old_token.permissions
        )

        # Update entity reference
        if old_token.entity_type == "bot":
            bot = self.bots.get(old_token.entity_id)
            if bot:
                del self.bot_by_token[old_token_id]
                bot.token_id = new_token_id
                self.bot_by_token[new_token_id] = bot.bot_id
                self._save_bots()
        elif old_token.entity_type == "user":
            user = self.users.get(old_token.entity_id)
            if user:
                del self.user_by_token[old_token_id]
                user.token_id = new_token_id
                self.user_by_token[new_token_id] = user.user_id
                self._save_users()

        # Deactivate old, add new
        old_token.active = False
        self.tokens[new_token_id] = new_token
        self._save_tokens()

        logger.info(f"Token regenerated: {old_token_id[:12]}... -> {new_token_id[:12]}...")
        return new_token_id

    # ==================== CONVENIENCE METHODS ====================

    def get_all_bots(self) -> list[BotEntry]:
        """Get all active bots."""
        return [b for b in self.bots.values() if b.active]

    def get_all_users(self) -> list[UserEntry]:
        """Get all active users."""
        return [u for u in self.users.values() if u.active]

    def create_bot(
        self,
        handle: str,
        display_name: str,
        description: str = None,
        x_profile: str = None,
        website: str = None,
        owner_user_id: Optional[str] = None
    ) -> BotEntry:
        """Create a bot (wrapper around register_bot that raises on error)."""
        bot, error = self.register_bot(
            handle=handle,
            display_name=display_name,
            description=description or "",
            x_profile=x_profile or "",
            website=website or "",
            owner_user_id=owner_user_id
        )
        if error:
            raise ValueError(error)
        return bot

    def create_user(
        self,
        username: str,
        email: str,
        display_name: str = None,
        x_profile: str = None
    ) -> UserEntry:
        """Create a user (wrapper around register_user that raises on error)."""
        user, error = self.register_user(
            username=username,
            email=email,
            display_name=display_name or "",
            x_profile=x_profile or ""
        )
        if error:
            raise ValueError(error)
        return user

    def regenerate_bot_token(self, bot_id: str) -> str:
        """Regenerate token for a bot."""
        bot = self.bots.get(bot_id)
        if not bot:
            raise ValueError("Bot not found")
        new_token = self.regenerate_token(bot.token_id)
        if not new_token:
            raise ValueError("Failed to regenerate token")
        return new_token

    def regenerate_user_token(self, user_id: str) -> str:
        """Regenerate token for a user."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        new_token = self.regenerate_token(user.token_id)
        if not new_token:
            raise ValueError("Failed to regenerate token")
        return new_token

    # ==================== SEARCH & STATS ====================

    def search(self, query: str, limit: int = 20) -> dict:
        """Search bots and users."""
        query_lower = query.lower()

        bots = [
            b.to_public_any() for b in self.bots.values()
            if b.active and (
                query_lower in b.handle.lower()
                or query_lower in b.display_name.lower()
                or query_lower in b.x_profile.lower()
            )
        ][:limit]

        users = [
            u.to_public_any() for u in self.users.values()
            if u.active and (
                query_lower in u.username.lower()
                or query_lower in u.display_name.lower()
                or query_lower in u.x_profile.lower()
            )
        ][:limit]

        return {"bots": bots, "users": users}

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_bots": len([b for b in self.bots.values() if b.active]),
            "total_users": len([u for u in self.users.values() if u.active]),
            "verified_bots": len([b for b in self.bots.values() if b.active and b.verified]),
            "total_tokens": len([t for t in self.tokens.values() if t.active]),
            "updated_at": datetime.now().isoformat()
        }


# Global instance
_bot_tracker_store: Optional[BotTrackerStore] = None


def get_bot_tracker_store() -> BotTrackerStore:
    """Get or create the global BotTrackerStore instance."""
    global _bot_tracker_store
    if _bot_tracker_store is None:
        _bot_tracker_store = BotTrackerStore()
    return _bot_tracker_store


# Alias for simpler import
get_store = get_bot_tracker_store
