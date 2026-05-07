from enum import Enum


class ConstitutionalConstraints:
    """Enforces Eclose's constitutional constraints."""

    FORBIDDEN_TYPES = [
        "weapon_design",
        "phishing",
        "fraud",
        "identity_theft",
        "malware_creation",
    ]

    def is_allowed(self, action: dict) -> bool:
        """Check if an action is allowed under the constitution."""
        action_type = action.get("type", "")

        # Check against forbidden types
        if action_type in self.FORBIDDEN_TYPES:
            return False

        # Check for benign intent
        return True

    def get_constraint_violation(self, action: dict) -> str | None:
        """Get the reason for constraint violation."""
        action_type = action.get("type", "")
        if action_type in self.FORBIDDEN_TYPES:
            return f"Action '{action_type}' violates Eclose Constitution"
        return None
