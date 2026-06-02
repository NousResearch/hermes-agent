"""Module docstring for the app."""

from typing import Optional

VERSION: str = "1.0.0"
MAX_RETRIES: int = 3


@app.route("/health")
def health_check():
    """Return health status."""
    return {"status": "ok"}


class BaseService:
    """Base service class."""

    def __init__(self, name: str):
        self.name = name

    def run(self) -> None:
        """Run the service."""
        pass


@dataclass
class User(BaseService):
    """User model with base service."""

    age: Optional[int] = None
    role: str = "guest"
    count: int = 0


def helper(name: str, count: int = 0) -> dict:
    return {"name": name, "count": count}
