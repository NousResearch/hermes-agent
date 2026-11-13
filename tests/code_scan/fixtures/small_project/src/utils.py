import json


def load_data(filepath: str) -> dict:
    """Load JSON from a file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_data(data: dict, filepath: str) -> None:
    """Save dict as JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
